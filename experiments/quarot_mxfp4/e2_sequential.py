"""Experimental BF16 capture adapter using LLMC's existing sequential engine."""

import contextlib
import hashlib
import shutil
from pathlib import Path

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from torch.fx import Graph

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_subgraphs
from llmcompressor.utils.helpers import calibration_forward_context

from .diagnose import atomic_json
from .e2_capture import file_hash, read_manifest, read_part

TARGETS = [r"re:.*\.input_layernorm$", "ExpertMLP", "GlmMoeDsaMLP"]


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def capture_prefix(subgraphs, selection):
    """Require observable boundaries; cut immediately after the last selected expert.

    Do not throw from a forward hook: all batches must execute the same prefix.
    Nodes following the cutoff (possibly next layer / LM head) are never executed.
    """
    calls = {}
    for i, subgraph in enumerate(subgraphs):
        for node in subgraph.graph.nodes:
            if node.op == "call_module":
                calls.setdefault(str(node.target), []).append((i, node))
    required = []
    for layer, experts in selection.items():
        base = f"model.layers.{layer}"
        names = [f"{base}.post_attention_layernorm", f"{base}.mlp.gate"]
        names += [f"{base}.mlp.experts.{e}" for e in experts]
        for name in names:
            if len(calls.get(name, [])) != 1:
                raise ValueError(f"capture graph needs exactly one call_module: {name}")
        norm, router = (calls[n][0][0] for n in names[:2])
        if norm != router:
            raise ValueError("norm/router in different subgraphs; capture unsupported")
        required.extend(calls[n][0] for n in names[2:])
    stop = max(i for i, _ in required)
    nodes = list(subgraphs[stop].graph.nodes)
    last = max(nodes.index(n) for i, n in required if i == stop)
    graph, env = Graph(), {}
    for node in nodes[: last + 1]:
        env[node] = graph.node_copy(node, lambda old: env[old])
    graph.output({})
    graph.lint()
    prefix = subgraphs[:stop] + [Subgraph(graph, subgraphs[stop].input_names, set())]
    # Refuse surprising ordering; no layer after the requested last layer or head.
    for subgraph in prefix:
        for node in subgraph.graph.nodes:
            if node.op != "call_module":
                continue
            name = str(node.target)
            if name == "lm_head" or (
                name.startswith("model.layers.")
                and int(name.split(".")[2]) > max(selection)
            ):
                raise ValueError(f"capture prefix executes unwanted module: {name}")
    return prefix


@torch.no_grad()
def collect(model, batches, sample_ids, collector, device=torch.device("cpu")):
    """One original-BF16 pass, with exact actual-dispatch input hash checks."""
    if (
        not batches
        or len(batches) != len(sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("nonempty unique global sample IDs required")
    if get_calibrate_all_experts_flag():
        raise ValueError("moe_calibrate_all_experts must be False")
    if any(b["input_ids"].shape[0] != 1 for b in batches):
        raise ValueError("capture requires batch_size=1")
    model.eval()
    set_onload_device(model, device)
    subgraphs = capture_prefix(
        trace_subgraphs(model, batches[0], TARGETS, DatasetArguments().tracing_ignore),
        collector.selection,
    )
    current = [None]
    pending, expected, verified = {}, {}, set()
    masks = [b["attention_mask"].reshape(-1).bool().to(device) for b in batches]

    def norm_hook(layer):
        def hook(module, args, output):
            key = (layer, current[0])
            if key in pending:
                raise ValueError("duplicate norm hook")
            pending[key] = (
                module,
                args[0].reshape(-1, args[0].shape[-1]),
                output.reshape(-1, output.shape[-1]),
            )

        return hook

    def router_hook(layer):
        def hook(module, args, output):
            batch = current[0]
            norm, hidden, norm_out = pending.pop((layer, batch))
            _, weights, indices = output
            if args[0].reshape_as(norm_out).dtype != norm_out.dtype or not torch.equal(
                args[0].reshape_as(norm_out), norm_out
            ):
                raise ValueError("router input differs from original norm output")
            collector.add(
                layer,
                sample_ids[batch],
                norm,
                hidden,
                norm_out,
                indices,
                weights,
                masks[batch],
            )
            for expert in collector.selection[layer]:
                # LinearExperts2D dispatch is SLOT-major (one_hot.permute(2,1,0)).
                # Include padding here: actual dispatch includes it; collector omits it.
                _, tokens = torch.where(indices.T == expert)
                value = norm_out[tokens]
                expected[(layer, expert, batch)] = (tuple(value.shape), digest(value))

        return hook

    def expert_hook(layer, expert):
        def hook(module, args):
            key = (layer, expert, current[0])
            if key in verified or key not in expected:
                raise ValueError("duplicate/missing actual expert dispatch")
            if (tuple(args[0].shape), digest(args[0])) != expected.pop(key):
                raise ValueError(f"actual expert input alignment failed: {key}")
            verified.add(key)

        return hook

    def weight_hook(layer, expert):
        def hook(module, args, output):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                if key not in collector.weight_hashes:
                    collector.weight_hashes[key] = digest(
                        module.get_submodule(projection).weight
                    )

        return hook

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        for layer, experts in collector.selection.items():
            base = f"model.layers.{layer}"
            for name, hook, pre in [
                (f"{base}.post_attention_layernorm", norm_hook(layer), False),
                (f"{base}.mlp.gate", router_hook(layer), False),
                *[
                    (f"{base}.mlp.experts.{e}", expert_hook(layer, e), True)
                    for e in experts
                ],
            ]:
                module = model.get_submodule(name)
                handle = (
                    module.register_forward_pre_hook(hook)
                    if pre
                    else module.register_forward_hook(hook)
                )
                stack.callback(handle.remove)
            for expert in experts:
                handle = model.get_submodule(
                    f"{base}.mlp.experts.{expert}"
                ).register_forward_hook(weight_hook(layer, expert))
                stack.callback(handle.remove)
        cache = IntermediatesCache.from_dataloader(batches, device, torch.device("cpu"))
        for i, subgraph in enumerate(subgraphs):
            with disable_offloading():
                for batch in range(len(batches)):
                    current[0] = batch
                    inputs = cache.fetch(batch, subgraph.input_names)
                    outputs = subgraph.forward(model, **inputs)
                    if i < len(subgraphs) - 1:
                        cache.update(batch, outputs)
                        cache.delete(batch, subgraph.consumed_names)
            print(
                f"capture subgraph={i+1}/{len(subgraphs)} batches={len(batches)}",
                flush=True,
            )
    required = sum(map(len, collector.selection.values())) * len(batches)
    if pending or expected or len(verified) != required:
        raise ValueError("incomplete norm/router/expert capture verification")
    return dict(
        routing_alignment_verified=True,
        verified_dispatches=len(verified),
        subgraphs=len(subgraphs),
    )


def merge_shards(paths, output, source):
    """Validate/copy one block at a time, then clean only verified temporary shards."""
    output = Path(output).resolve()
    directories = [Path(p).absolute() for p in paths]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("missing/overlapping rank shard paths")
    if any(
        p.is_symlink() or p.resolve().parent != output.parent or p.resolve() == output
        for p in directories
    ):
        raise ValueError(
            "rank shards must be distinct sibling directories of final archive"
        )
    directories = [p.resolve() for p in directories]
    if source.get("world_size", len(directories)) != len(directories):
        raise ValueError("missing rank shards")
    output.mkdir(parents=True, exist_ok=False)
    (output / "captured_activations").mkdir()
    merged = dict(
        status="incomplete",
        format_version=2,
        archive_policy="all_routed_valid_tokens",
        source=source,
        seed=1234,
        experts=[],
        seen_samples=[],
        norm_errors=[],
        norm_epsilon={},
        expert_weight_sha256={},
        gamma_bf16={},
        cleanup_status="pending",
    )
    atomic_json(output / "capture_manifest.json", merged)
    seen, identities, entries, cleanup_files = set(), {}, {}, {}
    seen_sample_ids = set()
    try:
        for rank, directory in enumerate(directories):
            manifest = read_manifest(directory)
            if (
                manifest["status"] != "rank_shard_complete"
                or manifest["source"] != source
            ):
                raise ValueError("incomplete/incompatible capture rank shard")
            if rank == 0:
                merged["selection"] = manifest["selection"]
                entries = {
                    (e["layer"], e["expert"]): dict(
                        layer=e["layer"],
                        expert=e["expert"],
                        available=0,
                        retained=0,
                        parts=[],
                    )
                    for e in manifest["experts"]
                }
                identities = {key: set() for key in entries}
            elif merged["selection"] != manifest["selection"]:
                raise ValueError("incompatible rank selection")
            samples = set(map(tuple, manifest["seen_samples"]))
            sample_ids = {sample for _, sample in samples}
            if seen_sample_ids & sample_ids:
                raise ValueError("overlapping rank sample partitions")
            if samples != {
                (int(layer), sample)
                for layer in manifest["selection"]
                for sample in sample_ids
            }:
                raise ValueError("incomplete rank sample/layer coverage")
            seen_sample_ids |= sample_ids
            seen |= samples
            merged["norm_errors"].extend(manifest["norm_errors"])
            for field in ("expert_weight_sha256", "norm_epsilon", "gamma_bf16"):
                for key, value in manifest[field].items():
                    if key in merged[field] and merged[field][key] != value:
                        raise ValueError(f"rank {field} mismatch")
                    merged[field][key] = value
            allowed = {directory / "capture_manifest.json"}
            for entry in manifest["experts"]:
                key = (entry["layer"], entry["expert"])
                dest = entries[key]
                dest["available"] += entry["available"]
                for part in entry["parts"]:
                    data, positions = read_part(directory, part)
                    if identities[key].intersection(positions):
                        raise ValueError("duplicate token identity across shards")
                    if any((key[0], sample) not in samples for sample, _ in positions):
                        raise ValueError("token outside rank sample partition")
                    gamma = torch.tensor(
                        manifest["gamma_bf16"][str(key[0])], dtype=torch.bfloat16
                    )
                    if not torch.equal(data["gamma"], gamma):
                        raise ValueError("archived gamma mismatch")
                    identities[key].update(positions)
                    src = (directory / part["file"]).resolve()
                    allowed.add(src)
                    relative = (
                        f"captured_activations/L{key[0]}-E{key[1]}-"
                        f"P{len(dest['parts'])}.safetensors"
                    )
                    target = output / relative
                    shutil.copyfile(src, target)
                    if file_hash(target) != part["sha256"]:
                        raise ValueError("merged archive copy hash mismatch")
                    dest["parts"].append(dict(part, file=relative))
                    dest["retained"] += len(positions)
                    del data
            # Exact owned file set, no symlinks/junctions or unrelated files. This
            # is also checked again immediately before recursive cleanup.
            cleanup_files[directory] = allowed
            _check_cleanup_tree(directory, allowed, output)
        if "global_sample_ids" in source:
            expected = {
                (int(layer), sample)
                for layer in merged["selection"]
                for sample in source["global_sample_ids"]
            }
            if seen != expected:
                raise ValueError("missing/extra global sample coverage")
        if any(
            e["retained"] != e["available"] or len(identities[key]) != e["available"]
            for key, e in entries.items()
        ):
            raise ValueError("lost/truncated/duplicate tokens in merge")
        merged["experts"] = list(entries.values())
        merged["seen_samples"] = sorted(seen)
        merged["tensor_bytes"] = sum(
            p["tensor_bytes"] for e in entries.values() for p in e["parts"]
        )
        merged["archive_file_bytes"] = sum(
            p["file_bytes"] for e in entries.values() for p in e["parts"]
        )
        merged["status"] = "complete"
        atomic_json(output / "capture_manifest.json", merged)
    except BaseException as exc:
        merged.update(status="incomplete", error=repr(exc))
        atomic_json(output / "capture_manifest.json", merged)
        raise  # Every original shard remains available; never clean on merge failure.
    cleanup_errors = []
    for directory, allowed in cleanup_files.items():
        try:
            _check_cleanup_tree(directory, allowed, output)
            shutil.rmtree(directory)
        except OSError as exc:
            cleanup_errors.append(dict(path=str(directory), error=repr(exc)))
        except ValueError as exc:
            cleanup_errors.append(dict(path=str(directory), error=repr(exc)))
    merged["cleanup_status"] = "failed" if cleanup_errors else "complete"
    merged["cleanup_errors"] = cleanup_errors
    atomic_json(output / "capture_manifest.json", merged)
    return merged


def _check_cleanup_tree(directory, allowed, output):
    """Deletion is restricted to explicitly supplied, verified sibling shards."""
    if (
        directory.resolve().parent != output.parent
        or directory == output
        or directory.is_symlink()
    ):
        raise ValueError("unsafe rank shard cleanup path")
    children = list(directory.rglob("*"))
    if any(
        p.is_symlink() or getattr(p, "is_junction", lambda: False)()
        for p in [directory, *children]
    ):
        raise ValueError("linked rank shard cleanup path")
    if {p.resolve() for p in children if p.is_file()} != allowed:
        raise ValueError("unexpected files in temporary rank shard; preserve it")
    if any(p.is_dir() and p != directory / "captured_activations" for p in children):
        raise ValueError("unexpected directories in rank shard; preserve it")
