"""Experimental BF16 capture adapter using LLMC's existing sequential engine."""

import contextlib
import hashlib
import json
from pathlib import Path

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from safetensors.torch import load_file
from torch.fx import Graph

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_subgraphs
from llmcompressor.utils.helpers import calibration_forward_context

from .e2_capture import RoutedCapture

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
    """Bounded merge: global priority reservoir, no per-rank minimum requirement."""
    merged = None
    seen = set()
    for directory in map(Path, paths):
        manifest = json.loads((directory / "capture_manifest.json").read_text())
        if manifest["status"] != "rank_shard_complete" or manifest["source"] != source:
            raise ValueError("incomplete/incompatible capture rank shard")
        selection = {int(k): v for k, v in manifest["selection"].items()}
        if merged is None:
            merged = RoutedCapture(selection, manifest["target"], manifest["minimum"])
        if (selection, manifest["target"], manifest["minimum"]) != (
            merged.selection,
            merged.target,
            merged.minimum,
        ):
            raise ValueError("incompatible rank capture configuration")
        samples = set(map(tuple, manifest["seen_samples"]))
        if seen & samples:
            raise ValueError("overlapping rank sample partitions")
        seen |= samples
        merged.norm_errors.extend(manifest["norm_errors"])
        for name, value in manifest["expert_weight_sha256"].items():
            if name in merged.weight_hashes and merged.weight_hashes[name] != value:
                raise ValueError("rank expert weight mismatch")
            merged.weight_hashes[name] = value
        for layer, eps in manifest["norm_epsilon"].items():
            layer = int(layer)
            if layer in merged.norm_eps and merged.norm_eps[layer] != eps:
                raise ValueError("rank norm epsilon mismatch")
            merged.norm_eps[layer] = eps
        entries = {(e["layer"], e["expert"]): e for e in manifest["experts"]}
        if set(entries) != set(merged.buffers) or len(entries) != len(
            manifest["experts"]
        ):
            raise ValueError("rank expert set mismatch")
        for key, entry in entries.items():
            merged.counts[key] += entry["available"]
            if not entry["retained"]:
                continue
            path = (directory / entry["file"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("rank tensor hash/path mismatch")
            tensors = load_file(str(path))
            layer, expert = key
            gamma = tensors["gamma"]
            if layer in merged.gammas and not torch.equal(merged.gammas[layer], gamma):
                raise ValueError("rank gamma mismatch")
            merged.gammas[layer] = gamma
            rows = merged.buffers[key]
            for i, (sample, pos) in enumerate(
                zip(tensors["sample_id"].tolist(), tensors["token_position"].tolist())
            ):
                priority = hashlib.sha256(
                    f"1234:{layer}:{expert}:{sample}:{pos}".encode()
                ).hexdigest()
                rows.append(
                    (
                        priority,
                        sample,
                        pos,
                        tensors["z"][i].clone(),
                        tensors["z_gamma"][i].clone(),
                        tensors["router_weight"][i].clone(),
                    )
                )
            merged.buffers[key] = sorted(rows, key=lambda r: (r[0], r[1], r[2]))[
                : merged.target
            ]
    if merged is None:
        raise ValueError("no capture shards")
    merged.seen = seen
    merged.save(output, source)
    return merged
