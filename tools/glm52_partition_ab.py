# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-layer Sequential partition A/B; never loads or saves a full model.

Uses the L4 identity-boundary harness and real session/SequentialPipeline APIs.
Each strategy runs in a fresh process. Disk backing is private scratch space;
the source checkpoint is read-only. See examples/glm52_precision/partition_ab.md.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path

import psutil
import torch
from compressed_tensors.offload import align_module_device, offload_module
from compressed_tensors.offload.cache import CPUCache, DiskCache
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as glm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from examples.glm52_precision.oneshot_profile import Profile  # noqa: E402
from llmcompressor.args import DatasetArguments  # noqa: E402
from llmcompressor.core import LifecycleCallbacks, create_session  # noqa: E402
from llmcompressor.modeling.moe.linear_experts import (  # noqa: E402
    ExpertMLP,
    LinearExperts2D,
)
from llmcompressor.modeling.offset_norm import norm_calibration_context  # noqa: E402
from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: E402
from llmcompressor.modifiers.transform.flex_smooth import (  # noqa: E402
    base as flex_base,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin  # noqa: E402
from llmcompressor.pipelines.cache import IntermediatesCache  # noqa: E402
from llmcompressor.pipelines.sequential import pipeline as pipeline_module  # noqa: E402
from llmcompressor.pipelines.sequential.helpers import Subgraph  # noqa: E402
from llmcompressor.recipe import Recipe  # noqa: E402
from tools.glm52_l4 import digest_tensor  # noqa: E402
from tools.glm52_layer_l4 import (  # noqa: E402
    LayerHarness,
    empty_layer,
    layer_plan,
    read_cache,
)
from tools.glm52_precision_probe import local_file  # noqa: E402
from tools.run_quarot_checks import git_info  # noqa: E402

RECIPE = REPO / "examples/glm52_precision/mixed_mxfp.yaml"
TARGETS = {
    "layer": ["GlmMoeDsaDecoderLayer"],
    "expert": [r"re:.*\.input_layernorm$", "ExpertMLP", "GlmMoeDsaMLP"],
}


class LayerBody(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.embed_tokens, self.layers, self.norm = (
            body.embed_tokens,
            body.layers,
            body.norm,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        cos,
        sin,
        prev_topk_indices=None,
    ):
        x = hidden_states @ self.embed_tokens.weight
        output, topk = self.layers[0](
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            prev_topk_indices=prev_topk_indices,
            past_key_values=None,
            use_cache=False,
        )
        return self.norm(output), topk


class SequentialLayer(LayerHarness):
    """Explicit inputs for FX; unchanged decoder and L4 identity boundaries."""

    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.model = LayerBody(self.model)

    @property
    def device(self):
        return torch.device("cpu")  # dummy input device; execution set explicitly

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        cos,
        sin,
        prev_topk_indices=None,
    ):
        output, topk = self.model(
            hidden_states, attention_mask, position_ids, cos, sin, prev_topk_indices
        )
        return self.lm_head(output), topk


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def synthetic_inputs(cfg, seed, samples, tokens, dtype):
    """Capacity/scheduling inputs, not evidence of production calibration accuracy."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        rotary = glm.GlmMoeDsaRotaryEmbedding(cfg)
        batches = []
        for _ in range(samples):
            x = torch.randn(1, tokens, cfg.hidden_size).to(dtype)
            positions = torch.arange(tokens).unsqueeze(0)
            cos, sin = rotary(x, positions)
            batch = dict(
                hidden_states=x,
                attention_mask=torch.full(
                    (1, 1, tokens, tokens),
                    torch.finfo(dtype).min,
                    dtype=dtype,
                ).triu(1),
                position_ids=positions,
                cos=cos,
                sin=sin,
            )
            if cfg.indexer_types[0] == "shared":
                # Fixed causal recent-token indices stand in for the previous layer.
                offsets = torch.arange(min(tokens, cfg.index_topk))
                batch["prev_topk_indices"] = (
                    (torch.arange(tokens)[:, None] - offsets[None, :])
                    .clamp_min(0)
                    .unsqueeze(0)
                )
            batches.append(batch)
    return batches


def tiny_fixture(seed, samples, tokens, dtype):
    cfg = GlmMoeDsaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=0,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        index_topk=2,
        index_head_dim=64,
        index_n_heads=2,
        indexer_types=["full"],
        max_position_embeddings=max(64, tokens),
        use_cache=False,
    )
    cfg._attn_implementation = "eager"
    expert_cls = LinearExperts2D.get_linear_experts_cls(glm.GlmMoeDsaExperts)
    with torch.random.fork_rng(), patch_attr(glm, "GlmMoeDsaExperts", expert_cls):
        torch.manual_seed(seed)
        model = SequentialLayer(cfg, glm.GlmMoeDsaDecoderLayer(cfg, 0)).to(dtype)
        # Direct DecoderLayer construction does not run HF post_init.
        for name, value in model.model.layers[0].named_parameters():
            if value.ndim > 1:
                nn.init.normal_(value, std=0.02)
            elif name.endswith("weight"):
                nn.init.ones_(value)
        batches = synthetic_inputs(cfg, seed, samples, tokens, dtype)
    return model.eval(), batches


def offload(model, device, kind, scratch):
    for module in model.modules():
        offload_module(
            module,
            onload_device=device,
            offload_device=kind,
            **({"offload_dir": str(scratch)} if kind == "disk" else {}),
        )


def load_selected(args, scratch):
    """Stream selected tensors to private backing, preserving source provenance."""
    dtype = getattr(torch, args.dtype)
    if args.tiny:
        model, batches = tiny_fixture(args.seed, args.samples, args.tokens, dtype)
        provenance = {"kind": "tiny synthetic", "seed": args.seed}
        offload(model, args.device, args.offload, scratch)
        return model, batches, provenance

    root = Path(args.model_dir).resolve()
    cfg, plan = layer_plan(root, args.layer, args.max_weight_gib)
    if plan["mlp_type"] != "sparse":
        raise ValueError("Choose a MoE layer, not a dense layer")
    # Validate small inputs before reading any weight payload.
    caches = [
        read_cache(Path(path), cfg, plan) for path in (args.activation_cache or [])
    ]
    batches = (
        synthetic_inputs(cfg, args.seed, args.samples, args.tokens, dtype)
        if args.synthetic_inputs
        else []
    )
    layer = empty_layer(cfg)
    parameters = set(dict(layer.named_parameters()))
    by_module = defaultdict(list)
    for name, entry in plan["selected_weights"].items():
        parent, leaf = name.rsplit(".", 1)
        by_module[parent].append((name, leaf, entry))
    model = SequentialLayer(cfg, layer)
    for name in ("model.embed_tokens", "model.norm", "lm_head"):
        model.get_submodule(name).to(dtype)
    hashes = {}
    for parent, entries in by_module.items():
        module = layer.get_submodule(parent)
        for name, leaf, entry in entries:
            with safe_open(
                str(local_file(root, entry["shard"])), framework="pt", device="cpu"
            ) as stream:
                value = stream.get_tensor(entry["key"])
            hashes[name] = digest_tensor(value)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source tensor: {name}")
            # Clone before any writeback; never mutate source mmap/checkpoint.
            value = value.to(dtype).clone()
            setattr(
                module,
                leaf,
                nn.Parameter(value, requires_grad=False)
                if name in parameters
                else value,
            )
        offload_module(
            module,
            args.device,
            args.offload,
            **({"offload_dir": str(scratch)} if args.offload == "disk" else {}),
        )
    # Offload adapters and modules without checkpoint entries exactly once.
    for module in model.modules():
        if not hasattr(module, "_original_forward_func"):
            offload_module(
                module,
                args.device,
                args.offload,
                **({"offload_dir": str(scratch)} if args.offload == "disk" else {}),
            )
    for values, metadata in caches:
        if metadata["weight_sha256"] != hashes:
            raise ValueError("Activation cache weight fingerprints differ from source")
        batches.append(
            {
                key: value.to(dtype) if value.is_floating_point() else value
                for key, value in values.items()
                if not key.startswith("baseline_")
            }
        )
    return (
        model.eval(),
        batches,
        {
            "kind": (
                "selected real weights; synthetic activations, NOT accuracy evidence"
                if args.synthetic_inputs
                else "selected real weights and L4 cache"
            ),
            "layer": args.layer,
            "indexer_type": cfg.indexer_types[0],
            "config_sha256": plan["config_sha256"],
            "weights_sha256": hashes,
            "cache_sha256": {
                str(path): file_hash(path) for path in (args.activation_cache or [])
            },
            "cache_metadata": [metadata for _, metadata in caches],
        },
    )


class Measurements:
    """Boundary-synchronized phases; hot calls are host time/counts only."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.accelerator = (
            getattr(torch, self.device.type) if self.device.type != "cpu" else None
        )
        self.process = psutil.Process()
        self.report = {
            "phases": {},
            "hot_calls": {},
            "partitions": [],
            "expert_calls": {},
            "weight_cache_by_module": {},
        }
        self.graph_indices = {}
        self.current = "setup"
        self.rss_peak = 0
        self.stop = threading.Event()

    def sync(self):
        if self.accelerator:
            self.accelerator.synchronize(self.device)

    def memory(self):
        result = {"rss_bytes": self.process.memory_info().rss}
        if self.accelerator:
            result.update(
                {
                    name: getattr(self.accelerator, function)(self.device)
                    for name, function in (
                        ("allocated_bytes", "memory_allocated"),
                        ("reserved_bytes", "memory_reserved"),
                        ("peak_allocated_bytes", "max_memory_allocated"),
                        ("peak_reserved_bytes", "max_memory_reserved"),
                    )
                }
            )
        return result

    def begin(self, name):
        print(f"[partition A/B] starting {name}", flush=True)
        self.sync()
        if self.accelerator:
            self.accelerator.reset_peak_memory_stats(self.device)
        self.current = name
        self.rss_peak = self.process.memory_info().rss
        self.started = time.perf_counter()
        self.io_before = self.process.io_counters()

    def end(self):
        submitted = time.perf_counter() - self.started
        self.sync()
        elapsed = time.perf_counter() - self.started
        memory = self.memory()
        io_after = self.process.io_counters()
        self.report["phases"][self.current] = {
            "synchronized_wall_seconds": elapsed,
            "host_wall_seconds": submitted,
            "end_sync_seconds": elapsed - submitted,
            "max_sampled_rss_bytes": max(self.rss_peak, memory["rss_bytes"]),
            "end_memory": memory,
            "process_read_bytes": io_after.read_bytes - self.io_before.read_bytes,
            "process_write_bytes": io_after.write_bytes - self.io_before.write_bytes,
        }
        print(f"[partition A/B] {self.current}: {elapsed:.3f}s", flush=True)
        self.current = "between_phases"

    @contextmanager
    def phase(self, name):
        self.begin(name)
        try:
            yield
        finally:
            self.end()

    def count(self, label, seconds=0, size=0):
        key = f"{self.current}/{label}"
        row = self.report["hot_calls"].setdefault(
            key, {"calls": 0, "host_seconds": 0.0, "logical_bytes": 0}
        )
        row["calls"] += 1
        row["host_seconds"] += seconds
        row["logical_bytes"] += size

    @contextmanager
    def sample_rss(self):
        def sample():
            while not self.stop.wait(0.25):
                self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self.stop.set()
            thread.join()

    @contextmanager
    def instrument(self, model):
        cache_names = {
            id(cache): f"{name}.{kind}"
            for name, module in model.named_modules()
            for kind, cache in (
                ("parameters", module._parameters),
                ("buffers", module._buffers),
            )
        }
        with ExitStack() as stack:

            def hot(owner, method, label, size_fn=lambda *a, **k: 0):
                original = getattr(owner, method)

                @wraps(original)
                def call(*args, **kwargs):
                    start = time.perf_counter()
                    result = original(*args, **kwargs)
                    self.count(
                        label, time.perf_counter() - start, size_fn(*args, **kwargs)
                    )
                    if id(args[0]) in cache_names:
                        key = f"{self.current}/{cache_names[id(args[0])]}/{method}"
                        row = self.report["weight_cache_by_module"].setdefault(
                            key, {"calls": 0, "logical_bytes": 0}
                        )
                        row["calls"] += 1
                        row["logical_bytes"] += size_fn(*args, **kwargs)
                    return result

                stack.enter_context(patch_attr(owner, method, call))

            for cls in (CPUCache, DiskCache):
                for method in ("onload", "update_offload"):
                    hot(
                        cls,
                        method,
                        f"{cls.__name__}.{method}",
                        lambda cache, value, *a, **k: 0
                        if value is None
                        else value.numel() * value.element_size(),
                    )
            for method in ("fetch", "update", "delete"):
                hot(IntermediatesCache, method, f"intermediates.{method}")
            for owner, method, label in (
                (flex_base, "search_alpha_beta", "flex_search"),
                (flex_base, "apply_scales", "flex_apply"),
                (QuantizationModifier, "on_sequential_epoch_end", "quant_epoch"),
            ):
                hot(owner, method, label)
            original_forward = Subgraph.forward

            def forward(graph, *args, **kwargs):
                phase = "propagation" if HooksMixin._HOOKS_DISABLED else "calibration"
                start = time.perf_counter()
                result = original_forward(graph, *args, **kwargs)
                elapsed = time.perf_counter() - start
                self.count(f"forward.{phase}", elapsed)
                row = self.report["partitions"][self.graph_indices[id(graph)]]
                stats = row.setdefault(phase, {"calls": 0, "host_seconds": 0})
                stats["calls"] += 1
                stats["host_seconds"] += elapsed
                return result

            stack.enter_context(patch_attr(Subgraph, "forward", forward))
            original_trace = pipeline_module.trace_subgraphs

            def trace(*args, **kwargs):
                with self.phase("trace_partition"):
                    graphs = original_trace(*args, **kwargs)
                self.report["partitions"] = [
                    {
                        "modules": [
                            str(node.target)
                            for node in graph.graph.nodes
                            if node.op == "call_module"
                        ]
                    }
                    for graph in graphs
                ]
                self.graph_indices = {
                    id(graph): index for index, graph in enumerate(graphs)
                }
                return graphs

            stack.enter_context(patch_attr(pipeline_module, "trace_subgraphs", trace))
            original_start = LifecycleCallbacks.calibration_start

            def calibration_start(*args, **kwargs):
                with self.phase("calibration_start_including_quarot"):
                    result = original_start(*args, **kwargs)
                self.begin("sequential_execution")
                return result

            stack.enter_context(
                patch_attr(LifecycleCallbacks, "calibration_start", calibration_start)
            )
            # Explicit execution device only within this standalone harness.
            stack.enter_context(
                patch_attr(pipeline_module, "get_main_device", lambda: self.device)
            )
            stack.enter_context(
                patch_attr(
                    pipeline_module,
                    "tqdm",
                    Profile("cpu").progress(pipeline_module.tqdm),
                )
            )
            for name, module in model.named_modules():
                if not isinstance(module, ExpertMLP):
                    continue

                def expert(module, inputs, name=name):
                    if self.current != "sequential_execution":
                        return
                    phase = (
                        "propagation" if HooksMixin._HOOKS_DISABLED else "calibration"
                    )
                    key = f"{name}/{phase}"
                    row = self.report["expert_calls"].setdefault(
                        key, {"calls": 0, "tokens": 0}
                    )
                    row["calls"] += 1
                    row["tokens"] += inputs[0].shape[0]

                handle = module.register_forward_pre_hook(expert)
                stack.callback(handle.remove)
            yield


def state_hashes(model):
    result = {}
    # A bounded post-run CPU audit, excluded from measured calibration phases.
    for name, module in model.named_modules():
        with align_module_device(module, "cpu"):
            for leaf, value in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False)
            ):
                result[f"{name}.{leaf}"] = digest_tensor(value)
    return result


@torch.no_grad()
def run_trial(args):
    output = Path(args.output)
    if not Path(pipeline_module.__file__).resolve().is_relative_to(REPO / "src"):
        raise ValueError("Install this checkout with pip install --no-deps -e . first")
    if args.model_dir:
        source = Path(args.model_dir).resolve()
        for destination in (output, Path(args.scratch) if args.scratch else output):
            if destination.resolve().is_relative_to(source):
                raise ValueError("Output and scratch must be outside the checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    device_type = args.device.split(":")[0]
    if device_type == "npu":
        import torch_npu  # noqa: F401
    if device_type != "cpu":
        getattr(torch, device_type).set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This single-rank layer experiment must not use torchrun")
    meter = Measurements(args.device)
    report = {
        "schema_version": 1,
        "status": "running",
        "arguments": vars(args),
        "git": git_info(REPO),
        "recipe_sha256": file_hash(RECIPE),
        "harness_sha256": file_hash(Path(__file__)),
        "recipe_overrides": {
            "QuaRot.precision": "float32",
            "FlexSmooth.max_tokens": None,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "compressed-tensors")
        },
        "scope": "one MoE layer with L4 identity boundaries; no full-model save",
        "measurement": meter.report,
        "limits": [
            "single rank; no HCCL/shared distributed backing",
            "allocator peaks exclude driver/HCCL memory",
            "RSS sampled every 250ms; not exact peak or PSS",
            "cache bytes are logical payload, not physical disk/H2D traffic",
            "hot timing is host dispatch, not synchronized device time",
            "OS page cache is not flushed; no claim of cold-cache results",
            "no packed checkpoint or downstream accuracy verification",
        ],
    }
    if device_type != "cpu":
        report["device_name"] = getattr(torch, device_type).get_device_name(args.device)
    if device_type == "npu":
        report["versions"]["torch-npu"] = importlib.metadata.version("torch-npu")
    write_json(output / "report.json", report)
    try:
        with meter.sample_rss(), tempfile.TemporaryDirectory(
            prefix="glm52-ab-", dir=args.scratch or output
        ) as scratch:
            with meter.phase("selected_layer_load"):
                model, batches, provenance = load_selected(args, Path(scratch))
            report["input"] = provenance
            report["batch_shapes"] = [list(b["hidden_states"].shape) for b in batches]
            report["input_tensor_hashes"] = [
                {key: digest_tensor(value) for key, value in batch.items()}
                for batch in batches
            ]
            report["initial_state_hashes"] = state_hashes(model)
            loader = DataLoader(batches, batch_size=None, shuffle=False)
            recipe = Recipe.create_instance(str(RECIPE)).modifiers
            recipe[0].precision = torch.float32
            recipe[1].max_tokens = None
            dataset_args = DatasetArguments(
                pipeline="sequential",
                sequential_targets=TARGETS[args.strategy],
                sequential_targets_per_subgraph=1,
                sequential_offload_device="cpu",
                propagate_error=True,
                moe_calibrate_all_experts=False,
            )
            with meter.instrument(model), create_session() as session:
                with norm_calibration_context(model):
                    with meter.phase("initialize"):
                        session.initialize(
                            model=model,
                            recipe=recipe,
                            start=-1,
                            calib_data=loader,
                            sequential_targets=TARGETS[args.strategy],
                        )
                    try:
                        pipeline_module.SequentialPipeline()(
                            model, loader, dataset_args
                        )
                    finally:
                        if meter.current == "sequential_execution":
                            meter.end()
                with meter.phase("finalize"):
                    session.finalize()
            report["final_state_hashes"] = state_hashes(model)
            report["flex"] = model.config.flex_smooth_config
            report["quarot"] = model.config.quarot_config
            tensors = {
                f"scale.{key}": value["scale"].contiguous()
                for key, value in recipe[1].diagnostics.items()
            }
            # Plain post-finalize forward; do not retain an entire layer on NPU.
            for index, batch in enumerate(batches):
                result, topk = model(**batch)
                if not torch.isfinite(result).all():
                    raise ValueError(f"Nonfinite post-finalize output: sample {index}")
                tensors[f"output.{index}"] = result.detach().cpu().contiguous()
                if topk is not None:
                    tensors[f"topk.{index}"] = topk.detach().cpu().contiguous()
            save_file(tensors, str(output / "numerics.safetensors"))
            report["status"] = "completed"
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def compare_trials(left, right):
    a = json.loads((left / "report.json").read_text())
    b = json.loads((right / "report.json").read_text())
    if a["status"] != "completed" or b["status"] != "completed":
        return {"comparable": False, "reason": "one or both trials did not complete"}
    settings = ("device", "offload", "dtype", "threads", "seed")
    comparable = (
        a["arguments"]["strategy"] == "layer"
        and b["arguments"]["strategy"] == "expert"
        and all(a["arguments"][key] == b["arguments"][key] for key in settings)
        and a["git"]["commit"] == b["git"]["commit"]
        and a["harness_sha256"] == b["harness_sha256"]
        and a["versions"] == b["versions"]
        and a["recipe_overrides"] == b["recipe_overrides"]
        and a["recipe_sha256"] == b["recipe_sha256"]
        and a["initial_state_hashes"] == b["initial_state_hashes"]
        and a["input_tensor_hashes"] == b["input_tensor_hashes"]
    )
    changed = [
        key
        for key in a["final_state_hashes"].keys() | b["final_state_hashes"].keys()
        if a["final_state_hashes"].get(key) != b["final_state_hashes"].get(key)
    ]
    ta, tb = (
        load_file(str(left / "numerics.safetensors")),
        load_file(str(right / "numerics.safetensors")),
    )
    numerics = {}
    for key in ta.keys() | tb.keys():
        if key not in ta or key not in tb or ta[key].shape != tb[key].shape:
            numerics[key] = {"shape_or_key_mismatch": True}
            continue
        x, y = ta[key].double(), tb[key].double()
        delta = x - y
        numerics[key] = {
            "exact": torch.equal(ta[key], tb[key]),
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / x.norm().clamp_min(1e-30)).item(),
        }
    return {
        "comparable": comparable,
        "changed_state_tensors": sorted(changed),
        "numerics": numerics,
        "flex_decisions_equal": a["flex"] == b["flex"],
        "exact_numerical_match": comparable
        and not changed
        and a["flex"] == b["flex"]
        and all(item.get("exact", False) for item in numerics.values()),
        "sequential_layer_over_expert_seconds": (
            a["measurement"]["phases"]["sequential_execution"][
                "synchronized_wall_seconds"
            ]
            / b["measurement"]["phases"]["sequential_execution"][
                "synchronized_wall_seconds"
            ]
        )
        if comparable
        else None,
        "phases": {
            name: {
                "layer": a["measurement"]["phases"][name],
                "expert": b["measurement"]["phases"][name],
            }
            for name in a["measurement"]["phases"]
        },
        "decision": "Inspect numerical differences before choosing a strategy; "
        "timing alone is not an acceptance criterion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "ab", "compare"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--strategy", choices=TARGETS, default="layer")
    parser.add_argument("--model-dir")
    parser.add_argument("--activation-cache", nargs="+")
    parser.add_argument(
        "--synthetic-inputs",
        action="store_true",
        help="real selected weights, seeded synthetic activations",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--max-weight-gib", type=float, default=48)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--samples", type=int, default=2, help="synthetic input count")
    parser.add_argument(
        "--tokens", type=int, default=8, help="synthetic sequence length"
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offload", choices=("cpu", "disk"), default="disk")
    parser.add_argument(
        "--scratch", help="existing scratch parent, never checkpoint dir"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs", type=int, default=2, help="AB then BA, fresh processes"
    )
    args = parser.parse_args()
    if args.command == "compare":
        write_json(Path(args.output), compare_trials(args.left, args.right))
        return
    if args.synthetic_inputs and args.activation_cache:
        parser.error("Choose synthetic inputs OR activation caches")
    if not args.tiny and not (
        args.model_dir and (args.activation_cache or args.synthetic_inputs)
    ):
        parser.error(
            "Use --tiny or local --model-dir with --synthetic-inputs "
            "or --activation-cache"
        )
    if min(args.samples, args.tokens, args.threads, args.pairs) < 1:
        parser.error("samples/tokens/threads/pairs must be positive")
    if args.command == "run":
        run_trial(args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    comparisons = []
    # Pass the exact same CLI to fresh workers, replacing only command/output/strategy.
    base = sys.argv[1:]
    base[0] = "run"
    for pair in range(args.pairs):
        for strategy in ("layer", "expert") if pair % 2 == 0 else ("expert", "layer"):
            destination = root / f"{pair}-{strategy}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *base,
                "--output",
                str(destination),
                "--strategy",
                strategy,
            ]
            with (root / f"{pair}-{strategy}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            print(f"{destination.name}: exit={result.returncode}", flush=True)
        left, right = root / f"{pair}-layer", root / f"{pair}-expert"
        if (left / "report.json").exists() and (right / "report.json").exists():
            comparisons.append(compare_trials(left, right))
        else:
            comparisons.append({"comparable": False, "reason": "worker report missing"})
        write_json(root / "comparison.json", comparisons)
    if not all(item.get("exact_numerical_match") for item in comparisons):
        raise SystemExit("A/B incomplete or numerics differ; inspect comparison.json")


if __name__ == "__main__":
    main()
