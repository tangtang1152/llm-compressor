"""CT CPU/disk loader and bounded sequential capture, adapted from QuaRot E2."""

import os
from pathlib import Path

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from torch.fx import Graph

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_subgraphs
from llmcompressor.utils.helpers import calibration_forward_context


def loader_options(dtype, offload_dir, max_memory=None):
    """Prepare only a new dedicated backing directory; never clean user data."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("FlexSmooth capture currently requires a single process")
    if not offload_dir:
        raise ValueError("auto_offload requires --offload-dir")
    path = Path(offload_dir).resolve()
    if max_memory is not None and not isinstance(max_memory, dict):
        raise ValueError("--max-memory must be a JSON object")
    path.mkdir(parents=True, exist_ok=False)
    options = dict(
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
        device_map="auto_offload",
        offload_folder=str(path),
    )
    if max_memory:
        options["max_memory"] = {
            int(k) if str(k).isdecimal() else k: v for k, v in max_memory.items()
        }
    return options


def capture_prefix(subgraphs, plan):
    """Stop after the last selected attention call, before head/later layers.

    Each mapping's hook observes one shared input, as in production FlexSmooth.
    Attention remains a leaf: its native helper methods are not FX-traced.
    """
    required = {".".join(m.source.split(".")[:3]) + ".self_attn" for m in plan}
    calls = {name: [] for name in required}
    for i, subgraph in enumerate(subgraphs):
        for node in subgraph.graph.nodes:
            if node.op == "call_module" and str(node.target) in calls:
                calls[str(node.target)].append((i, node))
    if not calls or any(len(nodes) != 1 for nodes in calls.values()):
        raise ValueError("capture graph requires exactly one call_module per mapping")
    stop = max(nodes[0][0] for nodes in calls.values())
    nodes = list(subgraphs[stop].graph.nodes)
    last = max(
        nodes.index(items[0][1]) for items in calls.values() if items[0][0] == stop
    )
    graph, env = Graph(), {}
    for node in nodes[: last + 1]:
        env[node] = graph.node_copy(node, lambda old: env[old])
    graph.output({})
    graph.lint()
    prefix = subgraphs[:stop] + [Subgraph(graph, subgraphs[stop].input_names, set())]
    max_layer = max(int(m.source.split(".")[2]) for m in plan)
    for subgraph in prefix:
        for node in subgraph.graph.nodes:
            if node.op != "call_module":
                continue
            name = str(node.target)
            if name == "lm_head" or (
                name.startswith("model.layers.") and int(name.split(".")[2]) > max_layer
            ):
                raise ValueError(f"unwanted module in capture prefix: {name}")
    return prefix


class SequentialRunner:
    def __init__(self, model, batches, plan, device):
        if get_calibrate_all_experts_flag():
            raise ValueError("moe_calibrate_all_experts must be False")
        self.model, self.device = model, torch.device(device)
        set_onload_device(model, self.device)
        # Norm boundaries and individual experts bound resident weight sets.
        # Keep attention opaque, as in E2; hooks still observe internal consumers.
        targets = [r"re:.*\.input_layernorm$", "ExpertMLP", "GlmMoeDsaMLP"]
        targets += [r"re:.*\.self_attn$"]
        self.subgraphs = capture_prefix(
            trace_subgraphs(
                model, batches[0], targets, DatasetArguments().tracing_ignore
            ),
            plan,
        )

    @torch.no_grad()
    def run(self, batches, before_batch):
        cache = IntermediatesCache.from_dataloader(
            batches, self.device, torch.device("cpu")
        )
        with calibration_forward_context(self.model):
            for index, subgraph in enumerate(self.subgraphs):
                # Retain only this subgraph's weights across samples, then release.
                with disable_offloading():
                    for batch in range(len(batches)):
                        before_batch(batch)
                        inputs = cache.fetch(batch, subgraph.input_names)
                        outputs = subgraph.forward(self.model, **inputs)
                        if index < len(self.subgraphs) - 1:
                            cache.update(batch, outputs)
                            cache.delete(batch, subgraph.consumed_names)
                        del inputs, outputs
                print(
                    f"capture subgraph={index+1}/{len(self.subgraphs)} "
                    f"batches={len(batches)}",
                    flush=True,
                )
