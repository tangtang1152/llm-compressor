"""Real BF16 capture via the historical load_context/auto_offload path.

No modifiers, quantization, checkpoint saving, generation, or shared-memory remount.
Run data preflight first; use a single-layer capture before the four-layer job.
"""

import argparse
import hashlib
import inspect
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import ReadInstruction
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from llmcompressor.datasets import get_rank_partition
from llmcompressor.utils import load_context

from .diagnose import atomic_json
from .e2_capture import SELECTION, RoutedCapture
from .e2_preflight import inspect_data
from .e2_sequential import collect, merge_shards
from .run import Checkpoint, provenance


def consensus(error):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(e is not None for e in errors):
        raise RuntimeError(f"capture phase failed by rank: {errors}")


def prepare(args, rank, world):
    """Verify original data and checkpoint metadata before any model load."""
    if args.samples < world:
        raise ValueError("at least one sample per rank required")
    preflight = json.loads(args.preflight.read_text())
    if preflight["status"] != "data_preflight_passed_capture_not_validated":
        raise ValueError("requires successful data preflight")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    data, tokens = inspect_data(
        args.dataset, tokenizer, args.samples, return_tokens=True
    )
    for key in (
        "dataset_sha256",
        "tokenized_sha256",
        "preprocessing",
        "samples_selected",
    ):
        if data[key] != preflight[key]:
            raise ValueError(f"preflight mismatch: {key}")
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if (
        config.model_type != "glm_moe_dsa"
        or getattr(config, "quantization_config", None)
        or getattr(config, "quarot_config", None)
    ):
        raise ValueError("requires original unquantized GLM checkpoint")
    if config.hidden_act != "silu" or config.mlp_bias:
        raise ValueError("replay requires SiLU and bias-free expert linears")
    cp = Checkpoint(args.checkpoint, [], 0, 1234)
    selection = {layer: SELECTION[layer] for layer in args.layers}
    cp.select(args.layers, 3, 1234, selections=selection)
    partition = ReadInstruction.from_spec(
        get_rank_partition("train", args.samples)
    ).to_absolute({"train": len(tokens)})[0]
    ids = list(range(partition.from_ or 0, partition.to or len(tokens)))
    collate = DataCollatorWithPadding(tokenizer)
    batches = [
        collate([{k: v for k, v in tokens[i].items() if k != "sample_id"}]) for i in ids
    ]
    source = dict(
        kind="real_bf16",
        **data,
        checkpoint=str(args.checkpoint),
        checkpoint_index_sha256=hashlib.sha256(cp.index_path.read_bytes()).hexdigest(),
        checkpoint_config_sha256=hashlib.sha256(
            (args.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        capture_driver="experiments.quarot_mxfp4.e2_collect",
        world_size=world,
        routing_alignment_verified=True,
        provenance=provenance(),
        experiment_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("e2*.py")
        },
    )
    return batches, ids, source, selection


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset", type=Path, default=Path("/mnt/share/datasets/qwen3_cot_w4a4.json")
    )
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offload-dir", type=Path, required=True)
    p.add_argument("--max-memory", type=json.loads)
    p.add_argument("--device", choices=("npu", "cpu"), default="npu")
    p.add_argument("--samples", type=int, default=42)
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        choices=sorted(SELECTION),
        default=sorted(SELECTION),
    )
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if args.device == "npu":
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        dist.init_process_group(
            backend="cpu:gloo,npu:hccl" if args.device == "npu" else "gloo",
            timeout=timedelta(hours=3),
        )
    # All ranks share new output/offload paths, as in the historical loader.
    error = None
    if rank == 0:
        try:
            if args.output.exists() or args.offload_dir.exists():
                raise FileExistsError("output and offload directories must both be new")
            args.output.mkdir(parents=True)
            args.offload_dir.mkdir(parents=True)
        except Exception as exc:
            error = repr(exc)
    consensus(error)
    status = dict(
        status="running",
        rank=rank,
        world_size=world,
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    status_path = args.output / f"status-rank{rank}.json"
    atomic_json(status_path, status)
    try:
        error = None
        try:
            batches, ids, source, selection = prepare(args, rank, world)
        except Exception as exc:
            error = repr(exc)
        consensus(error)
        status.update(stage="loading", global_sample_ids=ids)
        atomic_json(status_path, status)
        loading = dict(
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto_offload",
            offload_folder=str(args.offload_dir),
        )
        if args.max_memory:
            loading["max_memory"] = {
                int(k) if k.isdecimal() else k: v for k, v in args.max_memory.items()
            }
        with load_context():
            model = AutoModelForCausalLM.from_pretrained(
                str(args.checkpoint), **loading
            ).eval()
        # Preserve the model's native FP32 exceptions (e.g. indexer.weights_proj).
        # Only the captured norms and replayed expert weights require BF16.
        for layer, experts in selection.items():
            base = f"model.layers.{layer}"
            names = [f"{base}.post_attention_layernorm"] + [
                f"{base}.mlp.experts.{e}" for e in experts
            ]
            if any(
                p.dtype != torch.bfloat16
                for name in names
                for p in model.get_submodule(name).parameters()
            ):
                raise ValueError("selected norm/expert has non-BF16 parameters")
        source["modeling_sha256"] = hashlib.sha256(
            Path(inspect.getfile(type(model))).read_bytes()
        ).hexdigest()
        if dist.is_initialized():
            dist.barrier()
        status["stage"] = "capturing"
        atomic_json(status_path, status)
        collector = RoutedCapture(selection)
        device = (
            torch.device("npu", torch.npu.current_device())
            if args.device == "npu"
            else torch.device("cpu")
        )
        report = collect(model, batches, ids, collector, device)
        collector.save(args.output / f"rank{rank}", source, rank_shard=True)
        status.update(stage="rank_capture_complete", **report)
        atomic_json(status_path, status)
        if dist.is_initialized():
            dist.barrier()
        error = None
        if rank == 0:
            try:
                merge_shards(
                    [args.output / f"rank{i}" for i in range(world)],
                    args.output / "capture",
                    source,
                )
            except Exception as exc:
                error = repr(exc)
        consensus(error)
        status.update(status="complete", stage="global_capture_complete")
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(status_path, status)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
