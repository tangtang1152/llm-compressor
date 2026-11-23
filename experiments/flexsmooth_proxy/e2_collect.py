"""Two split ranks, historical QuaRot E2 loading, one load for smoke and capture."""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import psutil
import torch
import torch.distributed as dist

from .capture import split_records
from .core import memory, provenance, sync
from .distributed_capture import (
    capture_split,
    compare_reference,
    merge_shards,
    partition,
    prepare,
)
from .next_stage.common import csv_rows, file_hash, git_head, identities, load_cache
from .run import write_json


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    write_json(temp, value)
    temp.replace(path)


def consensus(error):
    # Same all-rank phase handshake as QuaRot E2. Loader-internal failures can
    # still strand CT collectives: process-group timeout/torchrun handles those.
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(e is not None for e in errors):
        raise RuntimeError(f"capture phase failed by rank: {errors}")


def phase(fn):
    result = None
    error = None
    try:
        result = fn()
    except Exception as exc:
        error = repr(exc)
    consensus(error)
    return result


def backing_signature(model):
    from compressed_tensors.offload.cache import (
        DistributedCPUCache,
        DistributedDiskCache,
    )

    entries = []
    for name, module in model.named_modules():
        for kind in ("_parameters", "_buffers"):
            cache = getattr(module, kind)
            values = getattr(cache, "offloaded_values", cache)
            for key, value in values.items():
                if value is None or not value.numel():
                    continue
                if isinstance(cache, DistributedCPUCache):
                    if not value.is_shared():
                        raise ValueError("Private CPU backing detected")
                    identity = str(value.untyped_storage()._share_filename_cpu_()[:2])
                elif isinstance(cache, DistributedDiskCache):
                    identity = str(cache.index[value]["safetensors_file"]) + str(
                        cache.index[value]["weight_name"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported/private/device backing at {name}.{key}: {type(cache).__name__}"
                    )
                entries.append(
                    (name, kind, key, str(value.dtype), tuple(value.shape), identity)
                )
    if not entries:
        raise ValueError("No shared weights found")
    return dict(
        tensors=len(entries),
        sha256=hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
    )


def runtime_contract():
    """Fail closed before model loading on an unrecognized CT loader contract."""
    from compressed_tensors.offload import load as loader
    from compressed_tensors.offload.cache import dist_cpu, dist_disk

    source = inspect.getsource(loader.load_offloaded_model)
    if not all(
        s in source
        for s in (
            "if not is_source_process():",
            'kwargs["device_map"] = "meta"',
            "from_accelerate(model)",
            '"auto_offload"',
        )
    ):
        raise ValueError(
            "Unrecognized CT distributed loader: inspect source before use"
        )
    return {
        Path(m.__file__).name: file_hash(Path(m.__file__))
        for m in (loader, dist_cpu, dist_disk)
    }


def loading_options(offload_dir):
    # Exactly the historical default: leave CPU/disk estimation to CT.
    return dict(
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto_offload",
        offload_folder=str(offload_dir),
    )


def check_memory(reserve_gib):
    if psutil.virtual_memory().available < reserve_gib * 2**30:
        raise MemoryError(
            "Insufficient free CPU working memory; preserving partial artifacts"
        )


def run_captures(
    model,
    splits,
    source,
    device,
    output,
    reference,
    rank,
    context,
    check,
    nmse_tol,
    max_abs_tol,
):
    """One already-loaded model; fail smoke before starting the long prefix."""
    timings = []
    for label, layers, mappings, tokens in [
        ("capture-smoke", [3], ["input_layernorm"], 128),
        ("capture", [6, 16, 60, 77], ["input_layernorm", "q_a_layernorm"], 2048),
    ]:
        root = output / label
        phase(lambda: root.mkdir() if rank == 0 else None)
        metadata = dict(
            source,
            layers=layers,
            mapping_suffixes=[
                "input_layernorm"
                if m == "input_layernorm"
                else "self_attn.q_a_layernorm"
                for m in mappings
            ],
            token_ids={
                s: prepare(records, tokens, context)[1]
                for s, records in zip(("calibration", "validation"), splits)
            },
            rank_plan=partition(splits, tokens, context, 2),
        )
        shard = phase(
            lambda: capture_split(
                model,
                splits[rank],
                rank,
                metadata,
                layers,
                mappings,
                tokens,
                context,
                device,
                check,
            )
        )
        timings.append(
            dict(
                stage=label,
                **{
                    k: shard[k]
                    for k in (
                        "capture_seconds",
                        "forward_seconds",
                        "trace_seconds",
                        "weight_snapshot_seconds",
                    )
                },
            )
        )

        def save():
            tmp = root / f"rank{rank}.tmp"
            torch.save(shard, tmp)
            tmp.replace(root / f"rank{rank}.pt")

        phase(save)
        shard = None

        def merge():
            if rank != 0:
                return
            check()
            result = merge_shards(
                [
                    torch.load(
                        root / f"rank{i}.pt", map_location="cpu", weights_only=True
                    )
                    for i in (0, 1)
                ]
            )
            if label == "capture-smoke":
                comparison = compare_reference(
                    result, load_cache(reference), nmse_tol, max_abs_tol
                )
                csv_rows(root / "comparison.csv", comparison)
                write_json(root / "comparison.json", comparison)
                if not all(r["passed"] for r in comparison):
                    raise ValueError(
                        "Smoke differs from old capture; formal capture not started"
                    )
            tmp = root / "pending-cache.pt"
            torch.save(result, tmp)
            tmp.replace(root / "cache.pt")

        phase(merge)
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "data", "old-cache", "output", "offload-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reference-cache", type=Path)
    p.add_argument(
        "--ov-only",
        action="store_true",
        help="Only new o_proj inputs; reuse E2 identities",
    )
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reserve-cpu-gib", type=float, default=16)
    p.add_argument("--nmse-tol", type=float, default=1e-5)
    p.add_argument("--max-abs-tol", type=float, default=0.125)
    args = p.parse_args()
    if (
        min(args.context, args.threads, args.reserve_cpu_gib) <= 0
        or min(args.nmse_tol, args.max_abs_tol) < 0
    ):
        p.error("Positive sizes/reserve and nonnegative tolerances required")
    rank, world, local = (
        int(os.environ.get(k, d))
        for k, d in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    )
    if world != 2 or rank not in (0, 1):
        p.error("Use torchrun --nproc-per-node=2: one complete split per rank")
    import torch_npu  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from llmcompressor.utils import load_context

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
    torch.npu.set_device(local)
    device = torch.device("npu", local)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dist.init_process_group(backend="cpu:gloo,npu:hccl", timeout=timedelta(hours=3))
    start = time.perf_counter()
    owned = False
    status = dict(
        status="running",
        rank=rank,
        device=str(device),
        split=("calibration", "validation")[rank],
        git_commit=git_head(),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    try:

        def directories():
            if rank == 0:
                if args.output.exists() or args.offload_dir.exists():
                    raise FileExistsError("Fresh output/offload paths required")
                args.output.mkdir(parents=True)
                args.offload_dir.mkdir(parents=True)

        phase(directories)
        owned = True

        def prepare_data():
            check_memory(2 * args.reserve_cpu_gib)
            cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
            if cfg.model_type != "glm_moe_dsa" or any(
                getattr(cfg, k, None)
                for k in ("quantization_config", "quarot_config", "flex_smooth_config")
            ):
                raise ValueError("Original GLM checkpoint required")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, local_files_only=True
            )
            return split_records(args.data, tokenizer, args.seed)

        splits = phase(prepare_data)
        ids = {
            s: prepare(records, 2048, args.context)[1]
            for s, records in zip(("calibration", "validation"), splits)
        }

        def validate_old():
            if rank == 0:
                old = load_cache(args.old_cache)
                if old["seed"] != args.seed or identities(old) != ids:
                    raise ValueError("Full 2048/2048 ordered identities differ from E1")

        phase(validate_old)

        # Basic free-space check without changing historical loader placement.
        def capacity():
            shm = shutil.disk_usage("/dev/shm")
            total = json.loads(
                (args.model_path / "model.safetensors.index.json").read_text()
            )["metadata"]["total_size"]
            expected = min(total, max(0, shm.total - 5_000_000_000))
            if (
                shm.free < expected
                or psutil.virtual_memory().available
                < expected + 2 * args.reserve_cpu_gib * 2**30
            ):
                raise MemoryError(
                    "Historical shared backing does not fit free SHM/RAM; do not allocate private model copies"
                )
            return dict(
                shm_free=shm.free,
                host_available=psutil.virtual_memory().available,
                offload_free=shutil.disk_usage(args.offload_dir).free,
            )

        status["capacity"] = phase(capacity)
        source = dict(
            world_size=2,
            seed=args.seed,
            model_path=str(args.model_path),
            provenance=provenance(),
            capture_sources={
                n: file_hash(Path(__file__).with_name(n))
                for n in (
                    "e2_collect.py",
                    "distributed_capture.py",
                    "capture.py",
                    "offload.py",
                )
            },
            data_sha256=file_hash(args.data),
            ct_backing_sources=phase(runtime_contract),
        )
        signatures = [None, None]
        dist.all_gather_object(signatures, (source, ids))
        if signatures[0] != signatures[1]:
            raise ValueError("Rank data/runtime/source mismatch")
        status.update(stage="loading", loading=loading_options(args.offload_dir))
        # torch.dtype is represented explicitly in the JSON record.
        status["loading"]["dtype"] = "torch.bfloat16"
        atomic_json(args.output / f"status-rank{rank}.json", status)
        load_start = time.perf_counter()

        def load():
            with load_context():
                return AutoModelForCausalLM.from_pretrained(
                    str(args.model_path), **loading_options(args.offload_dir)
                ).eval()

        model = phase(load)
        sync(device)
        status["load_seconds"] = time.perf_counter() - load_start
        shared = phase(lambda: backing_signature(model))
        backings = [None, None]
        dist.all_gather_object(backings, shared)
        if backings[0] != backings[1]:
            raise ValueError("Ranks do not share backing")
        status.update(stage="capturing", shared_backing=shared, memory=memory(device))
        atomic_json(args.output / f"status-rank{rank}.json", status)
        runner = run_captures
        if args.ov_only:
            from .recipe_mxfp8.capture import run_ov

            runner = run_ov
        status["timings"] = runner(
            model,
            splits,
            source,
            device,
            args.output,
            args.reference_cache or args.old_cache,
            rank,
            args.context,
            lambda: check_memory(args.reserve_cpu_gib),
            args.nmse_tol,
            args.max_abs_tol,
        )
        status.update(
            status="complete",
            wall_seconds=time.perf_counter() - start,
            memory=memory(device),
        )
        phase(lambda: atomic_json(args.output / f"status-rank{rank}.json", status))

        def publish():
            if rank != 0:
                return
            ranks = [
                json.loads((args.output / f"status-rank{i}.json").read_text())
                for i in (0, 1)
            ]
            if not all(s["status"] == "complete" for s in ranks):
                raise ValueError("Incomplete rank")
            csv_rows(
                args.output / "timings.csv",
                [
                    dict(
                        rank=s["rank"],
                        load_seconds=s["load_seconds"],
                        wall_seconds=s["wall_seconds"],
                        **t,
                    )
                    for s in ranks
                    for t in s["timings"]
                ],
            )
            report = dict(
                status="complete",
                ranks=ranks,
                source=source,
                cache_sha256=file_hash(args.output / "capture/cache.pt"),
            )
            (args.output / "report.md").write_text(
                (
                    "# Incremental OV capture\n\nOriginal E2 identities and rank shard validation passed. Native OV compensation is tested separately; no real Attention equivalence measurement is claimed here.\n"
                    if args.ov_only
                    else "# Two-rank capture\n\nOne model load per rank; Smoke comparison and formal shard validation passed.\n"
                )
                + "\nSee timings.csv and complete.json for measured timing, memory and source records.\n",
                encoding="utf-8",
            )
            atomic_json(args.output / "complete.json", report)

        phase(publish)
    except BaseException as exc:
        if owned:
            status.update(status="failed", error=repr(exc))
            atomic_json(args.output / f"status-rank{rank}.json", status)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
