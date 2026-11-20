"""QuaRot E2 orchestration adapted to independent FlexSmooth whole-prefill shards.

Source design: quarot-experiments e2_collect/e2_sequential. Model execution reuses
FlexSmooth SequentialRunner (already adapted from QuaRot E2). No model parallelism.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import socket
import threading
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
    split_for_rank,
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


def resources(args, device):
    if not Path("/dev/shm").is_dir():
        raise ValueError(
            "Distributed shared backing requires Linux /dev/shm; no private-copy fallback"
        )
    shared = int(args.shared_cpu_gib * 2**30)
    private = int(args.private_cpu_gib * 2**30)
    index = json.loads((args.model_path / "model.safetensors.index.json").read_text())
    # Bound tensor/page overhead and mmap count conservatively, including buffers.
    tensors = 2 * len(index["weight_map"]) + 4096
    overhead = tensors * os.sysconf("SC_PAGE_SIZE")
    map_limit = int(Path("/proc/sys/vm/max_map_count").read_text())
    current_maps = len(Path("/proc/self/maps").read_text().splitlines())
    if current_maps + 2 * tensors > map_limit:
        raise ValueError("Insufficient mmap capacity for distributed shared backing")
    free = psutil.virtual_memory().available
    if (
        shared + overhead > shutil.disk_usage("/dev/shm").free
        or 2 * shared + dist.get_world_size() * private + overhead > free
    ):
        raise ValueError(
            "Insufficient free SHM/RAM for shared backing, load transient and all rank private budgets"
        )
    total = int(index["metadata"]["total_size"])
    # Layer placement may leave unused CPU budget; allow a full disk backing.
    disk_needed = 2 * total
    disk_free = shutil.disk_usage(args.offload_dir).free
    output_free = shutil.disk_usage(args.output).free
    output_reserve = int(args.output_reserve_gib * 2**30)
    if args.output.stat().st_dev == args.offload_dir.stat().st_dev:
        disk_needed += output_reserve
    if output_free < output_reserve:
        raise ValueError("Insufficient output disk capacity")
    if disk_free < disk_needed:
        raise ValueError(
            f"Conservative offload/output storage requirement {disk_needed} exceeds free {disk_free}"
        )
    npu_free, npu_total = torch.npu.mem_get_info(device)
    budget = int(args.npu_gib * 2**30)
    if npu_free < budget:
        raise ValueError("Chosen NPU lacks free space for explicit per-rank budget")
    torch.npu.set_per_process_memory_fraction(budget / npu_total, device)
    return dict(
        shared_cpu_bytes=shared,
        private_cpu_bytes_per_rank=private,
        host_available=free,
        shm_free=shutil.disk_usage("/dev/shm").free,
        disk_needed=disk_needed,
        disk_free=disk_free,
        output_free=output_free,
        estimated_tensor_overhead=overhead,
        mmap_limit=map_limit,
        npu_budget=budget,
        npu_free=npu_free,
    )


class MemoryGuard:
    """Non-SHM PSS watchdog; source loading gets a bounded transient allowance.

    A breach terminates only this worker; torchrun tears down its peer. No final
    cache is published before all workers' success handshake.
    """

    def __init__(self, args, rank):
        self.args, self.rank = args, rank
        self.loading = True
        self.stop = threading.Event()
        self.peak_uss = 0
        self.peak_rss = 0
        self.peak_non_shm_pss = 0
        self.load_start = None

    def check(self):
        mem = psutil.Process().memory_full_info()
        self.peak_uss = max(self.peak_uss, mem.uss)
        self.peak_rss = max(self.peak_rss, mem.rss)
        # USS may count MAP_SHARED pages as private before other ranks touch
        # them. Do not mistake untouched CT backing for duplicated model RAM.
        rollup = dict(
            line.split(":", 1)
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
            if ":" in line
        )
        non_shm_pss = sum(
            int(rollup[k].split()[0]) * 1024 for k in ("Pss_Anon", "Pss_File")
        )
        self.peak_non_shm_pss = max(self.peak_non_shm_pss, non_shm_pss)
        limit = (
            self.args.private_cpu_gib
            + (self.args.shared_cpu_gib if self.loading and self.rank == 0 else 0)
        ) * 2**30
        timed_out = (
            self.load_start is not None
            and self.loading
            and time.perf_counter() - self.load_start > self.args.max_load_seconds
        )
        if non_shm_pss > limit or timed_out:
            atomic_json(
                self.args.output / f"budget-failure-rank{self.rank}.json",
                dict(
                    status="failed",
                    uss=mem.uss,
                    non_shm_pss=non_shm_pss,
                    limit=limit,
                    loading=self.loading,
                    load_timed_out=timed_out,
                ),
            )
            os._exit(86)

    def __enter__(self):
        self.check()

        def watch():
            while not self.stop.wait(0.5):
                try:
                    self.check()
                except Exception as exc:
                    atomic_json(
                        self.args.output / f"watchdog-failure-rank{self.rank}.json",
                        dict(status="failed", error=repr(exc)),
                    )
                    os._exit(87)

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


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


def require_gate(path, signature, kind):
    if path is None:
        raise ValueError(f"--{kind}-gate required")
    gate = json.loads(path.read_text())
    if (
        gate.get("status") != "complete"
        or gate.get("mode") != kind
        or gate.get("signature") != signature
    ):
        raise ValueError(
            f"{kind} gate missing/incompatible; rerun with current source/runtime/budgets"
        )
    if kind == "smoke" and not gate.get("comparison_passed"):
        raise ValueError("Smoke comparison did not pass")


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


def report_text(report):
    lines = [
        f"# FlexSmooth distributed {report['mode']}",
        "",
        f"Status: {report['status']}; active forward ranks: {report['active_ranks']}",
        "",
        "| Rank | Split | Device | Active | Tokens | Load s | Capture s | Peak non-SHM PSS GiB | Peak NPU allocated GiB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["ranks"]:
        npu_peak = s["memory"].get("device_peak_allocated_bytes", 0)
        lines.append(
            f"| {s['rank']} | {s['split']} | {s['device']} | {s.get('active', False)} | {s.get('token_count', 0)} | {s['load_seconds']:.3f} | {s.get('capture_seconds', 0):.3f} | {s['peak_non_shm_pss_bytes'] / 2**30:.3f} | {npu_peak / 2**30:.3f} |"
        )
    lines += [
        "",
        "Each split preserves original complete prefills and token order. Inactive ranks do not run model forward.",
        "Load includes rank readiness handshake; capture excludes shard serialization/merge. Peaks include runtime allocations.",
        "See timings.csv for weight snapshot, trace and forward times; complete.json for source hashes, budgets and rank status.",
    ]
    if report["mode"] == "smoke":
        lines += [
            "",
            f"Activation comparison passed: {report['comparison_passed']}; tolerances: {report['comparison_tolerances']}.",
            "See comparison.csv for measured NMSE/max_abs and bitwise equality per mapping/split.",
        ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["plan", "load", "smoke", "formal"], required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--old-cache", type=Path, required=True)
    p.add_argument("--reference-cache", type=Path)
    p.add_argument("--load-gate", type=Path)
    p.add_argument("--smoke-gate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offload-dir", type=Path, required=True)
    p.add_argument("--shared-cpu-gib", type=float, required=True)
    p.add_argument("--private-cpu-gib", type=float, required=True)
    p.add_argument("--npu-gib", type=float, required=True)
    p.add_argument("--output-reserve-gib", type=float, required=True)
    p.add_argument("--nmse-tol", type=float, default=1e-5)
    p.add_argument("--max-abs-tol", type=float, default=0.125)
    p.add_argument("--max-load-seconds", type=float, default=600)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    args = p.parse_args()
    rank, world, local = (
        int(os.environ.get(k, d))
        for k, d in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    )
    split_for_rank(rank, world)
    if any(
        getattr(args, k) <= 0
        for k in (
            "shared_cpu_gib",
            "private_cpu_gib",
            "npu_gib",
            "output_reserve_gib",
            "max_load_seconds",
            "context",
            "threads",
            "timeout_seconds",
        )
    ):
        p.error("Positive budgets required")
    if args.nmse_tol < 0 or args.max_abs_tol < 0:
        p.error("Nonnegative comparison tolerances required")
    import torch_npu  # noqa: F401

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
    torch.npu.set_device(local)
    device = torch.device("npu", local)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dist.init_process_group(
        backend="cpu:gloo,npu:hccl", timeout=timedelta(seconds=args.timeout_seconds)
    )
    status = dict(
        status="running",
        rank=rank,
        split=split_for_rank(rank, world),
        world_size=world,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        local_rank=local,
        device=str(device),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        git_commit=git_head(),
    )
    owned = False
    try:
        peers = [None] * world
        dist.all_gather_object(peers, (socket.gethostname(), local))
        if len({h for h, _ in peers}) != 1 or len({i for _, i in peers}) != world:
            raise ValueError("Distinct local devices on one host required")

        def directories():
            if rank == 0:
                if args.output.exists() or args.offload_dir.exists():
                    raise FileExistsError("Fresh output and offload paths required")
                args.output.mkdir(parents=True)
                args.offload_dir.mkdir(parents=True)

        phase(directories)
        owned = True
        budget = phase(lambda: resources(args, device))
        contract = phase(runtime_contract)
        status["budget"] = budget
        atomic_json(args.output / f"status-rank{rank}.json", status)
        with MemoryGuard(args, rank) as guard:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            from llmcompressor.utils import load_context

            def preparation():
                cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
                if cfg.model_type != "glm_moe_dsa" or any(
                    getattr(cfg, k, None)
                    for k in (
                        "quantization_config",
                        "quarot_config",
                        "flex_smooth_config",
                    )
                ):
                    raise ValueError("Original GLM checkpoint required")
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path, local_files_only=True
                )
                return split_records(args.data, tokenizer, args.seed)

            splits = phase(preparation)
            tokens = 2048 if args.mode in ("plan", "formal") else 128
            layers = [6, 16, 60, 77] if args.mode in ("plan", "formal") else [3]
            mappings = (
                ["input_layernorm", "q_a_layernorm"]
                if args.mode in ("plan", "formal")
                else ["input_layernorm"]
            )
            ids = {
                s: prepare(records, tokens, args.context)[1]
                for s, records in zip(("calibration", "validation"), splits)
            }
            if set(ids["calibration"]) & set(ids["validation"]):
                raise ValueError("Calibration/validation identities overlap")

            def old_identity():
                if rank != 0:
                    return None
                old = load_cache(args.old_cache)
                if old["seed"] != args.seed or any(
                    ids[s] != identities(old)[s][:tokens] for s in ids
                ):
                    raise ValueError("Old E1 data identities/seed differ")
                return file_hash(args.old_cache)

            oldhash = phase(old_identity)
            packet = [oldhash]
            dist.broadcast_object_list(packet, src=0)
            signature = dict(
                world_size=world,
                model_config=file_hash(args.model_path / "config.json"),
                model_index=file_hash(args.model_path / "model.safetensors.index.json"),
                model_path=str(args.model_path.resolve()),
                data=file_hash(args.data),
                old_cache=packet[0],
                seed=args.seed,
                context=args.context,
                provenance=provenance(),
                ct_shared_backing_sources=contract,
                runtime_versions={
                    k: importlib.metadata.version(k)
                    for k in ("transformers", "accelerate", "torch-npu")
                },
                budgets=[
                    args.shared_cpu_gib,
                    args.private_cpu_gib,
                    args.npu_gib,
                    args.output_reserve_gib,
                ],
                code={
                    n: file_hash(Path(__file__).with_name(n))
                    for n in (
                        "e2_collect.py",
                        "distributed_capture.py",
                        "capture.py",
                        "offload.py",
                    )
                },
            )
            signatures = [None] * world
            dist.all_gather_object(signatures, signature)
            if any(s != signatures[0] for s in signatures):
                raise ValueError("Rank source/runtime mismatch")
            if args.mode not in ("plan", "load"):
                phase(lambda: require_gate(args.load_gate, signature, "load"))
            if args.mode == "formal":
                phase(lambda: require_gate(args.smoke_gate, signature, "smoke"))
            source = dict(
                signature=signature,
                seed=args.seed,
                model_path=str(args.model_path),
                token_ids=ids,
                world_size=world,
                rank_plan=partition(splits, tokens, args.context, world),
                layers=layers,
                mapping_suffixes=[
                    "input_layernorm"
                    if m == "input_layernorm"
                    else "self_attn.q_a_layernorm"
                    for m in mappings
                ],
                provenance=provenance(),
                capture_sources=signature["code"],
            )
            phase(
                lambda: atomic_json(args.output / "partition.json", source)
                if rank == 0
                else None
            )
            status["assignment"] = source["rank_plan"][str(rank)]
            if args.mode == "plan":
                status.update(
                    status="complete",
                    stage="plan_only",
                    active=bool(status["assignment"]["positions"]),
                    token_count=len(status["assignment"]["positions"]),
                )
                phase(
                    lambda: atomic_json(args.output / f"status-rank{rank}.json", status)
                )

                def publish_plan():
                    if rank == 0:
                        rows = [
                            dict(
                                rank=int(r),
                                split=v["split"],
                                batches=len(v["batch_indices"]),
                                tokens=len(v["positions"]),
                                active=bool(v["positions"]),
                            )
                            for r, v in source["rank_plan"].items()
                        ]
                        csv_rows(args.output / "partition.csv", rows)
                        atomic_json(
                            args.output / "complete.json",
                            dict(
                                status="complete",
                                mode="plan",
                                signature=signature,
                                ranks=rows,
                                active_ranks=[r["rank"] for r in rows if r["active"]],
                            ),
                        )
                        print(json.dumps(rows, indent=2), flush=True)

                phase(publish_plan)
                return
            status["stage"] = "loading"
            atomic_json(args.output / f"status-rank{rank}.json", status)
            start = time.perf_counter()
            guard.load_start = start

            def load():
                with load_context():
                    return AutoModelForCausalLM.from_pretrained(
                        str(args.model_path),
                        local_files_only=True,
                        dtype=torch.bfloat16,
                        attn_implementation="eager",
                        device_map="auto_offload",
                        offload_folder=str(args.offload_dir),
                        max_memory={"cpu": budget["shared_cpu_bytes"]},
                    ).eval()

            model = phase(load)
            sync(device)
            status["load_seconds"] = time.perf_counter() - start
            shared = phase(lambda: backing_signature(model))
            backs = [None] * world
            dist.all_gather_object(backs, shared)
            if any(b != backs[0] for b in backs):
                raise ValueError("Ranks do not share identical backing")
            guard.loading = False
            guard.check()
            phase(
                lambda: (_ for _ in ()).throw(
                    RuntimeError("Load too slow; stop before capture")
                )
                if status["load_seconds"] > args.max_load_seconds
                else None
            )
            status.update(stage="loaded", shared_backing=shared, memory=memory(device))
            atomic_json(args.output / f"status-rank{rank}.json", status)
            comparison = []
            if args.mode != "load":
                status["stage"] = "capturing"
                atomic_json(args.output / f"status-rank{rank}.json", status)
                shard = phase(
                    lambda: capture_split(
                        model,
                        splits[rank % 2],
                        rank,
                        source,
                        layers,
                        mappings,
                        tokens,
                        args.context,
                        device,
                        guard.check,
                    )
                )
                status["capture_seconds"] = shard["capture_seconds"]
                for key in (
                    "forward_seconds",
                    "weight_snapshot_seconds",
                    "trace_seconds",
                ):
                    status[key] = shard[key]
                status["subgraphs"] = shard["subgraphs"]
                status["active"] = shard["active"]
                status["token_count"] = len(shard["positions"])
                status["mapping_hashes"] = {
                    n: {k: g[k] for k in ("weight_hash", "activation_hash")}
                    for n, g in shard["groups"].items()
                }

                def save_shard():
                    temp = args.output / f"rank{rank}.tmp"
                    torch.save(shard, temp)
                    temp.replace(args.output / f"rank{rank}.pt")

                phase(save_shard)
                status["shard_sha256"] = file_hash(args.output / f"rank{rank}.pt")
                del shard, model

                def merge():
                    if rank != 0:
                        return []
                    parts = [
                        torch.load(
                            args.output / f"rank{i}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        for i in range(world)
                    ]
                    result = merge_shards(parts)
                    checks = []
                    if args.mode == "smoke":
                        if args.reference_cache is None:
                            raise ValueError(
                                "Smoke needs --reference-cache from single-process capture"
                            )
                        checks = compare_reference(
                            result,
                            load_cache(args.reference_cache),
                            args.nmse_tol,
                            args.max_abs_tol,
                        )
                        csv_rows(args.output / "comparison.csv", checks)
                        write_json(args.output / "comparison.json", checks)
                        if not all(r["passed"] for r in checks):
                            raise ValueError(
                                "Smoke activation tolerance failed; no final cache published"
                            )
                    temp = args.output / "pending-cache.pt"
                    torch.save(result, temp)
                    return checks

                comparison = phase(merge)
            status.update(
                status="complete",
                stage="rank_work_complete",
                peak_uss_bytes=guard.peak_uss,
                peak_rss_bytes=guard.peak_rss,
                peak_non_shm_pss_bytes=guard.peak_non_shm_pss,
                memory=memory(device),
            )
            phase(lambda: atomic_json(args.output / f"status-rank{rank}.json", status))

            def publish():
                if rank != 0:
                    return
                statuses = [
                    json.loads((args.output / f"status-rank{i}.json").read_text())
                    for i in range(world)
                ]
                if not all(s["status"] == "complete" for s in statuses):
                    raise ValueError("Rank did not complete")
                if args.mode != "load":
                    (args.output / "capture").mkdir()
                    (args.output / "pending-cache.pt").replace(
                        args.output / "capture/cache.pt"
                    )
                report = dict(
                    status="complete",
                    mode=args.mode,
                    signature=signature,
                    comparison_passed=bool(comparison)
                    and all(r["passed"] for r in comparison),
                    comparison_tolerances=dict(
                        nmse=args.nmse_tol, max_abs=args.max_abs_tol
                    ),
                    ranks=statuses,
                    active_ranks=[s["rank"] for s in statuses if s.get("active")],
                )
                if args.mode != "load":
                    report["cache_sha256"] = file_hash(args.output / "capture/cache.pt")
                if args.mode == "smoke":
                    report["reference_cache_sha256"] = file_hash(args.reference_cache)
                csv_rows(
                    args.output / "timings.csv",
                    [
                        {
                            k: s.get(k)
                            for k in (
                                "rank",
                                "split",
                                "device",
                                "active",
                                "token_count",
                                "load_seconds",
                                "capture_seconds",
                                "forward_seconds",
                                "weight_snapshot_seconds",
                                "trace_seconds",
                                "peak_uss_bytes",
                                "peak_rss_bytes",
                                "peak_non_shm_pss_bytes",
                            )
                        }
                        for s in statuses
                    ],
                )
                (args.output / "report.md").write_text(
                    report_text(report),
                    encoding="utf-8",
                )
                atomic_json(args.output / "complete.json", report)

            phase(publish)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        if owned:
            atomic_json(args.output / f"status-rank{rank}.json", status)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
