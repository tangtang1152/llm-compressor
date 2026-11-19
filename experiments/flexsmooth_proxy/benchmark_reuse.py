"""Candidate-local activation reuse experiment; production and core.py unchanged."""

import argparse
import contextlib
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from .core import Objective, memory, provenance, qdq, sync

CANDIDATES = ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0))
SHAPES = {
    "Small": (128, 512, 512),
    "Medium": (512, 2048, 2048),
    "Large": (1024, 4096, 4096),
}


class Sections:
    """Only used in a separate diagnostic pass, not headline timing."""

    def __init__(self, device):
        self.device = device
        self.seconds = Counter()
        self.counts = Counter()

    @contextlib.contextmanager
    def section(self, name):
        sync(self.device)
        start = time.perf_counter()
        yield
        sync(self.device)
        self.seconds[name] += time.perf_counter() - start
        self.counts["section_sync_calls"] += 2


@torch.no_grad()
def assess(
    objective, scale, proxy, activation, optimized=True, profile=None, cache_mib=64
):
    """One candidate, device-resident token batches; no cross-candidate cache.

    Preserve storage-dtype scaling on CPU, CT QDQ on execution device, original
    weight-major GEMM order and CPU FP64 loss accumulation. Refuse cache budgets
    rather than silently change loop order or repeatedly spill across devices.
    """
    o = objective
    needed = o.x.numel() * o.x.element_size()
    if optimized and needed > cache_mib * 1024**2:
        raise ValueError(f"Activation cache needs {needed} bytes; increase --cache-mib")
    cache = {}
    counts = profile.counts if profile else Counter()

    def section(name):
        return profile.section(name) if profile else contextlib.nullcontext()

    def activation_batch(i):
        with section("activation_scale_transfer_qdq"):
            counts["activation_scale"] += 1
            counts["activation_to_device"] += 1
            value = (o.x[i : i + o.batch] / scale).to(o.device)
            if proxy and activation:
                counts["activation_qdq"] += 1
                value = qdq(value, proxy, "input_activations")
            return value

    error = energy = 0.0
    max_abs = 0.0
    for j in range(0, len(o.w), o.rows):
        with section("weight_scale_transfer_qdq"):
            counts["weight_scale"] += 1
            counts["weight_to_device"] += 1
            wd = (o.w[j : j + o.rows] * scale).to(o.device)
            if proxy:
                counts["weight_qdq"] += 1
                wd = qdq(wd, proxy)
        for i in range(0, len(o.x), o.batch):
            if optimized:
                if i not in cache:
                    cache[i] = activation_batch(i)
                xd = cache[i]
            else:
                xd = activation_batch(i)
            with section("gemm"):
                counts["gemm"] += 1
                output = xd @ wd.T
            with section("output_transfer_loss"):
                counts["output_to_cpu"] += 1
                y = output.cpu().double()
                r = o.reference[i : i + o.batch, j : j + o.rows].double()
                if not torch.isfinite(y).all():
                    return dict(
                        mse=math.inf,
                        nmse=math.inf,
                        nrmse=math.inf,
                        max_abs=math.inf,
                        finite=False,
                    )
                error += (r - y).square().sum().item()
                energy += r.square().sum().item()
                max_abs = max(max_abs, (r - y).abs().max().item())
                del output
    nmse = error / (energy + 1e-30)
    return dict(
        mse=error / o.reference.numel(),
        nmse=nmse,
        nrmse=math.sqrt(nmse),
        max_abs=max_abs,
        finite=math.isfinite(nmse),
    )


def timed(o, scales, proxy, activation, optimized, cache_mib):
    sync(o.device)
    start = time.perf_counter()
    values = [
        (
            assess(o, s, proxy, activation, cache_mib=cache_mib)
            if optimized
            else o.assess(s, proxy, activation)
        )
        for s in scales
    ]
    sync(o.device)
    return (time.perf_counter() - start) / len(scales), values


def selection(losses):
    # Ascending loss, later candidate wins equal values, consistent with <=.
    return sorted(range(len(losses)), key=lambda i: (losses[i], -i))


def benchmark(o, label, cache_mib):
    scales = [o.amax.pow(a) * o.wmax.pow(-b) for a, b in CANDIDATES]
    result = []
    for proxy in ("INT8", "MXFP4", "MXFP8"):
        for activation in (False, True):
            samples, outputs = {False: [], True: []}, {}
            for optimized in (False, True):
                timed(o, scales, proxy, activation, optimized, cache_mib)  # warmup
            for repeat in range(3):
                for optimized in (False, True) if repeat % 2 == 0 else (True, False):
                    seconds, values = timed(
                        o, scales, proxy, activation, optimized, cache_mib
                    )
                    samples[optimized].append(seconds)
                    outputs[optimized] = values
            left = [v["nmse"] for v in outputs[False]]
            right = [v["nmse"] for v in outputs[True]]
            differences = [abs(a - b) for a, b in zip(left, right)]
            if not all(math.isfinite(v) for v in left + right):
                raise ValueError("Nonfinite candidate")
            if any(d > 1e-12 + 1e-9 * abs(a) for d, a in zip(differences, left)):
                raise ValueError("Numerical mismatch")
            if selection(left) != selection(right):
                raise ValueError("Candidate ranking changed")
            for optimized in (False, True):
                # Separately warmed, synchronized section timing: three repeats.
                profiles = []
                for _ in range(3):
                    profile = Sections(o.device)
                    for s, expected in zip(scales, outputs[optimized]):
                        actual = assess(
                            o, s, proxy, activation, optimized, profile, cache_mib
                        )
                        if actual != expected:
                            raise ValueError(
                                "Instrumented path differs from timed path"
                            )
                    profiles.append(profile)
                per_candidate = {
                    k: v / len(scales) for k, v in profiles[0].counts.items()
                }
                row = dict(
                    shape=label,
                    proxy=proxy,
                    objective="W+A" if activation else "W-only",
                    version="Optimized" if optimized else "Baseline",
                    seconds=statistics.median(samples[optimized]),
                    repeats=samples[optimized],
                    nmse=right if optimized else left,
                    max_nmse_difference=max(differences),
                    candidate_ranking=selection(left),
                    selected_pair=CANDIDATES[selection(left)[0]],
                    counts=per_candidate,
                    sections={
                        k: statistics.median(
                            p.seconds[k] / len(scales) for p in profiles
                        )
                        for k in profiles[0].seconds
                    },
                    cache_bytes=o.x.numel() * o.x.element_size() if optimized else 0,
                    memory=memory(o.device),
                )
                result.append(row)
            print(
                f'{label} {proxy} {"W+A" if activation else "W-only"} done', flush=True
            )
    return result


def report(rows, metadata, output):
    lines = [
        "# FlexSmooth activation reuse benchmark",
        "",
        f"Device: {metadata['device']}; BF16, threads={metadata['threads']}, "
        f"batch={metadata['batch']}, rows={metadata['rows']}. Seed=42.",
        "",
        "Three representative pairs: (0,1), (0.5,0.5), (1,0). One warmup sweep, "
        "three timed sweeps; median seconds per candidate. No full search.",
        "",
        "| Size | Proxy | Objective | Baseline ms | Optimized ms | Speedup | A QDQ B→O | W QDQ | GEMM | Max NMSE diff |",  # noqa: E501
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, o in zip(rows[::2], rows[1::2]):
        lines.append(
            f"| {b['shape']} | {b['proxy']} | {b['objective']} | "
            f"{b['seconds']*1000:.3f} | {o['seconds']*1000:.3f} | "
            f"{b['seconds']/o['seconds']:.2f}x | "
            f"{b['counts'].get('activation_qdq',0):g}→{o['counts'].get('activation_qdq',0):g} | "  # noqa: E501
            f"{o['counts']['weight_qdq']:g} | {o['counts']['gemm']:g} | {o['max_nmse_difference']:.3g} |"  # noqa: E501
        )
    lines += [
        "",
        "| Size | Proxy | Baseline W+A/W-only | Optimized W+A/W-only |",
        "|---|---|---:|---:|",
    ]
    for i in range(0, len(rows), 4):
        bw, ow, ba, oa = rows[i : i + 4]
        lines.append(
            f"| {bw['shape']} | {bw['proxy']} | {ba['seconds']/bw['seconds']:.2f} | {oa['seconds']/ow['seconds']:.2f} |"  # noqa: E501
        )
    lines += [
        "",
        "## Separate instrumented timing (ms/candidate)",
        "",
        "| Size | Proxy | Objective | Version | A scale/copy/QDQ | W scale/copy/QDQ | GEMM | Output copy/loss |",  # noqa: E501
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        # Explicit column order independent of when activation first executes.
        parts = " | ".join(
            f"{row['sections'][k]*1000:.3f}"
            for k in (
                "activation_scale_transfer_qdq",
                "weight_scale_transfer_qdq",
                "gemm",
                "output_transfer_loss",
            )
        )
        lines.append(
            f"| {row['shape']} | {row['proxy']} | {row['objective']} | {row['version']} | {parts} |"  # noqa: E501
        )
    opt_wa = [
        r for r in rows if r["version"] == "Optimized" and r["objective"] == "W+A"
    ]
    dominant = Counter(max(r["sections"], key=r["sections"].get) for r in opt_wa)
    loss_share = [
        r["sections"]["output_transfer_loss"] / sum(r["sections"].values())
        for r in opt_wa
    ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        f"Dominant optimized W+A section counts: {dict(dominant)}. "
        f"Output-copy/loss fraction spans {min(loss_share):.1%}–{max(loss_share):.1%} in instrumented passes.",  # noqa: E501
        "All sampled NMSEs and candidate rankings agree; selected pairs are best ONLY among the three measured pairs. "  # noqa: E501
        "Full two-stage final alpha/beta was not measured. Identical per-candidate operations and reduction order "  # noqa: E501
        "preserve the search rule, but this is not evidence from a complete search.",
        "With R weight slabs and T token batches: baseline A scaling/QDQ=R*T (QDQ=0 in W-only), "  # noqa: E501
        "optimized=T; W scaling/QDQ=R and GEMM=R*T in both. CPU→device .to calls: "
        "baseline R+R*T, optimized R+T; output .cpu calls remain R*T. On CPU these copies are no-ops. "  # noqa: E501
        "On accelerators output .cpu introduces R*T blocking transfers; baseline assess has no explicit sync. "  # noqa: E501
        "Headline timing adds two sync calls per candidate sweep, outside assess. Instrumentation adds two "  # noqa: E501
        "sync calls per section (see JSON counts); CPU sync is a no-op. Scalar reads occur on CPU.",  # noqa: E501
        "Only activation batches are retained for one candidate. Cache payload is recorded, capped by --cache-mib; "  # noqa: E501
        "oversized caches fail rather than spill or alter reduction order. Extra payload: Small 0.125 MiB, "  # noqa: E501
        "Medium 2 MiB, Large 8 MiB. QDQ/GEMM temporary memory remains batch/slab bounded. "  # noqa: E501
        "CPU peak RSS is process-lifetime, not a per-version allocation measurement. "
        "Accelerator peaks, if available, are also process-lifetime counters.",
        "Headline baseline directly calls the original Objective.assess. The experimental path retains lightweight "  # noqa: E501
        "Python counters/null contexts but performs no section clocks or extra synchronization when uninstrumented. "  # noqa: E501
        "Section measurements are diagnostic and cannot be added to reproduce headline timing. "  # noqa: E501
        "QDQ and input-copy costs are grouped; no claim isolates pure transfer from QDQ on accelerator.",  # noqa: E501
        f"On this measured device, eliminating the output/loss section entirely would give only "  # noqa: E501
        f"{1/(1-min(loss_share)):.2f}–{1/(1-max(loss_share)):.2f}x ideal total speedup "
        "from these section fractions. CPU loss reduction is therefore low priority here; "  # noqa: E501
        "the NPU output-copy/reduction fraction needs a separate measurement.",
        "Next: output transfer/loss is worth a separate lightweight device-side reduction experiment if it is a "  # noqa: E501
        "material measured share on the target NPU; this CPU run alone cannot establish NPU payoff or explain "  # noqa: E501
        "the previously observed 3.0–3.4x ratio. Preserve finite checks and sufficient accumulation precision.",  # noqa: E501
        f"NPU: {metadata['npu_status']}. CPU speedups do not predict NPU speedups.",
        f"Skipped sizes: {metadata['skipped']}. No capture, model load, quantized export or GPQA.",  # noqa: E501
    ]
    (output / "report.md").write_text(
        "\n\n".join(lines[:1]) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps(dict(metadata=metadata, results=rows), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "shape",
            "proxy",
            "objective",
            "version",
            "seconds",
            "max_nmse_difference",
            "cache_bytes",
        ]
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sizes", nargs="+", choices=list(SHAPES), default=["Small", "Medium"]
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-mib", type=float, default=64)
    p.add_argument("--cache", type=Path)
    p.add_argument("--mapping", default="model.layers.3.input_layernorm")
    args = p.parse_args()
    if min(args.batch, args.rows, args.threads, args.cache_mib) <= 0:
        p.error("Budgets must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(torch.device(args.device))
    npu = hasattr(torch, "npu") and torch.npu.is_available()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    data = []
    if args.cache:
        cache = torch.load(args.cache, map_location="cpu", weights_only=True)
        g = cache["groups"][args.mapping]
        data = [(args.mapping, g["calibration"], g["weight"])]
    else:
        for name in args.sizes:
            n, m, k = SHAPES[name]
            generator = torch.Generator().manual_seed(42)
            data.append(
                (
                    name,
                    torch.randn(n, k, generator=generator).bfloat16(),
                    torch.randn(m, k, generator=generator).bfloat16(),
                )
            )
    rows = []
    for label, x, w in data:
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise ValueError("Benchmark requires BF16 cache tensors; no implicit cast")
        rows.extend(
            benchmark(
                Objective(x, w, args.device, args.batch, args.rows),
                label,
                args.cache_mib,
            )
        )
    report(
        rows,
        dict(
            device=args.device,
            threads=args.threads,
            batch=args.batch,
            rows=args.rows,
            provenance=provenance(),
            candidate_pairs=CANDIDATES,
            npu_status="available locally; only requested device tested"
            if npu
            else "unavailable locally; skipped, no remote connection",
            skipped=[s for s in SHAPES if s not in args.sizes]
            if not args.cache
            else list(SHAPES),
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
