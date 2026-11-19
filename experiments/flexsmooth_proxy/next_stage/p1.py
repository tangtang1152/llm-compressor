"""P1: real-cache NPU benchmark using the existing two candidate evaluators."""

import importlib.util

import torch

from ..benchmark_reuse import benchmark, report
from ..core import Objective, sync
from ..run import write_json
from .common import csv_rows, file_hash, load_cache


def run(args, out, manifest):
    if importlib.util.find_spec("torch_npu") is None:
        manifest.update(
            status="blocked",
            reason="No local torch_npu; no server connection attempted",
        )
        write_json(out / "results.json", manifest)
        csv_rows(
            out / "summary.csv", [dict(status="blocked", reason=manifest["reason"])]
        )
        (out / "report.md").write_text(
            "# P1 blocked\n\nNo local NPU runtime. No NPU result is measured. Run the documented P1 command on the server with the existing capture.\n",  # noqa: E501
            encoding="utf-8",
        )
        return
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch_npu installed but no usable local NPU")
    torch.npu.set_device(torch.device(args.device))
    cache = load_cache(args.cache)
    rows = []
    manifest.update(
        cache_sha256=file_hash(args.cache),
        capture_provenance=cache.get("provenance"),
        memory_scope="peak allocated/reserved since each mapping start, includes reference; not per-version",  # noqa: E501
    )
    for suffix in ("input_layernorm", "self_attn.q_a_layernorm"):
        name = "model.layers.3." + suffix
        g = cache["groups"][name]
        if g["calibration"].dtype != g["weight"].dtype:
            raise ValueError("Activation/weight dtypes differ")
        needed = g["calibration"].numel() * g["calibration"].element_size()
        if needed > args.cache_mib * 1024**2:
            raise ValueError(
                f"{name} cache payload {needed} exceeds --cache-mib; no silent fallback"
            )
        torch.npu.reset_peak_memory_stats(torch.device(args.device))
        obj = Objective(
            g["calibration"], g["weight"], args.device, args.batch, args.rows
        )
        # Extra full arm warmup happens within benchmark; do not count reference init.
        sync(args.device)
        group = benchmark(obj, name, args.cache_mib)
        for r in group:
            r["dtype"] = str(g["weight"].dtype)
        rows.extend(group)
        target = out / suffix.replace(".", "_")
        target.mkdir()
        report(
            group,
            dict(
                device=args.device,
                threads=args.threads,
                batch=args.batch,
                rows=args.rows,
                npu_status="measured on local Ascend",
                skipped=[],
                provenance=manifest,
            ),
            target,
        )
        text = (
            (target / "report.md")
            .read_text(encoding="utf-8")
            .replace("BF16, threads=", f"{g['weight'].dtype}, threads=")
        )
        measured = [
            r for r in group if r["version"] == "Optimized" and r["objective"] == "W+A"
        ]
        shares = [
            r["sections"]["output_transfer_loss"] / sum(r["sections"].values())
            for r in measured
        ]
        text = text.split("## Interpretation and limits")[0]
        text += (
            "## NPU interpretation\n\n"
            f"Optimized W+A output-copy/loss share: {min(shares):.1%}–{max(shares):.1%}. "  # noqa: E501
            "Stage sums are diagnostic synchronized timings, not the unsynchronized headline total. "  # noqa: E501
            "These are NPU measurements; CPU benchmark speedups are not used. "
            "Three sampled scales agree numerically and in ranking; no full search was performed. "  # noqa: E501
            "Cache payload is bounded by --cache-mib. Peak memory is since this mapping started "  # noqa: E501
            "(both versions and reference), not per-version. Output transfer and FP64 CPU loss "  # noqa: E501
            "remain grouped; use their measured share to prioritize a separate reduction study.\n"  # noqa: E501
        )
        (target / "report.md").write_text(text, encoding="utf-8")
    table = []
    for i in range(0, len(rows), 4):
        bw, ow, ba, oa = rows[i : i + 4]
        for b, o in ((bw, ow), (ba, oa)):
            table.append(
                dict(
                    mapping=b["shape"],
                    proxy=b["proxy"],
                    objective=b["objective"],
                    dtype=b["dtype"],
                    baseline_seconds=b["seconds"],
                    optimized_seconds=o["seconds"],
                    speedup=b["seconds"] / o["seconds"],
                    baseline_wa_over_w=ba["seconds"] / bw["seconds"],
                    optimized_wa_over_w=oa["seconds"] / ow["seconds"],
                    nmse_max_difference=o["max_nmse_difference"],
                    activation_qdq_baseline=b["counts"].get("activation_qdq", 0),
                    activation_qdq_optimized=o["counts"].get("activation_qdq", 0),
                    weight_qdq=o["counts"]["weight_qdq"],
                    gemm=o["counts"]["gemm"],
                    cache_bytes=o["cache_bytes"],
                )
            )
    manifest["status"] = "complete"
    csv_rows(out / "summary.csv", table)
    write_json(out / "results.json", dict(metadata=manifest, results=rows))
    lines = [
        "# P1 NPU benchmark",
        "",
        "Real cached L3 mappings; three candidates, warmup and three repeats, synchronized timing. Native cache dtype preserved.",  # noqa: E501
        "",
        "| Mapping | Proxy | Objective | Speedup | Baseline WA/W | Optimized WA/W | Max NMSE difference |",  # noqa: E501
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['mapping']} | {r['proxy']} | {r['objective']} | {r['speedup']:.3f} | "  # noqa: E501
            f"{r['baseline_wa_over_w']:.3f} | {r['optimized_wa_over_w']:.3f} | {r['nmse_max_difference']:.3g} |"  # noqa: E501
        )
    lines += [
        "",
        "See per-mapping reports for stage timing and memory scope. All numerical and ranking checks passed. CPU results are not used to infer these measurements.",  # noqa: E501
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
