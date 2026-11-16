"""Separate capture / search / evaluate stages. Outputs never contain quantized weights."""  # noqa: E501

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import torch

from .core import Objective, digest, memory, provenance, sync, validate_cache


def clean(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def write_json(path, value):
    path.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def cache_hashes(cache):
    return {
        name: {key: digest(g[key]) for key in ("weight", "calibration", "validation")}
        for name, g in cache["groups"].items()
    }


def synthetic(tokens=128, seed=42, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(48, 64, generator=gen).to(dtype)
    x = torch.randn(tokens, 64, generator=gen).to(dtype)
    v = torch.randn(tokens, 64, generator=gen).to(dtype)
    return dict(
        schema=1,
        kind="synthetic",
        capture_seconds=0,
        seed=seed,
        groups={
            "model.layers.3.input_layernorm": dict(
                weight=w,
                calibration=x,
                validation=v,
                consumers=["synthetic.q_a_proj", "synthetic.kv_a_proj"],
                consumer_rows=[32, 16],
                calibration_ids=[f"cal:{i}" for i in range(tokens)],
                validation_ids=[f"val:{i}" for i in range(tokens)],
            )
        },
    )


def search(cache, args):
    results = {}
    for name, g in cache["groups"].items():
        sync(args.device)
        start = time.perf_counter()
        objective = Objective(
            g["calibration"], g["weight"], args.device, args.batch, args.rows
        )
        sync(args.device)
        setup = time.perf_counter() - start
        results[name] = dict(
            reference_seconds=setup,
            runs=[objective.search(p, a) for p in args.proxies for a in (False, True)],
        )
    return dict(
        schema=1,
        kind=cache["kind"],
        hashes=cache_hashes(cache),
        provenance=provenance(),
        reference_policy="storage-dtype GEMM, CPU FP64 error accumulation, eps=1e-30",
        groups=results,
    )


def evaluate(cache, found, args):
    if cache_hashes(cache) != found["hashes"]:
        raise ValueError("Search/cache fingerprint mismatch")
    if found["provenance"] != provenance():
        raise ValueError("Runtime/source changed since search; rerun search")
    report, table, curves = {}, [], []
    markdown = [
        f"# FlexSmooth E0/E1 ({cache['kind']})",
        "",
        "Proxy search loss is not comparable across proxies. Target QDQ is not hardware-kernel validation.",  # noqa: E501
        "",
    ]
    for name, g in cache["groups"].items():
        sync(args.device)
        start = time.perf_counter()
        objective = Objective(
            g["validation"], g["weight"], args.device, args.batch, args.rows
        )
        runs = [
            dict(
                proxy="No Smooth",
                objective="baseline",
                alpha=None,
                beta=None,
                scale=torch.ones_like(objective.amax),
                search_seconds=0,
            )
        ] + found["groups"][name]["runs"]
        values = []
        for run in runs:
            entry = {
                k: v
                for k, v in run.items()
                if k not in ("scale", "candidates", "memory")
            }
            entry["e0"] = objective.assess(run["scale"])
            # Dtype-dependent algebra tolerance, not a quality threshold.
            tolerance = 64 * torch.finfo(g["weight"].dtype).eps ** 2
            entry["e0"]["pass"] = (
                entry["e0"]["finite"] and entry["e0"]["nmse"] <= tolerance
            )
            entry["e0"]["nmse_tolerance"] = tolerance
            for fmt, p in (("W4A4", "MXFP4"), ("W8A8", "MXFP8")):
                entry[fmt] = objective.assess(run["scale"], p, True)
                if not entry[fmt]["finite"]:
                    raise ValueError(f"{name}: nonfinite target evaluation")
            values.append(entry)
            curves.extend(
                dict(mapping=name, proxy=run["proxy"], objective=run["objective"], **c)
                for c in run.get("candidates", [])
            )
        for e in values:
            for fmt in ("W4A4", "W8A8"):
                base = values[0][fmt]["nmse"]
                e[fmt]["reduction_vs_no_smooth"] = (
                    (base - e[fmt]["nmse"]) / base if base else None
                )
            table.append(
                dict(
                    mapping=name,
                    proxy=e["proxy"],
                    objective=e["objective"],
                    alpha=e["alpha"],
                    beta=e["beta"],
                    W4A4_nmse=e["W4A4"]["nmse"],
                    W8A8_nmse=e["W8A8"]["nmse"],
                    W4A4_reduction=e["W4A4"]["reduction_vs_no_smooth"],
                    W8A8_reduction=e["W8A8"]["reduction_vs_no_smooth"],
                    search_seconds=e["search_seconds"],
                )
            )
        comparisons = []
        for left, right in itertools.combinations(runs[1:], 2):
            ratio = torch.log2(right["scale"].double() / left["scale"].double())
            left_entry = next(
                e
                for e in values
                if (e["proxy"], e["objective"]) == (left["proxy"], left["objective"])
            )
            r = next(
                e
                for e in values
                if (e["proxy"], e["objective"]) == (right["proxy"], right["objective"])
            )
            comparisons.append(
                dict(
                    left=f"{left['proxy']} {left['objective']}",
                    right=f"{right['proxy']} {right['objective']}",
                    delta_alpha=right["alpha"] - left["alpha"],
                    delta_beta=right["beta"] - left["beta"],
                    same_parameters=(left["alpha"], left["beta"])
                    == (right["alpha"], right["beta"]),
                    log2_scale_rms=ratio.square().mean().sqrt().item(),
                    log2_scale_max_abs=ratio.abs().max().item(),
                    time_ratio=right["search_seconds"] / left["search_seconds"],
                    **{
                        fmt + "_relative_gain": (
                            left_entry[fmt]["nmse"] - r[fmt]["nmse"]
                        )
                        / left_entry[fmt]["nmse"]
                        if left_entry[fmt]["nmse"]
                        else None
                        for fmt in ("W4A4", "W8A8")
                    },
                )
            )
        flatness = []
        for run in runs[1:]:
            for stage in ("alpha", "beta"):
                losses = [
                    c["loss"]
                    for c in run["candidates"]
                    if c["stage"] == stage and math.isfinite(c["loss"])
                ]
                low = min(losses) if losses else None
                flatness.append(
                    dict(
                        proxy=run["proxy"],
                        objective=run["objective"],
                        stage=stage,
                        finite_count=len(losses),
                        relative_range=(max(losses) - low) / max(low, 1e-30)
                        if losses
                        else None,
                        within_one_percent=sum(v <= low * 1.01 + 1e-15 for v in losses)
                        if losses
                        else 0,
                    )
                )
        sync(args.device)
        report[name] = dict(
            results=values,
            comparisons=comparisons,
            flatness=flatness,
            evaluation_seconds=time.perf_counter() - start,
            memory=memory(args.device),
        )
        markdown += [
            f"## {name}",
            "",
            "| Search Proxy | Objective | α | β | W4A4 NMSE | W8A8 NMSE | Search Time (s) |",  # noqa: E501
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for e in values:
            markdown.append(
                f"| {e['proxy']} | {e['objective']} | {e['alpha']} | {e['beta']} | {e['W4A4']['nmse']:.6g} | {e['W8A8']['nmse']:.6g} | {e['search_seconds']:.3f} |"  # noqa: E501
            )
        markdown += ["", f"E0 all pass: {all(e['e0']['pass'] for e in values)}.", ""]
        for c in comparisons:
            if c["left"].split()[0] == c["right"].split()[0]:
                markdown.append(
                    f"- {c['left']} → {c['right']}: Δα={c['delta_alpha']:.2f}, Δβ={c['delta_beta']:.2f}; log2-scale RMS={c['log2_scale_rms']:.4g}; W4A4 gain={c['W4A4_relative_gain']}; W8A8 gain={c['W8A8_relative_gain']}; time ratio={c['time_ratio']:.3g}."  # noqa: E501
                )
        markdown += [
            "",
            "INT8 reliability requires consistent held-out improvements across real mappings/seeds. Minimum proxy losses alone cannot establish it. Block NMSE is a separate future gate; synthetic results do not justify it.",  # noqa: E501
            "",
        ]
    out = Path(args.output)
    write_json(
        out / "evaluation.json",
        dict(
            kind=cache["kind"],
            capture_seconds=cache.get("capture_seconds"),
            groups=report,
            provenance=provenance(),
        ),
    )
    for filename, rows in [("summary.csv", table), ("candidates.csv", curves)]:
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(clean(rows))
    (out / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    # Dependency-free SVG curves, one panel per proxy/objective/stage; nulls excluded.
    panels = []
    for index, ((name, p, o, stage), items) in enumerate(
        itertools.groupby(
            sorted(
                curves,
                key=lambda c: (c["mapping"], c["proxy"], c["objective"], c["stage"]),
            ),
            key=lambda c: (c["mapping"], c["proxy"], c["objective"], c["stage"]),
        )
    ):
        items = [c for c in items if math.isfinite(c["loss"])]
        top = index * 130
        high = max((c["loss"] for c in items), default=1) or 1
        points = " ".join(
            f"{20+i*28},{top+115-c['loss']/high*80}" for i, c in enumerate(items)
        )
        panels.append(
            f'<text x="20" y="{top+20}">{name} {p} {o} {stage} (NRMSE 0..{high:.4g})</text><polyline fill="none" stroke="blue" points="{points}"/>'  # noqa: E501
        )
    (out / "loss_curves.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{len(panels)*130}">'  # noqa: E501
        + "".join(panels)
        + "</svg>",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["capture", "synthetic", "search", "evaluate"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--search")
    parser.add_argument("--model-path")
    parser.add_argument("--data")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 40])
    parser.add_argument(
        "--mappings",
        nargs="+",
        choices=["input_layernorm", "q_a_layernorm"],
        default=["input_layernorm", "q_a_layernorm"],
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--proxies",
        nargs="+",
        choices=["INT8", "MXFP4", "MXFP8"],
        default=["INT8", "MXFP4", "MXFP8"],
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()
    if min(args.batch, args.rows, args.tokens, args.context) <= 0:
        parser.error("Sizes must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Output directory must be empty (protect prior results)")
    if args.stage == "synthetic":
        cache = synthetic(args.tokens, args.seed, getattr(torch, args.dtype))
        torch.save(cache, out / "cache.pt")
    elif args.stage == "capture":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .capture import capture, split_records

        if not args.model_path or not args.data:
            parser.error("capture needs --model-path and --data")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
        splits = split_records(args.data, tokenizer, args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device_map,
            attn_implementation="eager",
        ).eval()
        capture(
            model,
            splits,
            out / "cache.pt",
            args.layers,
            args.mappings,
            args.tokens,
            args.context,
            args.device,
            args.model_path,
            args.seed,
        )
    else:
        cache = validate_cache(
            torch.load(args.cache, map_location="cpu", weights_only=True)
        )
        if args.stage == "search":
            found = search(cache, args)
            torch.save(found, out / "search.pt")
            write_json(out / "search.json", found)
        else:
            evaluate(
                cache,
                torch.load(args.search, map_location="cpu", weights_only=True),
                args,
            )


if __name__ == "__main__":
    main()
