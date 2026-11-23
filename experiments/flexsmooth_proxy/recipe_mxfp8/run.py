"""Offline recipe E2 search/evaluation and full-search P1 parity (no model load)."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from ..benchmark_reuse import Sections, assess
from ..core import memory, provenance, sync
from ..next_stage.common import (
    csv_rows,
    file_hash,
    identities,
    load_cache,
    source_signature,
)
from ..run import cache_hashes, write_json
from .objective import RecipeObjective, select_group


def audit(cache, checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {}
    for name, group in cache["groups"].items():
        offset, dtypes = 0, {}
        for consumer, rows in zip(group["consumers"], group["consumer_rows"]):
            if consumer + ".bias" in index:
                raise ValueError(
                    "Bias-bearing consumer requires explicit reference handling"
                )
            key = consumer + ".weight"
            with safe_open(
                str(checkpoint / index[key]), framework="pt", device="cpu"
            ) as f:
                original = f.get_tensor(key)
            saved = group["weight"][offset : offset + rows]
            if saved.shape != original.shape or not torch.equal(
                saved, original.to(saved.dtype)
            ):
                raise ValueError(f"Checkpoint/cache mismatch: {consumer}")
            dtypes[consumer] = str(original.dtype)
            offset += rows
        result[name] = select_group(name, group, dtypes)
    return result


def make_objective(g, args, optimized=True, split="calibration"):
    return RecipeObjective(
        g[split],
        g["weight"],
        args.device,
        args.batch,
        args.rows,
        optimized=optimized,
        cache_mib=args.cache_mib,
        geometry=g.get("geometry"),
    )


def parity(a, b):
    # Strict equality also preserves every tie and ranking. Fail closed on drift.
    return (
        a["alpha"] == b["alpha"]
        and a["beta"] == b["beta"]
        and torch.equal(a["scale"], b["scale"])
        and len(a["candidates"]) == len(b["candidates"])
        and all(
            x["alpha"] == y["alpha"]
            and x["beta"] == y["beta"]
            and same_loss(x["loss"], y["loss"])
            for x, y in zip(a["candidates"], b["candidates"])
        )
    )


def same_loss(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def ab(g, name, args, out):
    rows, candidates, records = [], [], []
    for activation in (False, True):
        measured = {False: [], True: []}
        for repeat in range(-1, args.repeats):
            pair = {}
            for optimized in (False, True) if repeat % 2 else (True, False):
                sync(args.device)
                start = time.perf_counter()
                o = make_objective(g, args, optimized)
                sync(args.device)
                setup = time.perf_counter() - start
                if str(args.device).startswith("npu"):
                    torch.npu.reset_peak_memory_stats(args.device)
                search_start = time.perf_counter()
                result = o.search("MXFP8", activation)
                sync(args.device)
                elapsed = time.perf_counter() - search_start
                pair[optimized] = result
                if repeat >= 0:
                    measured[optimized].append(result["search_seconds"])
                    records.append(
                        dict(
                            mapping=name,
                            objective=result["objective"],
                            optimized=optimized,
                            repeat=repeat,
                            reference_seconds=setup,
                            end_to_end_seconds=setup + elapsed,
                            result=result,
                            memory=memory(args.device),
                        )
                    )
                del o
            same = parity(pair[False], pair[True])
            for i, (left, right) in enumerate(
                zip(pair[False]["candidates"], pair[True]["candidates"])
            ):
                candidates.append(
                    dict(
                        mapping=name,
                        objective=pair[False]["objective"],
                        repeat=repeat,
                        candidate=i,
                        alpha=left["alpha"],
                        beta=left["beta"],
                        baseline_loss=left["loss"],
                        optimized_loss=right["loss"],
                        exact=same_loss(left["loss"], right["loss"]),
                    )
                )
            csv_rows(out / "p1_candidate_parity.csv", candidates)
            write_json(out / "ab_runs.json", records)
            if not same:
                raise ValueError(
                    "Full-search parity/ranking/selected scale changed; do not publish optimized results"
                )
        o = make_objective(g, args)
        counts, sections = {}, {}
        for optimized in (False, True):
            profile = Sections(args.device)
            for candidate in pair[optimized]["candidates"]:
                raw = o.amax.pow(candidate["alpha"]) * o.wmax.pow(-candidate["beta"])
                observed = assess(
                    o, raw, "MXFP8", activation, optimized, profile, args.cache_mib
                )
                if not same_loss(observed["nrmse"], candidate["loss"]):
                    raise ValueError("Instrumented candidate changed numerical result")
            counts[optimized], sections[str(optimized)] = (
                dict(profile.counts),
                dict(profile.seconds),
            )
        write_json(
            out / ("sections_wa.json" if activation else "sections_w.json"),
            dict(
                scope="42-candidate instrumented replay, excluded from headline timing",
                seconds=sections,
            ),
        )
        before, after = (statistics.median(measured[v]) for v in (False, True))
        peaks = [
            r["memory"].get("device_peak_allocated_bytes")
            for r in records
            if r["objective"] == pair[False]["objective"]
        ]
        rows.append(
            dict(
                mapping=name,
                objective=pair[False]["objective"],
                baseline_search_s=before,
                optimized_search_s=after,
                speedup=before / after,
                activation_qdq_before=counts[False].get("activation_qdq", 0),
                activation_qdq_after=counts[True].get("activation_qdq", 0),
                count_scope="42 candidates; selected-raw/applied diagnostics add two assessments",
                cache_bytes=g["calibration"].numel() * g["calibration"].element_size(),
                peak_npu_bytes=max((p for p in peaks if p is not None), default=None),
                numerical_parity=True,
                baseline_alpha=pair[False]["alpha"],
                baseline_beta=pair[False]["beta"],
                optimized_alpha=pair[True]["alpha"],
                optimized_beta=pair[True]["beta"],
                scale_max_abs_diff=(
                    pair[False]["scale"].double() - pair[True]["scale"].double()
                )
                .abs()
                .max()
                .item(),
                candidate_max_abs_diff=max(
                    0.0
                    if same_loss(a["loss"], b["loss"])
                    else abs(a["loss"] - b["loss"])
                    for a, b in zip(pair[False]["candidates"], pair[True]["candidates"])
                ),
                baseline_peak_npu_bytes=max(
                    (
                        r["memory"].get("device_peak_allocated_bytes", 0)
                        for r in records
                        if r["objective"] == pair[False]["objective"]
                        and not r["optimized"]
                    ),
                    default=0,
                ),
                optimized_peak_npu_bytes=max(
                    (
                        r["memory"].get("device_peak_allocated_bytes", 0)
                        for r in records
                        if r["objective"] == pair[False]["objective"] and r["optimized"]
                    ),
                    default=0,
                ),
                baseline_reference_s=statistics.median(
                    r["reference_seconds"]
                    for r in records
                    if r["objective"] == pair[False]["objective"] and not r["optimized"]
                ),
                optimized_reference_s=statistics.median(
                    r["reference_seconds"]
                    for r in records
                    if r["objective"] == pair[False]["objective"] and r["optimized"]
                ),
                baseline_end_to_end_s=statistics.median(
                    r["end_to_end_seconds"]
                    for r in records
                    if r["objective"] == pair[False]["objective"] and not r["optimized"]
                ),
                optimized_end_to_end_s=statistics.median(
                    r["end_to_end_seconds"]
                    for r in records
                    if r["objective"] == pair[False]["objective"] and r["optimized"]
                ),
            )
        )
        del o
    csv_rows(out / "p1_search_ab.csv", rows)
    write_json(
        out / "parity.json",
        dict(
            cache_sha256=file_hash(args.cache),
            mapping=name,
            provenance=provenance(),
            implementation=file_hash(Path(__file__).with_name("objective.py")),
            batch=args.batch,
            rows=args.rows,
            cache_mib=args.cache_mib,
            device=args.device,
            device_name=torch.npu.get_device_name(args.device)
            if args.device.startswith("npu")
            else "CPU (not A5)",
            reuse_sha256=file_hash(Path(__file__).parents[1] / "benchmark_reuse.py"),
            passed=True,
        ),
    )
    (out / "p1_search_ab.md").write_text(
        "# Full MXFP8 search A/B\n\n"
        + "\n".join(
            f"- {r['objective']}: {r['baseline_search_s']:.4f}s → {r['optimized_search_s']:.4f}s; {r['speedup']:.3f}x"
            for r in rows
        )
        + f"\n\nW+A/W-only: {rows[1]['baseline_search_s']/rows[0]['baseline_search_s']:.3f} → {rows[1]['optimized_search_s']/rows[0]['optimized_search_s']:.3f}.\n",
        encoding="utf-8",
    )


def reuse_qa(name, g, cache, old):
    if old["hashes"] != cache_hashes(cache):
        raise ValueError("Old search/cache fingerprints differ")
    if (
        old["reference_policy"]
        != "storage-dtype GEMM, CPU FP64 error accumulation, eps=1e-30"
    ):
        raise ValueError("Old reference/loss policy differs")
    before, now = old["provenance"], provenance()
    if any(before[k] != now[k] for k in ("torch", "compressed_tensors")):
        raise ValueError("Old runtime differs: QA reuse requires explicit audit")
    a, b = source_signature(before), source_signature(now)
    # core.py changed only by extracting applied_scale; audit this known old SHA.
    allowed_core = {
        "bafc6f20438223ad76411b0e6b2580b5ffec730b1c9f444476d3942cc045b0dd",
        b["core.py"],
    }
    if a.get("core.py") not in allowed_core or any(
        a.get(k) != v for k, v in b.items() if k != "core.py"
    ):
        raise ValueError("Old quantization/search implementation not audited")
    original = cache["groups"][name]
    if (
        any(
            not torch.equal(g[k], original[k]) or g[k].dtype != original[k].dtype
            for k in ("weight", "calibration", "validation")
        )
        or g["consumers"] != original["consumers"]
    ):
        raise ValueError("QA consumer/dtype/reference changed; cannot reuse")
    runs = [r for r in old["groups"][name]["runs"] if r["proxy"] == "MXFP8"]
    if len(runs) != 2 or {r["objective"] for r in runs} != {"W-only", "W+A"}:
        raise ValueError("Missing QA search arms")
    return runs


def complete(groups, cache, args, out):
    old = (
        torch.load(args.old_search, map_location="cpu", weights_only=True)
        if args.old_search
        else None
    )
    rows, results = [], {}
    for name, g in groups.items():
        reference_seconds = None
        if name.endswith("q_a_layernorm"):
            if old is None:
                raise ValueError("QA requires old search, not an implicit re-search")
            runs = reuse_qa(name, g, cache, old)
        else:
            sync(args.device)
            setup_start = time.perf_counter()
            o = make_objective(g, args)
            sync(args.device)
            reference_seconds = time.perf_counter() - setup_start
            runs = [o.search("MXFP8", a) for a in (False, True)]
            del o
        o = make_objective(g, args, split="validation")
        metrics = {"No Smooth": o.assess(torch.ones_like(o.amax), "MXFP8", True)}
        for r in runs:
            e0 = o.assess(r["scale"])
            if not e0["finite"] or e0["nmse"] > 1e-3:
                raise ValueError("Local no-QDQ equivalence failed")
            metrics[r["objective"]] = o.assess(r["scale"], "MXFP8", True)
        if not all(m["finite"] for m in metrics.values()):
            raise ValueError("Nonfinite validation metric")
        n, w, a = [metrics[k]["nmse"] for k in ("No Smooth", "W-only", "W+A")]
        rows.append(
            dict(
                layer=int(name.split(".")[2]),
                mapping=name,
                no_smooth_nmse=n,
                w_only_nmse=w,
                w_a_nmse=a,
                gain_wa_vs_w=100 * (w - a) / w if w else None,
                gain_wa_vs_no_smooth=100 * (n - a) / n if n else None,
                no_smooth_mse=metrics["No Smooth"]["mse"],
                w_only_mse=metrics["W-only"]["mse"],
                w_a_mse=metrics["W+A"]["mse"],
            )
        )
        results[name] = dict(
            geometry=g.get("geometry"),
            search_origin="historical QA reuse"
            if name.endswith("q_a_layernorm")
            else "optimized recipe search",
            reference_seconds=reference_seconds,
            candidate_cache_bytes=g["calibration"].numel()
            * g["calibration"].element_size(),
            runs=runs,
            metrics=metrics,
            consumers=g["consumers"],
            compensation_consumers=g["compensation_consumers"],
            dtypes=g["native_dtypes"],
        )
        torch.save(results, out / "results.pt")
        del o
    csv_rows(out / "mxfp8_e2_completed.csv", rows)
    write_json(
        out / "results.json",
        dict(
            provenance=provenance(),
            groups=results,
            token_ids=identities(cache),
            seed=cache["seed"],
            cache_sha256=file_hash(args.cache),
        ),
    )
    (out / "mxfp8_e2_completed.md").write_text(
        "# Recipe MXFP8 local NMSE\n\n| Layer | Mapping | No Smooth | W-only | W+A | WA vs W % | WA vs No Smooth % |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(
            "| "
            + " | ".join(
                str(r[k])
                for k in (
                    "layer",
                    "mapping",
                    "no_smooth_nmse",
                    "w_only_nmse",
                    "w_a_nmse",
                    "gain_wa_vs_w",
                    "gain_wa_vs_no_smooth",
                )
            )
            + " |"
            for r in rows
        ),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=["ab", "complete"])
    for key in ("cache", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--old-search", type=Path)
    p.add_argument("--ab", type=Path)
    p.add_argument("--mapping")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--cache-mib", type=float, default=64)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.batch, args.rows, args.cache_mib, args.repeats) <= 0:
        p.error("Positive sizes/repeats required")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("npu"):
        import torch_npu  # noqa:F401

        torch.npu.set_device(args.device)
    cache = load_cache(args.cache)
    groups = audit(cache, args.checkpoint)
    if {int(n.split(".")[2]) for n in groups} != {6, 16, 60, 77}:
        raise ValueError("Formal recipe experiment requires L6/L16/L60/L77 cache")
    suffixes = (
        ["self_attn.kv_b_proj"]
        if all("geometry" in g for g in groups.values())
        else ["input_layernorm", "self_attn.q_a_layernorm"]
    )
    if set(groups) != {
        f"model.layers.{layer}.{suffix}"
        for layer in (6, 16, 60, 77)
        for suffix in suffixes
    }:
        raise ValueError("Incomplete or unexpected recipe mapping coverage")
    if args.stage == "ab":
        if args.mapping not in groups:
            p.error("--mapping must select one actual mapping")
        ab(groups[args.mapping], args.mapping, args, args.output)
    else:
        if args.ab is None:
            p.error("--ab is the passing full-search parity.json")
        gate = json.loads(args.ab.read_text())
        suffix = (
            "self_attn.kv_b_proj"
            if all("geometry" in g for g in groups.values())
            else "input_layernorm"
        )
        if gate["mapping"] != "model.layers.6." + suffix:
            raise ValueError(
                "Expected passing L6 input/OV A/B for the corresponding capture"
            )
        if (
            not gate["passed"]
            or gate["cache_sha256"] != file_hash(args.cache)
            or gate["provenance"] != provenance()
            or gate["implementation"]
            != file_hash(Path(__file__).with_name("objective.py"))
            or gate.get("reuse_sha256")
            != file_hash(Path(__file__).parents[1] / "benchmark_reuse.py")
            or any(
                gate[k] != getattr(args, k)
                for k in ("batch", "rows", "cache_mib", "device")
            )
        ):
            raise ValueError("A/B does not match cache/code/runtime/device/chunking")
        complete(groups, cache, args, args.output)


if __name__ == "__main__":
    main()
