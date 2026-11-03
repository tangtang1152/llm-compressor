"""Norm-fusion attribution; separate outputs, never overwrites E1 sources."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .core import metrics, percentile, qdq, transform
from .run import Checkpoint, provenance


def summary(x):
    x = x.double().flatten()
    return dict(
        mean=x.mean().item(),
        **{f"p{p}": percentile(x, p / 100) for p in (50, 90, 95, 99)},
    )


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    return dict(
        n=len(x),
        pearson=float(np.corrcoef(x, y)[0, 1])
        if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0
        else None,
    )


def groups(w, quantized=None, scale=None):
    x = w.double().reshape(-1, 32)
    energy = x.square().sum(-1)
    rms = (energy / 32).sqrt()
    maximum = x.abs().amax(-1)
    others = ((energy - maximum.square()).clamp_min(0) / 31).sqrt()
    result = dict(
        rms=rms,
        absmax=maximum,
        max_rms=maximum / (rms + 1e-30),
        max_other_rms=maximum / (others + 1e-30),
        abs_cv=x.abs().std(-1, correction=0) / (x.abs().mean(-1) + 1e-30),
        energy=energy,
    )
    if quantized is not None:
        q = quantized.double().reshape_as(x)
        sse = (q - x).square().sum(-1)
        result.update(
            sse=sse,
            relative_mse=sse / (energy + 1e-30),
            zero_fraction=(q == 0).double().mean(-1),
        )
        # CT get_qparams exposes decoded power-of-two scales, not encoded bytes.
        decoded = scale.double().flatten()
        if decoded.numel() != rms.numel() or not (decoded > 0).all():
            raise ValueError("unexpected CT scale layout/value")
        exponent = decoded.log2()
        if not torch.equal(exponent, exponent.round()):
            raise ValueError("CT MX scale is not power-of-two")
        result["scale_log2"] = exponent
    return result


def distribution(w, g):
    x = w.double()
    rms = x.square().mean().sqrt()
    result = dict(matrix_absmax_rms=(x.abs().max() / (rms + 1e-30)).item())
    for key, value in g.items():
        result.update({f"{key}_{k}": v for k, v in summary(value).items()})
    if "sse" in g:
        result["weight_nmse"] = (g["sse"].sum() / (g["energy"].sum() + 1e-30)).item()
        vals, counts = torch.unique(g["scale_log2"], return_counts=True)
        result["scale_histogram"] = json.dumps(
            dict(zip(vals.tolist(), counts.tolist()))
        )
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reuse(paths, cp, prov):
    rows, selection, sources = {}, {}, []
    for path in paths:
        data = json.loads(path.read_text())
        if (
            data["status"] != "complete"
            or data["model_config"] != cp.config
            or data["quantization"].get("group_size") != 32
            or data["quantization"].get("axis") != -1
            or data["index_sha256"]
            != hashlib.sha256(cp.index_path.read_bytes()).hexdigest()
            or data["rotation"]
            != dict(block_size=32, seed=1234, precision="float32", shifted=False)
        ):
            raise ValueError(f"incompatible prior E1: {path}")
        previous = data["provenance"]

        def fingerprints(p):
            return {
                k.replace("\\", "/").split("/")[-1]: v
                for k, v in p["implementation_sha256"].items()
            }

        if previous["versions"] != prov["versions"] or fingerprints(
            previous
        ) != fingerprints(prov):
            raise ValueError(f"quantizer/environment fingerprint differs: {path}")
        for row in data["rows"]:
            key = (row["layer"], row["expert"], row["projection"])
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting E1 row: {key}")
            rows[key] = row
            selection.setdefault(row["layer"], set()).add(row["expert"])
        sources.append(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    return rows, selection, sources


def plots(out, matrices, gamma, comparisons, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(label)
    layers = sorted({r["layer"] for r in matrices})
    for projection in ("gate_proj", "up_proj"):
        rows = [r for r in comparisons if r["projection"] == projection]
        axes[0, 0].plot(
            layers,
            [
                np.mean([r["gain_percent"] for r in rows if r["layer"] == layer_id])
                for layer_id in layers
            ],
            label=projection,
            marker="o",
        )
    axes[0, 0].set_ylabel("Mean per-matrix NMSE gain (%)")
    axes[0, 1].plot(
        [r["layer"] for r in gamma], [r["absmax_rms"] for r in gamma], marker="o"
    )
    axes[0, 1].set_ylabel("Gamma AbsMax/RMS")
    for stage in ("Raw", "Fused", "Rotated"):
        rows = [r for r in matrices if r["stage"] == stage]
        axes[1, 0].plot(
            layers,
            [
                np.mean([r["max_rms_p95"] for r in rows if r["layer"] == layer_id])
                for layer_id in layers
            ],
            label=stage,
            marker="o",
        )
        if stage != "Raw":
            axes[1, 1].plot(
                layers,
                [
                    np.mean(
                        [
                            r["zero_fraction_mean"]
                            for r in rows
                            if r["layer"] == layer_id
                        ]
                    )
                    for layer_id in layers
                ],
                label=stage,
                marker="o",
            )
    axes[1, 0].set_ylabel("Mean matrix Group Max/RMS P95")
    axes[1, 1].set_ylabel("Mean QDQ zero fraction")
    for ax in axes.flat:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.savefig(out / "layer_trends.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(label)
    for stage, offset in [("Fused", -0.15), ("Rotated", 0.15)]:
        for i, layer in enumerate(layers):
            rows = [r for r in matrices if r["stage"] == stage and r["layer"] == layer]
            # Each vertical series is one matrix's exact group quantiles, not pooled.
            for r in rows:
                ax.plot(
                    [i + offset] * 4,
                    [r[f"relative_mse_p{p}"] for p in (50, 90, 95, 99)],
                    color="C0" if stage == "Fused" else "C1",
                    alpha=0.3,
                    marker=".",
                )
        ax.plot([], [], label=stage, color="C0" if stage == "Fused" else "C1")
    ax.set_xticks(range(len(layers)), layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Per-matrix Group relative MSE P50/P90/P95/P99")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out / "group_error_distribution.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 40, 77],
    )
    p.add_argument("--experts-per-layer", type=int, default=16)
    p.add_argument("--reuse-e1", type=Path, nargs="*", default=[])
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True, exist_ok=False)
    prov = provenance()
    # Metadata-only construction: select prior IDs first, validate
    # actual selections below.
    meta = Checkpoint(a.checkpoint, [], 0, 1234)
    old, selections, sources = reuse(a.reuse_e1, meta, prov)
    result = dict(
        status="running",
        provenance=prov,
        sources=sources,
        selection={},
        checkpoint=str(a.checkpoint),
        model_config=meta.config,
        data_kind="SYNTHETIC"
        if meta.config.get("_experiment_synthetic")
        else "checkpoint diagnostics",
        gamma=[],
        matrices=[],
        comparisons=[],
        group_correlations=[],
        e1_rows=[],
    )
    try:
        for layer in a.layers:
            ids = sorted(selections[layer]) if layer in selections else None
            cp = Checkpoint(a.checkpoint, [layer], a.experts_per_layer, 1234, ids)
            ids = cp.selection[layer]
            result["selection"][layer] = ids
            gain = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
            g = gain.double()
            gr = groups(g.reshape(1, -1))
            gamma = dict(
                layer=layer,
                min=g.min().item(),
                max=g.max().item(),
                mean=g.mean().item(),
                rms=g.square().mean().sqrt().item(),
                absmax_rms=(g.abs().max() / (g.square().mean().sqrt() + 1e-30)).item(),
                **{f"abs_p{v}": percentile(g.abs(), v / 100) for v in (50, 90, 99)},
            )
            gamma.update(
                {f"group_abs_cv_{k}": v for k, v in summary(gr["abs_cv"]).items()}
            )
            result["gamma"].append(gamma)
            for expert in ids:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = (layer, expert, projection)
                    base = dict(layer=layer, expert=expert, projection=projection)
                    formal = old.get(key)
                    if projection == "down_proj" and formal is not None:
                        result["e1_rows"].append(formal)
                        continue
                    weight = cp.load(cp.key(*key))
                    calculated = dict(
                        **base, tensor_key=cp.key(*key), shard=cp.index[cp.key(*key)]
                    )
                    stages = {}
                    for stage in ("Raw", "Fused", "Rotated"):
                        if stage == "Raw" and projection == "down_proj":
                            continue
                        w = (
                            weight
                            if stage == "Raw"
                            else transform(weight, gain, projection, stage == "Rotated")
                        )
                        q, params = (
                            (None, None)
                            if stage == "Raw"
                            else qdq(w, return_params=True)
                        )
                        if stage != "Raw" and formal is None:
                            label = "B" if stage == "Fused" else "C"
                            calculated.update(
                                {f"{label}_{k}": v for k, v in metrics(w, q).items()}
                            )
                        if projection != "down_proj":
                            gs = groups(
                                w, q, None if params is None else params["scale"]
                            )
                            stages[stage] = gs
                            result["matrices"].append(
                                dict(**base, stage=stage, **distribution(w, gs))
                            )
                        del w, q, params
                    if formal is None:
                        b, c = calculated["B_weight_nmse"], calculated["C_weight_nmse"]
                        calculated["gain_percent"] = 100 * (b - c) / b if b else None
                        formal = calculated
                    result["e1_rows"].append(formal)
                    if projection == "down_proj":
                        continue
                    raw, b, c = (stages[s] for s in ("Raw", "Fused", "Rotated"))
                    result["group_correlations"].append(
                        dict(
                            **base,
                            feature="gamma_abs_cv_vs_fusion_max_rms_delta",
                            **corr(
                                gr["abs_cv"].repeat(weight.shape[0]).numpy(),
                                (b["max_rms"] - raw["max_rms"]).numpy(),
                            ),
                        )
                    )
                    if key in old:
                        for label, gs in (("B", b), ("C", c)):
                            measured = (
                                gs["sse"].sum() / (gs["energy"].sum() + 1e-30)
                            ).item()
                            if not np.isclose(
                                measured,
                                formal[f"{label}_weight_nmse"],
                                rtol=1e-6,
                                atol=1e-12,
                            ):
                                raise ValueError(
                                    f"prior E1 metric mismatch: {key} {label}"
                                )
                    comparison = dict(
                        **base,
                        gain_percent=formal["gain_percent"],
                        e1_source="reused" if key in old else "new",
                    )
                    for name in ("max_rms", "zero_fraction", "relative_mse"):
                        comparison[f"{name}_delta_mean"] = (
                            (c[name] - b[name]).mean().item()
                        )
                        comparison[f"{name}_delta_p95"] = percentile(
                            c[name], 0.95
                        ) - percentile(b[name], 0.95)
                    comparison["fusion_max_rms_delta_mean"] = (
                        (b["max_rms"] - raw["max_rms"]).mean().item()
                    )
                    result["comparisons"].append(comparison)
                    for name in ("max_rms", "zero_fraction"):
                        result["group_correlations"].append(
                            dict(
                                **base,
                                feature=name,
                                **corr(
                                    (c[name] - b[name]).numpy(),
                                    (c["relative_mse"] - b["relative_mse"]).numpy(),
                                ),
                            )
                        )
                    # Stream all paired groups; bounded to one matrix, gzip limits disk
                    # use.
                    columns = {
                        f"{s}_{k}": v.tolist()
                        for s, gs in stages.items()
                        for k, v in gs.items()
                    }
                    path = a.output / f"groups-L{layer}-E{expert}-{projection}.csv.gz"
                    with gzip.open(path, "wt", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            ["row", "input_group", "gamma_abs_cv", *columns]
                        )
                        ncol = weight.shape[1] // 32
                        gamma_cv = gr["abs_cv"].tolist()
                        for i, values in enumerate(zip(*columns.values())):
                            writer.writerow(
                                [i // ncol, i % ncol, gamma_cv[i % ncol], *values]
                            )
                    del weight, stages
                print(f"layer={layer} expert={expert} complete", flush=True)
        result["status"] = "complete"
        result["matrix_correlations"] = []
        for projection in ("gate_proj", "up_proj"):
            for layer in [None, *a.layers]:
                rows = [
                    r
                    for r in result["comparisons"]
                    if r["projection"] == projection
                    and (layer is None or r["layer"] == layer)
                ]
                for feature in (
                    "fusion_max_rms_delta_mean",
                    "max_rms_delta_mean",
                    "zero_fraction_delta_mean",
                    "relative_mse_delta_p95",
                ):
                    pairs = [r for r in rows if r["gain_percent"] is not None]
                    result["matrix_correlations"].append(
                        dict(
                            projection=projection,
                            layer=layer,
                            feature=feature,
                            **corr(
                                [r[feature] for r in pairs],
                                [r["gain_percent"] for r in pairs],
                            ),
                        )
                    )
        plots(
            a.output,
            result["matrices"],
            result["gamma"],
            result["comparisons"],
            result["data_kind"],
        )
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        (a.output / "diagnostics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        for name in (
            "gamma",
            "matrices",
            "comparisons",
            "group_correlations",
            "matrix_correlations",
            "e1_rows",
        ):
            write_csv(a.output / f"{name}.csv", result.get(name, []))
        lines = [
            "# Norm fusion / MXFP4 diagnostics",
            "",
            f"Status: {result['status']}; data: {result['data_kind']}",
            "",
            "Raw is distribution-only. Formal B/C rows retain E1 columns.",
            "All correlations are descriptive; groups share gamma and matrices "
            "within layers are dependent.",
            "Pooled correlations can reflect layer differences. Within-layer "
            "results are also provided.",
            "No causal attribution or full Expert W4A4 conclusion follows from "
            "these measurements.",
            "",
            "|Layer|Projection|N matrices|Mean gain %|Mean Fusion Group "
            "Max/RMS change|",
            "|---|---|---|---|---|",
        ]
        for layer in a.layers:
            for proj in ("gate_proj", "up_proj"):
                rows = [
                    r
                    for r in result["comparisons"]
                    if r["layer"] == layer
                    and r["projection"] == proj
                    and r["gain_percent"] is not None
                ]
                if rows:
                    fusion_mean = np.mean(
                        [r["fusion_max_rms_delta_mean"] for r in rows]
                    )
                    lines.append(
                        f"|{layer}|{proj}|{len(rows)}|"
                        f"{np.mean([r['gain_percent'] for r in rows]):.4g}|"
                        f"{fusion_mean:.4g}|"
                    )
        lines += [
            "",
            "![Layer trends](layer_trends.png)",
            "![Group quantiles](group_error_distribution.png)",
            "",
            "Hypotheses: gamma heterogeneity may amplify within-group "
            "imbalance; B32 may reduce it.",
            "Check paired groups, scale histograms and zero rates before "
            "accepting this explanation.",
            "Shared gamma is compatible with synchronous gate/up gains but "
            "does not prove causation.",
        ]
        (a.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
