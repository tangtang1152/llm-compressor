"""E1.3 raw-weight / group-flat-gamma CPU quantization difficulty experiment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from llmcompressor.modifiers.transform.quarot.rotation import rotate_axis

from .core import qdq, rotation, transform
from .diagnose import atomic_json, distribution, groups, reuse, write_csv
from .run import Checkpoint, provenance

STAGES = ("A", "B", "C", "D", "E")
FIELDS = (
    "weight_nmse",
    "matrix_absmax_rms",
    "max_rms_mean",
    "max_rms_p50",
    "max_rms_p95",
    "max_rms_p99",
    "zero_fraction_mean",
    "relative_mse_p95",
)


class HistorySource:
    """Normalize E1.2's row key for the existing strict E1 validator.

    Hashes always refer to the original source bytes, not the normalized view.
    Older E1.2 reports missing provenance fields must use their original E1 files.
    """

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return str(self.path)

    def read_bytes(self):
        return self.path.read_bytes()

    def read_text(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if "rows" not in data and "e1_rows" in data:
            data["rows"] = data["e1_rows"]
        required = (
            "status",
            "rows",
            "selection",
            "model_config",
            "index_sha256",
            "rotation",
            "quantization",
            "provenance",
        )
        if any(key not in data for key in required):
            raise ValueError(
                f"incomplete historical schema: {self.path}; "
                "use original E1 results.json"
            )
        return json.dumps(data)


def flat_gamma(gamma):
    if gamma.ndim != 1 or not gamma.numel() or gamma.numel() % 32:
        raise ValueError("gamma must be a nonempty vector divisible by 32")
    if not torch.isfinite(gamma).all():
        raise ValueError("nonfinite gamma")
    g = gamma.float().reshape(-1, 32)
    rms = g.square().mean(-1, keepdim=True).sqrt()
    flattened = (torch.where(g < 0, -1.0, 1.0) * rms).reshape_as(gamma)
    if not torch.isfinite(flattened).all():
        raise ValueError("FP32 group RMS overflow")
    actual = flattened.reshape(-1, 32).square().mean(-1).sqrt()
    return flattened, dict(
        negative_count=int((g < 0).sum()),
        zero_count=int((g == 0).sum()),
        all_zero_groups=int((rms == 0).sum()),
        rms_preservation_max_abs=(actual - rms.flatten()).abs().max().item(),
        zero_policy=(
            "zero (including negative zero) uses positive RMS; "
            "all-zero group stays zero"
        ),
        dtype="float32; no BF16 gamma rounding before fusion",
    )


def variants(weight, gamma, flat):
    yield "A", weight
    rot = rotation(gamma.numel())
    for unrotated, rotated, gain in [("B", "C", gamma), ("D", "E", flat)]:
        fused = transform(weight, gain, "gate_proj")  # same input-axis fusion for up
        yield unrotated, fused
        yield rotated, rotate_axis(fused, rot, axis=1, precision=torch.float32)


def measure(weight):
    q, params = qdq(weight, return_params=True)
    gs = groups(weight, q, params["scale"])
    stats = distribution(weight, gs)
    return {k: stats[k] for k in FIELDS}


def gain(reference, candidate):
    return 100 * (reference - candidate) / reference if reference else None


def comparisons(values):
    a, b, c, d, e = (values[s] for s in STAGES)
    cb, ed = gain(b, c), gain(d, e)
    return dict(
        B_minus_A_nmse=b - a,
        B_vs_A_increase_percent=-gain(a, b) if a else None,
        C_vs_A_gain_percent=gain(a, c),
        B_minus_D_nmse=b - d,
        D_vs_B_gain_percent=gain(b, d),
        C_vs_B_gain_percent=cb,
        E_vs_D_gain_percent=ed,
        rotation_gain_drop_pp=cb - ed if cb is not None and ed is not None else None,
    )


def mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def layer_summaries(rows, paired):
    stages, deltas = [], []
    for layer in sorted({r["layer"] for r in rows}):
        for projection in ("gate_proj", "up_proj"):
            for stage in STAGES:
                selected = [
                    r
                    for r in rows
                    if (r["layer"], r["projection"], r["stage"])
                    == (layer, projection, stage)
                ]
                stages.append(
                    dict(
                        layer=layer,
                        projection=projection,
                        stage=stage,
                        n=len(selected),
                        **{k: mean([r[k] for r in selected]) for k in FIELDS},
                    )
                )
            selected = [
                r
                for r in paired
                if (r["layer"], r["projection"]) == (layer, projection)
            ]
            keys = [
                k for k in selected[0] if k not in ("layer", "expert", "projection")
            ]
            deltas.append(
                dict(
                    layer=layer,
                    projection=projection,
                    n=len(selected),
                    **{k: mean([r[k] for r in selected]) for k in keys},
                )
            )
    return stages, deltas


def report(result):
    lines = [
        "# E1.3 quantization difficulty",
        "",
        f"Data: {result['data_kind']}",
        "NMSE is relative to each stage's own weight representation. D/E modify gamma.",
        "These measurements do not establish Expert W4A4 accuracy or "
        "function equivalence.",
        "Layer values are arithmetic means of paired expert metrics, not "
        "pooled energy ratios.",
        "Positive gains mean lower NMSE; positive B-A means fusion increased NMSE.",
        "Positive rotation-gain drop means E/D gain is smaller than C/B, "
        "in percentage points.",
        "Three experts per layer support descriptive comparisons, not "
        "statistical significance.",
        "",
        "|Layer|Projection|B-A NMSE|B/A increase %|C/A gain %|D/B gain "
        "%|C/B gain %|E/D gain %|Gain drop pp|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    keys = (
        "B_minus_A_nmse",
        "B_vs_A_increase_percent",
        "C_vs_A_gain_percent",
        "D_vs_B_gain_percent",
        "C_vs_B_gain_percent",
        "E_vs_D_gain_percent",
        "rotation_gain_drop_pp",
    )
    for r in result["layer_comparisons"]:
        values = ["NA" if r[k] is None else f"{r[k]:.6g}" for k in keys]
        lines.append(f"|{r['layer']}|{r['projection']}|" + "|".join(values) + "|")
    lines += [
        "",
        "See matrices.csv for every expert/stage NMSE, Group Max/RMS, "
        "zeros and Block P95.",
        "Flat-gamma effects isolate a magnitude-heterogeneity intervention "
        "on these weights only;",
        "zero-channel replacement and BF16 rounding are recorded additional effects.",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reuse-e1", type=Path, nargs="+", required=True)
    p.add_argument(
        "--expert-selection",
        type=Path,
        help="JSON layer -> exactly 3 historical expert IDs, if history contains more",
    )
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 16, 40])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1 or len(set(a.layers)) != len(a.layers):
        p.error("positive threads and unique layers required")
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="running",
        data_kind="not validated",
        rows=[],
        paired=[],
        gamma=[],
        configuration={
            k: [str(x) for x in v] if k == "reuse_e1" else str(v)
            for k, v in vars(a).items()
        },
    )
    atomic_json(a.output / "results.json", result)
    start = time.perf_counter()
    try:
        prov = provenance()
        cp = Checkpoint(a.checkpoint, [], 0, 1234)
        old, selections, sources = reuse(
            [HistorySource(path) for path in a.reuse_e1], cp, prov
        )
        requested = (
            {int(k): v for k, v in json.loads(a.expert_selection.read_text()).items()}
            if a.expert_selection
            else selections
        )
        selected = {}
        for layer in a.layers:
            ids = requested.get(layer, [])
            if (
                len(ids) != 3
                or len(set(ids)) != 3
                or not set(ids) <= selections.get(layer, set())
            ):
                raise ValueError(
                    f"layer {layer}: require exactly 3 distinct historical experts; "
                    "supply --expert-selection if ambiguous"
                )
            selected[layer] = sorted(ids)
        cp.select(a.layers, 3, 1234, selections=selected)
        result.update(
            provenance=prov,
            sources=sources,
            selection=selected,
            model_config=cp.config,
            index_sha256=hashlib.sha256(cp.index_path.read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256(a.expert_selection.read_bytes()).hexdigest()
            if a.expert_selection
            else None,
            experiment_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            data_kind="SYNTHETIC"
            if cp.config.get("_experiment_synthetic")
            else "checkpoint weights",
            rotation=dict(
                block_size=32, seed=1234, precision="float32", writeback="bfloat16"
            ),
            quantization=dict(
                format="CT MXFP4", group_size=32, axis=-1, activation_quantization=False
            ),
        )
        for layer, ids in selected.items():
            gamma = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
            flat, info = flat_gamma(gamma)
            result["gamma"].append(dict(layer=layer, **info))
            for expert in ids:
                for projection in ("gate_proj", "up_proj"):
                    key = (layer, expert, projection)
                    weight = cp.load(cp.key(*key))
                    values = {}
                    for stage, w in variants(weight, gamma, flat):
                        stats = measure(w)
                        if stage in ("B", "C") and not np.isclose(
                            stats["weight_nmse"],
                            old[key][f"{stage}_weight_nmse"],
                            rtol=1e-6,
                            atol=1e-12,
                        ):
                            raise ValueError(f"historical NMSE mismatch: {key} {stage}")
                        result["rows"].append(
                            dict(
                                layer=layer,
                                expert=expert,
                                projection=projection,
                                stage=stage,
                                **stats,
                            )
                        )
                        values[stage] = stats["weight_nmse"]
                    result["paired"].append(
                        dict(
                            layer=layer,
                            expert=expert,
                            projection=projection,
                            **comparisons(values),
                        )
                    )
                    del weight, w
                print(f"layer={layer} expert={expert} complete", flush=True)
        result["layer_metrics"], result["layer_comparisons"] = layer_summaries(
            result["rows"], result["paired"]
        )
        for filename, key in [
            ("matrices", "rows"),
            ("comparisons", "paired"),
            ("gamma", "gamma"),
            ("layers", "layer_metrics"),
            ("layer_comparisons", "layer_comparisons"),
        ]:
            write_csv(a.output / f"{filename}.csv", result[key])
        (a.output / "summary.md").write_text(report(result), encoding="utf-8")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        (a.output / "summary.md").write_text(
            f"# E1.3 failed\n\n{exc}\n"
            "Partial results are not a completed experiment.\n",
            encoding="utf-8",
        )
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        atomic_json(a.output / "results.json", result)


if __name__ == "__main__":
    main()
