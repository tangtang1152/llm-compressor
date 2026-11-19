"""E2 cache smoke gates and conservative cross-layer analysis."""

import json
import math
from pathlib import Path

import torch

from ..core import Objective, smooth_scale
from ..run import write_json
from .common import (
    check_search,
    compatibility,
    csv_rows,
    identities,
    load_cache,
    runtime_signature,
)

NEW_LAYERS = {6, 16, 60, 77}
ALL_LAYERS = {3, 6, 16, 40, 60, 77}


def smoke(cache, device="cpu", batch=128, rows=256, tokens=128):
    identities(cache)
    results = []
    for name, g in cache["groups"].items():
        o = Objective(g["calibration"][:tokens], g["weight"], device, batch, rows)
        for a, b in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
            scale = smooth_scale(o.amax, o.wmax, a, b)
            e0 = o.assess(scale)
            passed = (
                e0["finite"]
                and e0["nmse"] <= 64 * torch.finfo(g["weight"].dtype).eps ** 2
            )
            for proxy in ("INT8", "MXFP4", "MXFP8"):
                metric = o.assess(scale, proxy, True)
                results.append(
                    dict(
                        mapping=name,
                        alpha=a,
                        beta=b,
                        proxy=proxy,
                        e0_nmse=e0["nmse"],
                        finite=metric["finite"],
                        e0_pass=passed,
                        nmse=metric["nmse"],
                    )
                )
    if not all(r["e0_pass"] and r["finite"] for r in results):
        raise ValueError("E2 smoke E0/finite gate failed; do not run formal search")
    return results


def load_batch(root):
    root = Path(root)
    cache = load_cache(root / "capture/cache.pt")
    search = torch.load(
        root / "search/search.pt", map_location="cpu", weights_only=True
    )
    check_search(cache, search)
    ev = json.loads((root / "evaluation/evaluation.json").read_text(encoding="utf-8"))
    if runtime_signature(ev.get("provenance", {})) != runtime_signature(
        search["provenance"]
    ):
        raise ValueError("Evaluation/search provenance differs within one batch")
    if set(ev["groups"]) != set(cache["groups"]):
        raise ValueError("Evaluation/cache mapping coverage differs")
    for name, g in ev["groups"].items():
        entries = g["results"]
        expected = {
            (r["proxy"], r["objective"]): (r["alpha"], r["beta"])
            for r in search["groups"][name]["runs"]
        }
        if (
            len(entries) != 7
            or len({(e["proxy"], e["objective"]) for e in entries}) != 7
        ):
            raise ValueError("Evaluation must have six arms plus baseline")
        for e in entries:
            if e["proxy"] != "No Smooth" and expected[(e["proxy"], e["objective"])] != (
                e["alpha"],
                e["beta"],
            ):
                raise ValueError("Search/evaluation parameter mismatch")
            if not e["e0"]["pass"]:
                raise ValueError("Prior E0 failed; batch not eligible")
            for fmt in ("W4A4", "W8A8"):
                if not math.isfinite(e[fmt]["nmse"]) or e[fmt]["nmse"] < 0:
                    raise ValueError("Invalid target NMSE")
    return cache, search, ev


def tables(search, ev, batch_id):
    raw, gains = [], []
    for name, g in ev["groups"].items():
        layer = int(name.split(".")[2])
        mapping = name.split(".")[-1]
        entries = {(e["proxy"], e["objective"]): e for e in g["results"]}
        scales = {
            (r["proxy"], r["objective"]): r["scale"].double()
            for r in search["groups"][name]["runs"]
        }
        for fmt, proxy in (("W4A4", "MXFP4"), ("W8A8", "MXFP8")):
            base = entries[("No Smooth", "baseline")][fmt]["nmse"]
            for key, e in entries.items():
                nmse = e[fmt]["nmse"]
                scale = scales.get(key)
                raw.append(
                    dict(
                        batch=batch_id,
                        layer=layer,
                        mapping=mapping,
                        format=fmt,
                        proxy=key[0],
                        objective=key[1],
                        alpha=e["alpha"],
                        beta=e["beta"],
                        search_loss=e.get("loss"),
                        nmse=nmse,
                        no_smooth_nmse=base,
                        absolute_gain_vs_no_smooth=base - nmse,
                        relative_gain_vs_no_smooth=(base - nmse) / base
                        if base
                        else None,
                        search_seconds=e["search_seconds"],
                        scale_min=scale.min().item() if scale is not None else 1.0,
                        scale_max=scale.max().item() if scale is not None else 1.0,
                        log2_scale_mean=scale.log2().mean().item()
                        if scale is not None
                        else 0.0,
                    )
                )
            for kind, left, right in (
                ("matched_proxy", ("INT8", "W+A"), (proxy, "W+A")),
                ("activation_aware", (proxy, "W-only"), (proxy, "W+A")),
            ):
                a, b = (entries[k][fmt]["nmse"] for k in (left, right))
                gains.append(
                    dict(
                        batch=batch_id,
                        layer=layer,
                        mapping=mapping,
                        format=fmt,
                        comparison=kind,
                        reference_nmse=a,
                        candidate_nmse=b,
                        absolute_gain=a - b,
                        relative_gain=(a - b) / a if a else None,
                    )
                )
    return raw, gains


def plots(out, raw, gains):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in ("matched_proxy", "activation_aware", "no_smooth"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
        for ax, fmt in zip(axes[0], ("W4A4", "W8A8")):
            if kind == "no_smooth":
                values = [
                    r for r in raw if r["format"] == fmt and r["proxy"] == "No Smooth"
                ]
                key = "nmse"
            else:
                values = [
                    r for r in gains if r["format"] == fmt and r["comparison"] == kind
                ]
                key = "relative_gain"
            for batch in sorted({r["batch"] for r in values}):
                for mapping in ("input_layernorm", "q_a_layernorm"):
                    series = sorted(
                        [
                            r
                            for r in values
                            if r["batch"] == batch
                            and r["mapping"] == mapping
                            and r[key] is not None
                        ],
                        key=lambda r: r["layer"],
                    )
                    if series:
                        ax.plot(
                            [r["layer"] for r in series],
                            [r[key] for r in series],
                            marker="o",
                            label=f"{batch}: {mapping}",
                        )
            ax.set(
                title=fmt,
                xlabel="Layer index",
                ylabel="NMSE" if kind == "no_smooth" else "Relative NMSE reduction",
            )
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        fig.suptitle(kind + " (exploratory; lines connect only compatible batches)")
        fig.tight_layout()
        fig.savefig(out / (kind + ".png"), dpi=150)
        plt.close(fig)


def analyze(args, out, manifest):
    old, os, oe = load_batch(args.old)
    new, ns, ne = load_batch(args.new)
    old_layers = {int(n.split(".")[2]) for n in old["groups"]}
    new_layers = {int(n.split(".")[2]) for n in new["groups"]}
    if old_layers != {3, 40} or new_layers != NEW_LAYERS:
        raise ValueError("Expected old L3/L40 and new L6/L16/L60/L77 only")
    for c in (old, new):
        layers = {int(n.split(".")[2]) for n in c["groups"]}
        expected = {
            f"model.layers.{i}.{suffix}"
            for i in layers
            for suffix in ("input_layernorm", "self_attn.q_a_layernorm")
        }
        if set(c["groups"]) != expected:
            raise ValueError("Expected exact input_layernorm and q_a_layernorm groups")
        for name, g in c["groups"].items():
            if len(g["calibration"]) != 2048 or len(g["validation"]) != 2048:
                raise ValueError("Formal analysis requires 2048 tokens per split")
        if len(c["groups"]) != 2 * len({n.split(".")[2] for n in c["groups"]}):
            raise ValueError("Both norm mappings required for each layer")
    matched = compatibility(old, new, os, ns)
    r1, g1 = tables(os, oe, "paired" if matched["strictly_comparable"] else "original")
    r2, g2 = tables(ns, ne, "paired" if matched["strictly_comparable"] else "extension")
    raw, gains = r1 + r2, g1 + g2
    manifest["status"] = "complete"
    csv_rows(out / "raw_layers.csv", raw)
    csv_rows(out / "gains.csv", gains)
    write_json(
        out / "results.json",
        dict(metadata=manifest, compatibility=matched, raw=raw, gains=gains),
    )
    plots(out, raw, gains)
    counts = {
        kind: dict(
            improved=sum(
                r["absolute_gain"] > 0 for r in gains if r["comparison"] == kind
            ),
            tied=sum(r["absolute_gain"] == 0 for r in gains if r["comparison"] == kind),
            worse=sum(r["absolute_gain"] < 0 for r in gains if r["comparison"] == kind),
        )
        for kind in ("matched_proxy", "activation_aware")
    }
    negative = [
        r
        for r in raw
        if r["relative_gain_vs_no_smooth"] is not None
        and r["relative_gain_vs_no_smooth"] <= 0
        and r["proxy"] != "No Smooth"
    ]
    text = [
        "# E2 layer extension",
        "",
        str(matched),
        "",
        "Counts are descriptive, not causal depth evidence: " + str(counts),
        f"{len(negative)} arm/format entries have zero or negative gain versus No Smooth; inspect raw_layers.csv.",  # noqa: E501
        "Both absolute and relative gains are in gains.csv. Smaller percentage gains alone do not establish deep-layer failure. "  # noqa: E501
        "Six selected layers cannot establish a monotone trend or justify Selective FlexSmooth without E3 and replication.",  # noqa: E501
        "Incompatible batches are plotted separately and are not a strictly paired layer-depth series.",  # noqa: E501
        "![Proxy](matched_proxy.png)",
        "![Activation](activation_aware.png)",
        "![Baseline](no_smooth.png)",
    ]
    (out / "report.md").write_text("\n\n".join(text), encoding="utf-8")
    manifest["compatibility"] = matched
