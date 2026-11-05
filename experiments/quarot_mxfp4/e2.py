"""E2 replay of verified original-BF16 routed activations. No model loader."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from .core import inverse_hidden, qdq, rotation, transform
from .diagnose import atomic_json, corr, summary, write_csv
from .e2_capture import SELECTION
from .run import Checkpoint, provenance

MODES = {
    "FP": (False, False),
    "W4A16": (True, False),
    "W16A4": (False, True),
    "W4A4": (True, True),
}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def error_stats(ref, actual):
    r, a = ref.double(), actual.double()
    if r.shape != a.shape or not torch.isfinite(a).all() or not torch.isfinite(r).all():
        raise ValueError("invalid comparison tensors")
    diff = a - r
    sse = diff.square().sum()
    energy = r.square().sum()
    per_token = diff.square().sum(-1) / (r.square().sum(-1) + 1e-30)
    denom = r.norm() * a.norm()
    return dict(
        nmse=(sse / (energy + 1e-30)).item(),
        sse=sse.item(),
        reference_energy=energy.item(),
        max_abs=diff.abs().max().item(),
        mean_abs=diff.abs().mean().item(),
        rmse=diff.square().mean().sqrt().item(),
        cosine=((r * a).sum() / denom).item() if denom else None,
        **{f"token_nmse_{k}": v for k, v in summary(per_token).items() if k != "mean"},
    )


@torch.no_grad()
def replay(x, weights, quantized, mode, dtype=torch.bfloat16):
    wq, aq = MODES[mode]
    records = []

    def linear(value, index):
        value = value.to(dtype)
        rounded = qdq(value, activation=True)
        records.append(
            dict(
                projection=PROJECTIONS[index], applied=aq, **error_stats(value, rounded)
            )
        )
        w = (quantized if wq else weights)[index].to(dtype)
        return F.linear(rounded if aq else value, w)

    gate, up = linear(x, 0), linear(x, 1)
    # Down receives this mode's actual gate/up computation, never ideal activations.
    output = linear(F.silu(gate) * up, 2)
    return output, records


@torch.no_grad()
def evaluate(z, z_gamma, gamma, weights, dtype=torch.bfloat16):
    rot = rotation(gamma.numel())
    branches = {
        "A": weights,
        "B": [transform(w, gamma, p) for w, p in zip(weights, PROJECTIONS)],
        "C": [transform(w, gamma, p, True) for w, p in zip(weights, PROJECTIONS)],
    }
    inputs = {"A": z_gamma, "B": z, "C": rot.apply(z.float()).to(z.dtype)}
    outputs, activations, weight_rows = [], [], []
    common = None
    for branch, ws in branches.items():
        quantized = [qdq(w) for w in ws]
        for p, w, q in zip(PROJECTIONS, ws, quantized):
            # Weight matrices are not tokens; only NMSE is required here.
            weight_rows.append(
                dict(
                    branch=branch,
                    projection=p,
                    nmse=(
                        (q.double() - w.double()).square().sum()
                        / (w.double().square().sum() + 1e-30)
                    ).item(),
                )
            )
        own = None
        for mode in MODES:
            y, acts = replay(inputs[branch], ws, quantized, mode, dtype)
            if branch == "C":
                y = inverse_hidden(y.float(), rot).to(dtype)
            if mode == "FP":
                own = y
            if branch == "A" and mode == "FP":
                common = y
            outputs.append(
                dict(
                    branch=branch,
                    mode=mode,
                    tokens=len(z),
                    **{f"common_{k}": v for k, v in error_stats(common, y).items()},
                    **{f"branch_fp_{k}": v for k, v in error_stats(own, y).items()},
                )
            )
            activations.extend(dict(branch=branch, mode=mode, **r) for r in acts)
    return outputs, activations, weight_rows


def run(checkpoint, capture, output, dtype=torch.bfloat16, *, synthetic=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", outputs=[], activations=[], weights=[])
    atomic_json(output / "results.json", result)
    try:
        capture = Path(capture)
        manifest = json.loads((capture / "capture_manifest.json").read_text())
        selection = {int(k): v for k, v in manifest["selection"].items()}
        if manifest["status"] != "complete" or (
            not synthetic
            and (manifest["source"]["kind"] != "real_bf16" or selection != SELECTION)
        ):
            raise ValueError("requires complete real capture with exact E1.3 selection")
        cp = Checkpoint(checkpoint, [], 0, 1234)
        cp.select(list(selection), 3, 1234, selections=selection)
        index_hash = hashlib.sha256(cp.index_path.read_bytes()).hexdigest()
        if not synthetic and (
            manifest["source"].get("checkpoint_index_sha256") != index_hash
            or manifest["source"].get("checkpoint_config_sha256")
            != hashlib.sha256(
                (Path(checkpoint) / "config.json").read_bytes()
            ).hexdigest()
            or manifest["source"].get("routing_alignment_verified") is not True
        ):
            raise ValueError("capture checkpoint/alignment mismatch")
        expected = {(layer_id, e) for layer_id, ids in selection.items() for e in ids}
        actual = [(e["layer"], e["expert"]) for e in manifest["experts"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("capture expert set mismatch")
        result.update(
            provenance=provenance(),
            capture_manifest_sha256=hashlib.sha256(
                (capture / "capture_manifest.json").read_bytes()
            ).hexdigest(),
            data_kind=manifest["source"]["kind"],
            arithmetic=str(dtype),
            selection=selection,
            activation_quantization="CT dynamic per-token per-32 input group",
            reference="Original A FP, unweighted expert output",
            configuration=dict(
                checkpoint=str(checkpoint),
                capture=str(capture.resolve()),
                rotation_block_size=32,
                seed=1234,
                transform_precision="float32",
                weight_writeback="bfloat16",
                quantization="CT MXFP4 E2M1/E8M0",
                group_size=32,
                axis=-1,
                weights="static minmax",
                activations="dynamic per token/group minmax",
            ),
            experiment_source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        )
        for entry in manifest["experts"]:
            layer, expert = entry["layer"], entry["expert"]
            path = (capture / entry["file"]).resolve()
            if (
                not path.is_relative_to(capture.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("capture path/hash mismatch")
            data = load_file(str(path))
            if any(data[k].dtype != torch.bfloat16 for k in ("z", "z_gamma", "gamma")):
                raise ValueError("capture must preserve original BF16 tensors")
            if not all(torch.isfinite(t).all() for t in data.values()):
                raise ValueError("nonfinite capture")
            n = len(data["z"])
            if data["router_weight"].shape != (n,) or (data["router_weight"] < 0).any():
                raise ValueError("invalid captured router weights")
            if (
                n < (manifest["minimum"] if synthetic else 128)
                or n > 512
                or n != entry["retained"]
            ):
                raise ValueError("insufficient/mismatched token count")
            gamma = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
            if not torch.equal(gamma, data["gamma"]):
                raise ValueError("captured gamma mismatch")
            if (
                data["z"].shape != data["z_gamma"].shape
                or data["z"].shape[1] != gamma.numel()
            ):
                raise ValueError("capture shape mismatch")
            positions = list(
                zip(data["sample_id"].tolist(), data["token_position"].tolist())
            )
            if (
                len(set(positions)) != n
                or [list(p) for p in positions] != entry["sample_positions"]
            ):
                raise ValueError("token identity mismatch")
            ws = [cp.load(cp.key(layer, expert, p)) for p in PROJECTIONS]
            if not synthetic:
                for p, w in zip(PROJECTIONS, ws):
                    digest = hashlib.sha256(
                        w.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    if (
                        manifest.get("expert_weight_sha256", {}).get(
                            cp.key(layer, expert, p)
                        )
                        != digest
                    ):
                        raise ValueError("captured expert weight identity mismatch")
            out, act, weight = evaluate(data["z"], data["z_gamma"], gamma, ws, dtype)
            for key, rows in [
                ("outputs", out),
                ("activations", act),
                ("weights", weight),
            ]:
                result[key].extend(dict(layer=layer, expert=expert, **r) for r in rows)
            print(f"layer={layer} expert={expert} tokens={n} complete", flush=True)
        aggregate = []
        for layer in selection:
            for branch in "ABC":
                for mode in MODES:
                    rows = [
                        r
                        for r in result["outputs"]
                        if (r["layer"], r["branch"], r["mode"]) == (layer, branch, mode)
                    ]
                    tokens = sum(r["tokens"] for r in rows)
                    aggregate.append(
                        dict(
                            layer=layer,
                            branch=branch,
                            mode=mode,
                            experts=len(rows),
                            tokens=tokens,
                            mean_nmse=float(np.mean([r["common_nmse"] for r in rows])),
                            token_weighted_nmse=sum(
                                r["common_nmse"] * r["tokens"] for r in rows
                            )
                            / tokens,
                            pooled_nmse=sum(r["common_sse"] for r in rows)
                            / (sum(r["common_reference_energy"] for r in rows) + 1e-30),
                        )
                    )
        result["layer_summary"] = aggregate
        associations = []
        for branch in "ABC":
            for mode in ("W4A16", "W16A4", "W4A4"):
                for projection in PROJECTIONS:
                    ys = [
                        r
                        for r in result["outputs"]
                        if r["branch"] == branch and r["mode"] == mode
                    ]
                    for kind, table in (
                        ("weight", result["weights"]),
                        ("activation", result["activations"]),
                    ):
                        xs = {
                            (r["layer"], r["expert"]): r["nmse"]
                            for r in table
                            if r["branch"] == branch
                            and r["projection"] == projection
                            and (kind == "weight" or r["mode"] == mode)
                        }
                        associations.append(
                            dict(
                                branch=branch,
                                mode=mode,
                                projection=projection,
                                kind=kind,
                                **corr(
                                    [xs[(r["layer"], r["expert"])] for r in ys],
                                    [r["common_nmse"] for r in ys],
                                ),
                            )
                        )
        result["associations"] = associations
        write_csv(output / "associations.csv", associations)
        for key in ("weights", "activations", "outputs", "layer_summary"):
            write_csv(output / f"{key}.csv", result[key])
        atomic_json(output / "capture_manifest.json", manifest)
        lines = [
            "# E2 local expert reconstruction",
            "",
            f"Source: {result['data_kind']}; arithmetic: {dtype}.",
            "All errors use common original A-FP coordinates; C is inverse-rotated.",
            "FP rows expose transform rounding. Inspect branch_fp errors separately.",
            "W4A16 and W16A4 are switch controls, not additive causal contributions.",
            "Activation QDQ errors with applied=False are diagnostic only.",
            "",
            "|Layer|Mode|A mean NMSE|B mean NMSE|C mean NMSE|C/B gain %|C/A gain %|",
            "|---|---|---|---|---|---|---|",
        ]
        for layer in selection:
            for mode in MODES:
                nmse = {
                    r["branch"]: r["mean_nmse"]
                    for r in aggregate
                    if r["layer"] == layer and r["mode"] == mode
                }

                def gain(b):
                    return 100 * (nmse[b] - nmse["C"]) / nmse[b] if nmse[b] else None

                lines.append(
                    f"|{layer}|{mode}|{nmse['A']:.6g}|{nmse['B']:.6g}|"
                    f"{nmse['C']:.6g}|{gain('B')}|{gain('A')}|"
                )
        lines += [
            "",
            "Positive gains indicate lower common-reference output error.",
            "No routing weights are applied. Unequal token counts are reported; pooled",
            "and token-weighted NMSE are separate columns in layer_summary.csv.",
            "No downstream accuracy or selective rotation is evaluated.",
            "Capture tensors remain in the input capture directory, not duplicated.",
        ]
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(output / "results.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arithmetic", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(a.checkpoint, a.capture, a.output, getattr(torch, a.arithmetic))


if __name__ == "__main__":
    main()
