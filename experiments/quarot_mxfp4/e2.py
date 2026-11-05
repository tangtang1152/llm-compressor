"""E2 replay of verified original-BF16 routed activations. No model loader."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from .core import inverse_hidden, qdq, rotation, transform
from .diagnose import atomic_json, write_csv
from .e2_capture import SELECTION, file_hash, read_manifest, read_part, token_priority
from .run import Checkpoint, provenance

MODES = {
    "FP": (False, False),
    "W4A16": (True, False),
    "W16A4": (False, True),
    "W4A4": (True, True),
}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metric_values(sse, energy, elements, token_nmse=None):
    result = dict(
        sse=sse,
        reference_energy=energy,
        elements=elements,
        mse=sse / elements,
        nmse=ratio(sse, energy),
    )
    if token_nmse is None:
        return result
    finite = token_nmse[np.isfinite(token_nmse)]
    result.update(
        token_nmse_p50=float(np.quantile(finite, 0.5)) if len(finite) else None,
        token_nmse_p95=float(np.quantile(finite, 0.95)) if len(finite) else None,
        token_nmse_defined=len(finite),
        _token_nmse=token_nmse,
    )
    return result


def error_stats(ref, actual, *, per_token=True):
    r, a = ref.double(), actual.double()
    if (
        r.shape != a.shape
        or not r.numel()
        or not torch.isfinite(a).all()
        or not torch.isfinite(r).all()
    ):
        raise ValueError("invalid comparison tensors")
    squared = (a - r).square()
    per_energy = r.square().sum(-1)
    token_nmse = (
        torch.where(per_energy > 0, squared.sum(-1) / per_energy, torch.nan).numpy()
        if per_token
        else None
    )
    return metric_values(
        squared.sum().item(), per_energy.sum().item(), r.numel(), token_nmse
    )


@torch.no_grad()
def replay(x, weights, quantized, mode, dtype=torch.bfloat16):
    wq, aq = MODES[mode]
    records = []

    def linear(value, index):
        value = value.to(dtype)
        if aq:
            rounded = qdq(value, activation=True)
            records.append(
                dict(
                    projection=PROJECTIONS[index],
                    **error_stats(value, rounded, per_token=False),
                )
            )
        w = (quantized if wq else weights)[index].to(dtype)
        return F.linear(rounded if aq else value, w)

    gate, up = linear(x, 0), linear(x, 1)
    # Down receives this mode's actual gate/up computation, never ideal activations.
    output = linear(F.silu(gate) * up, 2)
    return output, records


@torch.no_grad()
def prepare_weights(gamma, weights):
    rot = rotation(gamma.numel())
    branches = {
        "A": weights,
        "B": [transform(w, gamma, p) for w, p in zip(weights, PROJECTIONS)],
        "C": [transform(w, gamma, p, True) for w, p in zip(weights, PROJECTIONS)],
    }
    return rot, {b: (ws, [qdq(w) for w in ws]) for b, ws in branches.items()}


@torch.no_grad()
def evaluate(z, z_gamma, gamma, weights, dtype=torch.bfloat16, *, prepared=None):
    rot, branches = (
        prepared if prepared is not None else prepare_weights(gamma, weights)
    )
    inputs = {"A": z_gamma, "B": z, "C": rot.apply(z.float()).to(z.dtype)}
    outputs, activations = [], []
    common = None
    for branch, (ws, quantized) in branches.items():
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
    return outputs, activations


def select_tokens(capture, entry, gamma, seen_samples, max_tokens=512):
    """Validate the full expert archive; choose by identity, never file/rank order."""
    if max_tokens < 0 or 0 < max_tokens < 128:
        raise ValueError("max_replay_tokens must be 0 or >=128")
    identities, candidates = set(), []
    layer, expert = entry["layer"], entry["expert"]
    for part_index, part in enumerate(entry["parts"]):
        data, positions = read_part(capture, part)
        if not torch.equal(data["gamma"], gamma):
            raise ValueError("captured gamma mismatch")
        if identities.intersection(positions):
            raise ValueError("duplicate token identity")
        if any((layer, sample) not in seen_samples for sample, _ in positions):
            raise ValueError("token outside sample coverage")
        identities.update(positions)
        candidates.extend(
            (token_priority(layer, expert, sample, pos), sample, pos, part_index, i)
            for i, (sample, pos) in enumerate(positions)
        )
    if len(candidates) != entry["available"] or entry["retained"] != entry["available"]:
        raise ValueError("lost/truncated archive tokens")
    if len(candidates) < 128:
        raise ValueError("insufficient routed tokens: requires at least128")
    return heapq.nsmallest(max_tokens, candidates) if max_tokens else sorted(candidates)


def archive_batches(capture, entry, chosen, hidden, batch_size=512):
    """Only one replay batch and one mmap-backed archive block at a time."""
    for offset in range(0, len(chosen), batch_size):
        rows = chosen[offset : offset + batch_size]
        z = torch.empty(len(rows), hidden, dtype=torch.bfloat16)
        zg = torch.empty_like(z)
        by_part = {}
        for dest, (_, _, _, part, row) in enumerate(rows):
            by_part.setdefault(part, []).append((dest, row))
        for part, indexes in by_part.items():
            data = load_file(str(Path(capture) / entry["parts"][part]["file"]))
            dest, src = zip(*indexes)
            z[list(dest)] = data["z"][list(src)]
            zg[list(dest)] = data["z_gamma"][list(src)]
        yield z, zg


def combine_batches(rows, keys, prefixes):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for values, items in groups.items():
        out = dict(zip(keys, values))
        out["tokens"] = sum(r["tokens"] for r in items)
        for prefix in prefixes:
            stats = metric_values(
                sum(r[prefix + "sse"] for r in items),
                sum(r[prefix + "reference_energy"] for r in items),
                sum(r[prefix + "elements"] for r in items),
                np.concatenate([r[prefix + "_token_nmse"] for r in items])
                if prefix + "_token_nmse" in items[0]
                else None,
            )
            stats.pop("_token_nmse", None)
            out.update({prefix + k: v for k, v in stats.items()})
        result.append(out)
    return result


def summarize(rows, keys, prefixes):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    results = []
    for values, items in groups.items():
        out = dict(
            zip(keys, values),
            experts=len(items),
            tokens=sum(r["tokens"] for r in items),
        )
        for prefix in prefixes:
            sse = sum(r[prefix + "sse"] for r in items)
            energy = sum(r[prefix + "reference_energy"] for r in items)
            elements = sum(r[prefix + "elements"] for r in items)
            nmse = [r[prefix + "nmse"] for r in items]
            out.update(
                {
                    prefix + "pooled_mse": sse / elements,
                    prefix + "pooled_nmse": ratio(sse, energy),
                    prefix + "mean_expert_nmse": sum(nmse) / len(nmse)
                    if all(v is not None for v in nmse)
                    else None,
                    prefix + "sse": sse,
                    prefix + "reference_energy": energy,
                    prefix + "elements": elements,
                }
            )
        results.append(out)
    return results


def run(
    checkpoint,
    capture,
    output,
    dtype=torch.bfloat16,
    *,
    synthetic=False,
    max_replay_tokens=512,
):
    if max_replay_tokens < 0 or 0 < max_replay_tokens < 128:
        raise ValueError("max_replay_tokens must be 0 or >=128")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", outputs=[], activations=[], replay_selection=[])
    atomic_json(output / "results.json", result)
    try:
        capture = Path(capture)
        manifest = read_manifest(capture)
        selection = {int(k): v for k, v in manifest["selection"].items()}
        if manifest["status"] != "complete" or (
            not synthetic
            and (manifest["source"]["kind"] != "real_bf16" or selection != SELECTION)
        ):
            raise ValueError("requires complete real capture with exact E1.3 selection")
        cp = Checkpoint(checkpoint, [], 0, 1234)
        cp.select(list(selection), 3, 1234, selections=selection)
        if not synthetic and (
            manifest["source"].get("checkpoint_index_sha256")
            != file_hash(cp.index_path)
            or manifest["source"].get("checkpoint_config_sha256")
            != file_hash(Path(checkpoint) / "config.json")
            or manifest["source"].get("routing_alignment_verified") is not True
        ):
            raise ValueError("capture checkpoint/alignment mismatch")
        result.update(
            provenance=provenance(),
            source=manifest["source"],
            capture_manifest_sha256=file_hash(capture / "capture_manifest.json"),
            data_kind=manifest["source"]["kind"],
            arithmetic=str(dtype),
            selection=selection,
            reference="Original A FP, unweighted expert output; C inverse-rotated",
            configuration=dict(
                checkpoint=str(checkpoint),
                capture=str(capture.resolve()),
                max_replay_tokens=max_replay_tokens,
                replay_batch_size=512,
                seed=1234,
                rotation_block_size=32,
                transform_precision="float32",
                weight_writeback="bfloat16",
                quantization="CT MXFP4 E2M1/E8M0",
                group_size=32,
                axis=-1,
                weights="static minmax",
                activations="dynamic per token/group minmax",
                zero_reference_energy="NMSE null; token quantiles over defined NMSE "
                "only",
            ),
            experiment_source_sha256={
                p.name: file_hash(p) for p in Path(__file__).parent.glob("e2*.py")
            },
        )
        for entry in manifest["experts"]:
            layer, expert = entry["layer"], entry["expert"]
            gamma = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
            chosen = select_tokens(
                capture,
                entry,
                gamma,
                set(map(tuple, manifest["seen_samples"])),
                max_replay_tokens,
            )
            positions = [[r[1], r[2]] for r in chosen]
            selection_info = dict(
                layer=layer,
                expert=expert,
                available_tokens=entry["available"],
                replay_tokens=len(chosen),
                token_identity_sha256=hashlib.sha256(
                    json.dumps(positions, separators=(",", ":")).encode()
                ).hexdigest(),
            )
            # Small default selection is inspectable. All-token runs retain only a
            # fingerprint; identities remain in the full server archive.
            if max_replay_tokens:
                selection_info["sample_positions"] = positions
            result["replay_selection"].append(selection_info)
            ws = [cp.load(cp.key(layer, expert, p)) for p in PROJECTIONS]
            if not synthetic:
                for p, w in zip(PROJECTIONS, ws):
                    digest = hashlib.sha256(
                        w.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    if (
                        manifest["expert_weight_sha256"].get(cp.key(layer, expert, p))
                        != digest
                    ):
                        raise ValueError("captured expert weight identity mismatch")
            prepared = prepare_weights(gamma, ws)
            outs, acts = [], []
            for z, zg in archive_batches(capture, entry, chosen, gamma.numel()):
                out, act = evaluate(z, zg, gamma, ws, dtype, prepared=prepared)
                outs.extend(out)
                acts.extend(dict(r, tokens=len(z)) for r in act)
            for table, rows, keys, prefixes in (
                ("outputs", outs, ("branch", "mode"), ("common_", "branch_fp_")),
                ("activations", acts, ("branch", "mode", "projection"), ("",)),
            ):
                result[table].extend(
                    dict(
                        layer=layer,
                        expert=expert,
                        available_tokens=entry["available"],
                        **r,
                    )
                    for r in combine_batches(rows, keys, prefixes)
                )
            del prepared, ws, outs, acts
            print(
                f"layer={layer} expert={expert} available={entry['available']} "
                f"replay={len(chosen)}",
                flush=True,
            )
        result["layer_summary"] = summarize(
            result["outputs"], ("layer", "branch", "mode"), ("common_", "branch_fp_")
        )
        result["activation_summary"] = summarize(
            result["activations"], ("layer", "projection", "branch", "mode"), ("",)
        )
        for key in ("activations", "outputs", "activation_summary", "layer_summary"):
            write_csv(output / f"{key}.csv", result[key])
        lines = [
            "# E2 expert reconstruction",
            "",
            f"Source: {result['data_kind']}; arithmetic: {dtype}.",
            "Common reference is original A FP; C outputs are inverse-rotated.",
            "Branch-FP errors separately expose quantization vs transform rounding.",
            "Activation metrics include only W16A4/W4A4 actual QDQ inputs.",
            "Pooled MSE = total SSE/elements; pooled NMSE = total "
            "SSE/reference energy.",
            "Mean Expert NMSE is the arithmetic mean of expert NMSE, a "
            "different measure.",
            "",
            "|Layer|Mode|A pooled NMSE|B pooled NMSE|C pooled NMSE|C/B gain "
            "%|C/A gain %|",
            "|---|---|---|---|---|---|---|",
        ]
        for layer in selection:
            for mode in MODES:
                ns = {
                    r["branch"]: r["common_pooled_nmse"]
                    for r in result["layer_summary"]
                    if r["layer"] == layer and r["mode"] == mode
                }

                def gain(b):
                    return (
                        100 * (ns[b] - ns["C"]) / ns[b]
                        if ns[b] and ns["C"] is not None
                        else None
                    )

                lines.append(
                    f"|{layer}|{mode}|{ns['A']}|{ns['B']}|{ns['C']}|{gain('B')}|{gain('A')}|"
                )
        lines += [
            "",
            "Positive gain means lower common-reference expert output NMSE.",
            "Inspect W16A4 vs W4A16 to compare activation and weight "
            "sensitivity; errors are not additive.",
            "No downstream accuracy or selective-rotation conclusion follows "
            "from this local reconstruction.",
            "Full activations remain in the server archive; only this "
            "six-file result directory needs sharing.",
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
    p.add_argument(
        "--max-replay-tokens",
        type=int,
        default=512,
        help="0 replays all archived tokens; otherwise >=128",
    )
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(
        a.checkpoint,
        a.capture,
        a.output,
        getattr(torch, a.arithmetic),
        max_replay_tokens=a.max_replay_tokens,
    )


if __name__ == "__main__":
    main()
