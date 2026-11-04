"""Selective, CPU-only E1 runner. Run from the repository with python -m ...run."""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import random
import re
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from .core import e0, metrics, qdq, transform

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class Checkpoint:
    def __init__(
        self, root, layers, count, seed, experts=None, max_elements=64_000_000
    ):
        self.root = Path(root).resolve()
        self.max_elements = max_elements
        self.config = json.loads((self.root / "config.json").read_text())
        self.index_path = self.root / "model.safetensors.index.json"
        self.index = json.loads(self.index_path.read_text())["weight_map"]
        cfg = self.config
        if cfg.get("quantization_config") or cfg.get("quarot_config"):
            raise ValueError("requires original, unquantized, untransformed checkpoint")
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError("only SiLU experts supported")
        h, m = cfg["hidden_size"], cfg["moe_intermediate_size"]
        if h % 32 or m % 32:
            raise ValueError("hidden/intermediate must be divisible by 32")
        n = cfg["num_hidden_layers"]
        actual = {
            int(match[1])
            for key in self.index
            if (match := re.match(r"model\.layers\.(\d+)\.", key))
        }
        expected = set(range(n))
        allowed = (expected, expected | {n})
        if actual not in allowed:
            raise ValueError(
                "index layer topology does not match config: "
                f"expected main layers 0..{n-1} "
                f"with optional MTP layer {n}, "
                f"got {sorted(actual)}"
            )
        self.select(layers, count, seed, experts)

    def select(self, layers, count, seed, experts=None, selections=None):
        """Validate selections using the already loaded config/index."""
        cfg = self.config
        h, m = cfg["hidden_size"], cfg["moe_intermediate_size"]
        n, ne = cfg["num_hidden_layers"], cfg["n_routed_experts"]
        self.selection = {}
        for layer in layers:
            if not 0 <= layer < n or layer < cfg.get("first_k_dense_replace", 0):
                raise ValueError(f"not a routed layer: {layer}")
            selected = (selections or {}).get(layer, experts)
            ids = (
                sorted(selected)
                if selected is not None
                else sorted(random.Random(seed + layer).sample(range(ne), count))
            )
            if (
                not ids
                or len(ids) != len(set(ids))
                or any(i < 0 or i >= ne for i in ids)
            ):
                raise ValueError("invalid expert selection")
            self.selection[layer] = ids
            norm = f"model.layers.{layer}.post_attention_layernorm.weight"
            if norm.removesuffix(".weight") + ".bias" in self.index:
                raise ValueError("biased norm unsupported")
            self.check(norm, (h,))
            for expert in ids:
                for p in PROJECTIONS:
                    key = self.key(layer, expert, p)
                    if key.removesuffix(".weight") + ".bias" in self.index:
                        raise ValueError("biased experts unsupported")
                    self.check(key, (h, m) if p == "down_proj" else (m, h))

    @staticmethod
    def key(layer, expert, projection):
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def shard(self, key):
        if key not in self.index:
            raise ValueError(
                f"unsupported/missing explicit tensor key: {key}; "
                "fused layouts require an adapter"
            )
        path = (self.root / self.index[key]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("shard outside checkpoint directory")
        return path

    def check(self, key, shape):
        with safe_open(self.shard(key), framework="pt", device="cpu") as f:
            s = f.get_slice(key)
            if tuple(s.get_shape()) != shape or s.get_dtype() != "BF16":
                raise ValueError(
                    f"{key}: expected BF16 {shape}, got {s.get_dtype()} {s.get_shape()}"
                )
            if __import__("math").prod(shape) > self.max_elements:
                raise ValueError(f"{key}: exceeds --max-elements memory guard")

    def load(self, key):
        with safe_open(self.shard(key), framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {key}")
        return value


def provenance():
    import compressed_tensors
    from compressed_tensors.quantization import quant_args, quant_scheme
    from compressed_tensors.quantization.lifecycle import forward
    from compressed_tensors.quantization.utils import fp4_utils, helpers, mxfp_utils

    import llmcompressor
    from llmcompressor.modeling import fuse
    from llmcompressor.modifiers.transform.quarot import rotation

    paths = [
        Path(inspect.getfile(m))
        for m in (
            mxfp_utils,
            fp4_utils,
            helpers,
            quant_scheme,
            quant_args,
            forward,
            rotation,
            fuse,
        )
    ]
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return dict(
        git_commit=git("rev-parse", "HEAD"),
        git_status=git("status", "--short"),
        versions={
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "llmcompressor",
                "compressed-tensors",
                "safetensors",
                "transformers",
            )
        },
        package_paths=dict(
            llmcompressor=llmcompressor.__file__,
            compressed_tensors=compressed_tensors.__file__,
        ),
        implementation_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 40, 77])
    p.add_argument("--experts", type=int, nargs="+")
    p.add_argument("--experts-per-layer", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-elements", type=int, default=64_000_000)
    p.add_argument("--e0-hidden", type=int, default=6144)
    p.add_argument("--e0-intermediate", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        provenance=provenance(),
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        },
        rotation=dict(block_size=32, seed=a.seed, precision="float32", shifted=False),
        quantization=dict(
            format="MXFP4 E2M1/E8M0",
            group_size=32,
            axis=-1,
            weight="static minmax",
            activation="dynamic minmax",
        ),
        evaluation=(
            "BF16 stored transforms and CT QDQ; FP64 metric reduction; "
            "E0 FP32 matmul"
        ),
        rows=[],
    )
    try:
        result["e0"] = e0(a.e0_hidden, a.e0_intermediate, a.seed)
        result["data_kind"] = (
            "synthetic E0 only" if a.checkpoint is None else "checkpoint E1"
        )
        if a.checkpoint is not None:
            cp = Checkpoint(
                a.checkpoint,
                a.layers,
                a.experts_per_layer,
                a.seed,
                a.experts,
                a.max_elements,
            )
            if cp.config.get("_experiment_synthetic"):
                result["data_kind"] = (
                    "SYNTHETIC sharded fixture E1; NOT GLM measurements"
                )
            result["model_config"] = cp.config
            result["index_sha256"] = hashlib.sha256(
                cp.index_path.read_bytes()
            ).hexdigest()
            result["selection"] = cp.selection
            for layer, ids in cp.selection.items():
                gain = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
                for expert in ids:
                    for projection in PROJECTIONS:
                        key = cp.key(layer, expert, projection)
                        weight = cp.load(key)
                        row = dict(
                            layer=layer,
                            expert=expert,
                            projection=projection,
                            tensor_key=key,
                            shard=cp.index[key],
                        )
                        for label, rotated in [("B", False), ("C", True)]:
                            w = transform(weight, gain, projection, rotated, a.seed)
                            q = qdq(w)
                            row.update(
                                {f"{label}_{k}": v for k, v in metrics(w, q).items()}
                            )
                            del w, q
                        b, c = row["B_weight_nmse"], row["C_weight_nmse"]
                        row["gain_percent"] = 100 * (b - c) / b if b else None
                        result["rows"].append(row)
                        del weight
                        print(
                            f'{key}: B={b:.6g} C={c:.6g} gain={row["gain_percent"]}',
                            flush=True,
                        )
                del gain
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        (a.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        rows = result["rows"]
        if rows:
            with (a.output / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        lines = [
            "# QuaRot MXFP4 measurement",
            "",
            f'Status: {result["status"]}.',
            f'Data: {result.get("data_kind", "E0 failed")}.',
            "Full Expert real-activation W4A4 NMSE: **not measured**.",
            "See results.json for E0 errors, provenance, config, "
            "selection and exclusions.",
            "",
            "| Layer | Expert | Projection | B weight NMSE | C weight NMSE | Gain % |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f'| {r["layer"]} | {r["expert"]} | {r["projection"]} | '
            f'{r["B_weight_nmse"]:.6g} | {r["C_weight_nmse"]:.6g} | '
            f'{r["gain_percent"]} |'
            for r in rows
        ]
        (a.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
