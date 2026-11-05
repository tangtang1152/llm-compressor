"""Validated capture boundary, not a model loader or unverified pipeline driver."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .diagnose import atomic_json

SELECTION = {
    3: [76, 158, 202],
    6: [101, 182, 240],
    16: [70, 81, 164],
    40: [60, 165, 236],
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_priority(layer, expert, sample, position):
    return hashlib.sha256(
        f"1234:{layer}:{expert}:{sample}:{position}".encode()
    ).hexdigest()


def read_part(root, part):
    """Validate one lossless archive block; memory is bounded by one sample."""
    root = Path(root).resolve()
    path = (root / part["file"]).resolve()
    if (
        not path.is_relative_to(root)
        or path.is_symlink()
        or file_hash(path) != part["sha256"]
    ):
        raise ValueError("capture path/hash mismatch")
    data = load_file(str(path))
    if set(data) != {
        "z",
        "z_gamma",
        "gamma",
        "router_weight",
        "sample_id",
        "token_position",
    }:
        raise ValueError("capture tensor keys mismatch")
    z = data["z"]
    n = part["retained"]
    if (
        z.ndim != 2
        or len(z) != n
        or n <= 0
        or data["z_gamma"].shape != z.shape
        or data["gamma"].shape != (z.shape[1],)
        or any(data[k].dtype != torch.bfloat16 for k in ("z", "z_gamma", "gamma"))
        or any(
            data[k].shape != (n,)
            for k in ("router_weight", "sample_id", "token_position")
        )
        or any(data[k].dtype != torch.int64 for k in ("sample_id", "token_position"))
        or not data["router_weight"].is_floating_point()
        or not all(torch.isfinite(t).all() for t in data.values())
        or any(
            (data[k] < 0).any()
            for k in ("router_weight", "sample_id", "token_position")
        )
    ):
        raise ValueError("invalid capture tensors/count/dtype")
    identities = list(zip(data["sample_id"].tolist(), data["token_position"].tolist()))
    if len(set(identities)) != n:
        raise ValueError("duplicate token identity")
    if path.stat().st_size != part["file_bytes"]:
        raise ValueError("capture byte count mismatch")
    if sum(t.numel() * t.element_size() for t in data.values()) != part["tensor_bytes"]:
        raise ValueError("capture tensor byte count mismatch")
    return data, identities


def read_manifest(root):
    manifest = json.loads((Path(root) / "capture_manifest.json").read_text())
    if (
        manifest.get("format_version") != 2
        or manifest.get("archive_policy") != "all_routed_valid_tokens"
    ):
        raise ValueError(
            "requires full archive format 2; legacy capped captures are not "
            "full archives"
        )
    expected = {
        (int(layer), e)
        for layer, experts in manifest["selection"].items()
        for e in experts
    }
    keys = [(e["layer"], e["expert"]) for e in manifest["experts"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("capture expert set mismatch")
    for entry in manifest["experts"]:
        if entry["available"] != entry["retained"] or entry["retained"] != sum(
            p["retained"] for p in entry["parts"]
        ):
            raise ValueError("truncated capture: retained != available/parts")
    seen = list(map(tuple, manifest["seen_samples"]))
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate captured sample")
    if set(manifest["gamma_bf16"]) != set(manifest["selection"]):
        raise ValueError("missing layer gamma")
    return manifest


class RoutedCapture:
    """Caller supplies actual BF16 norm/router outputs and global sample IDs.

    Feed each original, nonpadding sample exactly once per layer, only after the
    sequential graph's norm/router/expert alignment has been verified.
    """

    def __init__(self, selection=None, *, output=None):
        self.selection = SELECTION if selection is None else selection
        self.buffers = {
            (layer_id, e): [] for layer_id, ids in self.selection.items() for e in ids
        }
        self.counts = dict.fromkeys(self.buffers, 0)
        self.seen = set()
        self.norm_errors = []
        self.gammas = {}
        self.norm_eps = {}
        self.weight_hashes = {}
        self.parts = {key: [] for key in self.buffers}
        self.output = Path(output).resolve() if output is not None else None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output / "captured_activations").mkdir()

    def write_part(self, key, tensors):
        layer, expert = key
        path = (
            self.output
            / "captured_activations"
            / f"L{layer}-E{expert}-P{len(self.parts[key])}.safetensors"
        )
        save_file(tensors, str(path))
        self.parts[key].append(
            dict(
                file=path.relative_to(self.output).as_posix(),
                sha256=file_hash(path),
                retained=len(tensors["z"]),
                file_bytes=path.stat().st_size,
                tensor_bytes=sum(
                    t.numel() * t.element_size() for t in tensors.values()
                ),
            )
        )

    @torch.no_grad()
    def add(
        self,
        layer,
        sample_id,
        norm,
        hidden,
        norm_output,
        indices,
        routing_weights,
        valid_mask,
    ):
        if (layer, sample_id) in self.seen:
            raise ValueError(
                "duplicate sample/layer: propagation/tracing hooks must be excluded"
            )
        if (
            layer not in self.selection
            or hidden.ndim != 2
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError("expected selected layer and BF16 [tokens,hidden]")
        if (
            indices.ndim != 2
            or indices.shape != routing_weights.shape
            or indices.shape[0] != hidden.shape[0]
            or indices.dtype != torch.int64
        ):
            raise ValueError("invalid original routing tensors")
        if (
            norm_output.shape != hidden.shape
            or norm_output.dtype != hidden.dtype
            or valid_mask.shape != (len(hidden),)
        ):
            raise ValueError("norm/mask alignment mismatch")
        if type(norm).__name__ != "GlmMoeDsaRMSNorm":
            raise ValueError("only inspected GlmMoeDsaRMSNorm implementation supported")
        if not all(
            torch.isfinite(t).all()
            for t in (hidden, norm_output, routing_weights, norm.weight)
        ):
            raise ValueError("nonfinite capture")
        if (routing_weights < 0).any() or (indices < 0).any():
            raise ValueError("negative route weight/index")
        # Instantiate only a tiny norm; invoke the actual class's forward with unit
        # gain, never divide by gamma or mutate the offloaded model's norm.
        unit = type(norm)(hidden.shape[-1], eps=norm.variance_epsilon).to(hidden)
        unit.weight.fill_(1)
        z = unit(hidden)
        reconstructed = z * norm.weight.to(z)
        # NPU kernels do not generally support FP64. Diagnostic reductions are
        # FP32 here; CPU replay metrics use FP64. No full norm batch CPU copy.
        delta = reconstructed.float() - norm_output.float()
        discrepancy = dict(
            mse=delta.square().mean().item(),
            nmse=(
                delta.square().sum() / (norm_output.float().square().sum() + 1e-30)
            ).item(),
        )
        torch.testing.assert_close(reconstructed, norm_output, rtol=0.01, atol=0.01)
        self.norm_errors.append(dict(layer=layer, sample_id=sample_id, **discrepancy))
        gamma = norm.weight.detach().cpu().clone()
        if layer in self.gammas and not torch.equal(gamma, self.gammas[layer]):
            raise ValueError("gamma changed during capture")
        self.gammas[layer] = gamma
        self.norm_eps[layer] = norm.variance_epsilon
        self.seen.add((layer, sample_id))
        for expert in self.selection[layer]:
            token, slot = torch.where((indices == expert) & valid_mask.bool()[:, None])
            if len(token) != len(torch.unique(token)):
                raise ValueError("duplicate expert assignment within a token")
            key = (layer, expert)
            self.counts[key] += len(token)
            if len(token):
                tensors = dict(
                    z=z[token].cpu(),
                    z_gamma=norm_output[token].cpu(),
                    router_weight=routing_weights[token, slot].cpu(),
                    sample_id=torch.full((len(token),), sample_id, dtype=torch.int64),
                    token_position=token.cpu(),
                    gamma=gamma,
                )
                if self.output is None:  # tiny in-memory fixtures only
                    self.buffers[key].append(tensors)
                else:
                    self.write_part(key, tensors)

    def save(self, output, source, *, rank_shard=False):
        output = Path(output).resolve()
        if self.output is None:
            output.mkdir(parents=True, exist_ok=False)
            (output / "captured_activations").mkdir()
            self.output = output
            for key, chunks in self.buffers.items():
                for tensors in chunks:
                    self.write_part(key, tensors)
        elif output != self.output or (output / "capture_manifest.json").exists():
            raise FileExistsError(
                "capture output must be the collector's fresh directory"
            )
        real = source.get("kind") == "real_bf16"
        required = (
            "dataset",
            "dataset_sha256",
            "preprocessing",
            "checkpoint_index_sha256",
            "capture_driver",
            "routing_alignment_verified",
        )
        if real and (
            any(not source.get(k) for k in required)
            or source["routing_alignment_verified"] is not True
        ):
            raise ValueError(
                "real capture requires verified source and routing alignment metadata"
            )
        if source.get("kind") not in ("real_bf16", "synthetic_test"):
            raise ValueError("unknown capture source")
        manifest = dict(
            status="incomplete",
            source=source,
            selection=self.selection,
            format_version=2,
            archive_policy="all_routed_valid_tokens",
            seed=1234,
            experts=[],
            norm_errors=self.norm_errors,
            norm_epsilon=self.norm_eps,
            gamma_bf16={layer: g.float().tolist() for layer, g in self.gammas.items()},
            expert_weight_sha256=self.weight_hashes,
            seen_samples=sorted(self.seen),
        )
        for (layer, expert), parts in self.parts.items():
            manifest["experts"].append(
                dict(
                    layer=layer,
                    expert=expert,
                    available=self.counts[(layer, expert)],
                    retained=sum(p["retained"] for p in parts),
                    parts=parts,
                )
            )
        atomic_json(output / "capture_manifest.json", manifest)
        read_manifest(output)
        for entry in manifest["experts"]:
            identities = set()
            for part in entry["parts"]:
                _, positions = read_part(output, part)
                if identities.intersection(positions):
                    raise ValueError("duplicate token identity")
                identities.update(positions)
        manifest["tensor_bytes"] = sum(
            p["tensor_bytes"] for e in manifest["experts"] for p in e["parts"]
        )
        manifest["archive_file_bytes"] = sum(
            p["file_bytes"] for e in manifest["experts"] for p in e["parts"]
        )
        manifest["status"] = "rank_shard_complete" if rank_shard else "complete"
        atomic_json(output / "capture_manifest.json", manifest)
        return manifest
