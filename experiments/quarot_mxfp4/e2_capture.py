"""Validated capture boundary, not a model loader or unverified pipeline driver."""

import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file

from .diagnose import atomic_json

SELECTION = {
    3: [76, 158, 202],
    6: [101, 182, 240],
    16: [70, 81, 164],
    40: [60, 165, 236],
}


class RoutedCapture:
    """Caller supplies actual BF16 norm/router outputs and global sample IDs.

    Feed each original, nonpadding sample exactly once per layer, only after the
    sequential graph's norm/router/expert alignment has been verified.
    """

    def __init__(self, selection=None, target=512, minimum=128):
        if not 1 <= minimum <= target:
            raise ValueError("invalid token thresholds")
        self.selection = SELECTION if selection is None else selection
        self.target, self.minimum = target, minimum
        self.buffers = {
            (layer_id, e): [] for layer_id, ids in self.selection.items() for e in ids
        }
        self.counts = dict.fromkeys(self.buffers, 0)
        self.seen = set()
        self.norm_errors = []
        self.gammas = {}
        self.norm_eps = {}
        self.weight_hashes = {}

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
            max_abs=delta.abs().max().item(),
            mean_abs=delta.abs().mean().item(),
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
            # Deterministic hash-priority reservoir: order/rank-shard independent
            # when fed global sample IDs. Store only the best target candidates.
            candidates = [
                (
                    hashlib.sha256(
                        f"1234:{layer}:{expert}:{sample_id}:{int(t)}".encode()
                    ).hexdigest(),
                    int(t),
                    int(s),
                )
                for t, s in zip(token.cpu().tolist(), slot.cpu().tolist())
            ]
            existing = self.buffers[key]
            threshold = (
                sorted(r[0] for r in existing)[-1]
                if len(existing) == self.target
                else None
            )
            chosen = [
                r
                for r in sorted(candidates)[: self.target]
                if threshold is None or r[0] < threshold
            ]
            if chosen:
                ts = torch.tensor([r[1] for r in chosen], device=z.device)
                ss = torch.tensor([r[2] for r in chosen], device=z.device)
                zs, ys, rs = (
                    z[ts].cpu(),
                    norm_output[ts].cpu(),
                    routing_weights[ts, ss].cpu(),
                )
                for i, (priority, t, _) in enumerate(chosen):
                    existing.append(
                        (
                            priority,
                            sample_id,
                            t,
                            zs[i].clone(),
                            ys[i].clone(),
                            rs[i].clone(),
                        )
                    )
            self.buffers[key] = sorted(existing, key=lambda r: (r[0], r[1], r[2]))[
                : self.target
            ]

    def save(self, output, source, *, rank_shard=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
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
            target=self.target,
            minimum=self.minimum,
            seed=1234,
            experts=[],
            norm_errors=self.norm_errors,
            norm_epsilon=self.norm_eps,
            expert_weight_sha256=self.weight_hashes,
            seen_samples=sorted(self.seen),
        )
        for (layer, expert), rows in self.buffers.items():
            manifest["experts"].append(
                dict(
                    layer=layer,
                    expert=expert,
                    available=self.counts[(layer, expert)],
                    retained=len(rows),
                )
            )
        atomic_json(output / "capture_manifest.json", manifest)
        if not rank_shard and any(
            len(rows) < self.minimum for rows in self.buffers.values()
        ):
            raise ValueError(
                "insufficient routed tokens; increase samples from same dataset, "
                "never fill randomly"
            )
        directory = output / "captured_activations"
        directory.mkdir()
        for entry in manifest["experts"]:
            layer, expert = entry["layer"], entry["expert"]
            rows = self.buffers[(layer, expert)]
            if not rows:
                continue
            tensors = dict(
                z=torch.stack([r[3] for r in rows]),
                z_gamma=torch.stack([r[4] for r in rows]),
                router_weight=torch.stack([r[5] for r in rows]),
                sample_id=torch.tensor([r[1] for r in rows]),
                token_position=torch.tensor([r[2] for r in rows]),
                gamma=self.gammas[layer],
            )
            path = directory / f"L{layer}-E{expert}.safetensors"
            save_file(tensors, str(path))
            entry.update(
                file=path.relative_to(output).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sample_positions=[[r[1], r[2]] for r in rows],
                router_weight_min=float(tensors["router_weight"].min()),
                router_weight_max=float(tensors["router_weight"].max()),
            )
        manifest["status"] = "rank_shard_complete" if rank_shard else "complete"
        atomic_json(output / "capture_manifest.json", manifest)
