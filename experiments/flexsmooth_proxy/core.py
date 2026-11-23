"""Shared, auditable numerics; no production Modifier changes."""

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path

import torch
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_scheme import MXFP4, MXFP8
from compressed_tensors.quantization.utils import calculate_qparams

from llmcompressor.modifiers.transform.flex_smooth.search import (
    _row_int8,
    _search_candidates,
    smooth_scale,
)


def digest(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def provenance():
    import compressed_tensors
    import compressed_tensors.quantization.lifecycle.forward_helpers as helpers
    import compressed_tensors.quantization.utils.mxfp_utils as mx

    import llmcompressor.modifiers.transform.flex_smooth.search as search

    files = [
        search.__file__,
        mx.__file__,
        helpers.__file__,
        __file__,
        str(Path(__file__).with_name("run.py")),
    ]
    return dict(
        torch=str(torch.__version__),
        compressed_tensors=importlib.metadata.version("compressed-tensors"),
        ct_path=compressed_tensors.__file__,
        sources={
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files
        },
    )


def sync(device):
    device = torch.device(device)
    if device.type in ("cuda", "npu"):
        getattr(torch, device.type).synchronize(device)


def memory(device):
    import psutil

    info = psutil.Process().memory_info()
    result = {
        "cpu_rss_bytes": info.rss,
        "cpu_peak_bytes": getattr(info, "peak_wset", None),
    }
    if result["cpu_peak_bytes"] is None:
        try:
            import resource
            import sys

            result["cpu_peak_bytes"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            pass
    if torch.device(device).type in ("cuda", "npu"):
        backend = getattr(torch, torch.device(device).type)
        result["device_peak_allocated_bytes"] = backend.max_memory_allocated(device)
        result["device_peak_reserved_bytes"] = backend.max_memory_reserved(device)
    return result


def qdq(x, proxy, role="weights"):
    if proxy == "INT8":
        return _row_int8(
            x
        )  # Preserve zero-row nonfinite behavior, not an invented epsilon.
    args = (MXFP4 if proxy == "MXFP4" else MXFP8 if proxy == "MXFP8" else {})[role]
    if x.ndim != 2 or x.shape[-1] % 32:
        raise ValueError(
            "MXFP requires 2D inputs and input-channel width divisible by 32; no implicit padding"  # noqa: E501
        )
    blocks = x.unflatten(-1, (-1, 32))
    scales, zp = calculate_qparams(blocks.amin(-1), blocks.amax(-1), args)
    return fake_quantize(x, scales, zp, args)


def metrics(reference, output, eps=1e-30):
    # Cast BEFORE subtract/square; CPU FP64 accumulation also works for NPU outputs.
    r, y = reference.detach().cpu().double(), output.detach().cpu().double()
    finite = bool(torch.isfinite(r).all() and torch.isfinite(y).all())
    if not finite:
        return dict(
            mse=math.inf, nmse=math.inf, nrmse=math.inf, max_abs=math.inf, finite=False
        )
    error = (r - y).square().sum().item()
    nmse = error / (r.square().sum().item() + eps)
    return dict(
        mse=error / r.numel(),
        nmse=nmse,
        nrmse=math.sqrt(nmse),
        max_abs=(r - y).abs().max().item(),
        finite=True,
    )


class Objective:
    """Keep originals/reference on CPU; one weight slab and token batch on accelerator.

    GEMM uses storage dtype. Reference GEMM is cached once per split. CPU FP64
    moments intentionally replace production storage-dtype error arithmetic.
    """

    def __init__(self, x, w, device="cpu", batch=128, rows=256):
        self.x, self.w = x.cpu(), w.cpu()
        self.device, self.batch, self.rows = device, batch, rows
        self.amax, self.wmax = x.abs().amax(0), w.abs().amax(0)
        self.reference = torch.empty((len(x), len(w)), dtype=x.dtype)
        for j in range(0, len(w), rows):
            wd = w[j : j + rows].to(device)
            for i in range(0, len(x), batch):
                self.reference[i : i + batch, j : j + rows] = (
                    x[i : i + batch].to(device) @ wd.T
                ).cpu()

    @torch.no_grad()
    def assess(self, scale, proxy=None, activation=False):
        error = energy = 0.0
        max_abs = 0.0
        for j in range(0, len(self.w), self.rows):
            wd = (self.w[j : j + self.rows] * scale).to(self.device)
            if proxy:
                wd = qdq(wd, proxy)
            for i in range(0, len(self.x), self.batch):
                xd = (self.x[i : i + self.batch] / scale).to(self.device)
                if proxy and activation:
                    xd = qdq(xd, proxy, "input_activations")
                y = (xd @ wd.T).cpu().double()
                r = self.reference[i : i + self.batch, j : j + self.rows].double()
                if not torch.isfinite(y).all():
                    return dict(
                        mse=math.inf,
                        nmse=math.inf,
                        nrmse=math.inf,
                        max_abs=math.inf,
                        finite=False,
                    )
                error += (r - y).square().sum().item()
                energy += r.square().sum().item()
                max_abs = max(max_abs, (r - y).abs().max().item())
        nmse = error / (energy + 1e-30)
        return dict(
            mse=error / self.reference.numel(),
            nmse=nmse,
            nrmse=math.sqrt(nmse),
            max_abs=max_abs,
            finite=math.isfinite(nmse),
        )

    def applied_scale(self, alpha, beta):
        return smooth_scale(self.amax, self.wmax, alpha, beta)

    def search(self, proxy, activation):
        candidates = []

        def evaluate(a, b):
            sync(self.device)
            start = time.perf_counter()
            scale = self.amax.pow(a) * self.wmax.pow(-b)
            loss = self.assess(scale, proxy, activation)["nrmse"]
            sync(self.device)
            candidates.append(
                dict(
                    stage="alpha" if len(candidates) < 21 else "beta",
                    alpha=a,
                    beta=b,
                    loss=loss,
                    seconds=time.perf_counter() - start,
                )
            )
            return loss

        sync(self.device)
        start = time.perf_counter()
        result = _search_candidates(evaluate)
        sync(self.device)
        elapsed = time.perf_counter() - start
        scale = self.applied_scale(result.alpha, result.beta)
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Selected applied scale is invalid")
        return dict(
            proxy=proxy,
            objective="W+A" if activation else "W-only",
            alpha=result.alpha,
            beta=result.beta,
            loss=result.loss,
            scale=scale,
            candidates=candidates,
            search_seconds=elapsed,
            candidate_mean_seconds=sum(c["seconds"] for c in candidates) / 42,
            selected_raw_loss=self.assess(
                self.amax.pow(result.alpha) * self.wmax.pow(-result.beta),
                proxy,
                activation,
            )["nrmse"],
            applied_calibration_loss=self.assess(scale, proxy, activation)["nrmse"],
            boundary=result.alpha in (0, 1) or result.beta in (0, 1),
            memory=memory(self.device),
        )


def validate_cache(cache):
    if cache.get("schema") != 1 or not cache.get("groups"):
        raise ValueError("Expected norm-linear capture schema 1")
    for key, group in cache["groups"].items():
        x, v, w = (group[k] for k in ("calibration", "validation", "weight"))
        if any(
            t.ndim != 2 or not t.numel() or not torch.isfinite(t).all()
            for t in (x, v, w)
        ):
            raise ValueError(f"{key}: invalid tensors")
        if (
            len({t.shape[1] for t in (x, v, w)}) != 1
            or len({t.dtype for t in (x, v, w)}) != 1
        ):
            raise ValueError("Inconsistent mapping widths/dtypes")
        ids, vids = group["calibration_ids"], group["validation_ids"]
        if (
            len(ids) != len(x)
            or len(vids) != len(v)
            or len(set(ids)) != len(ids)
            or len(set(vids)) != len(vids)
            or set(ids) & set(vids)
        ):
            raise ValueError("Duplicate/overlapping or missing token identities")
        if sum(group["consumer_rows"]) != len(w) or len(group["consumers"]) != len(
            group["consumer_rows"]
        ):
            raise ValueError("Incomplete consumer group")
    return cache
