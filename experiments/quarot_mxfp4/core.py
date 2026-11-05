"""Isolated routed-expert transforms and CT MXFP4 measurement primitives."""

import math

import torch
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_scheme import MXFP4
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from torch import nn
from torch.nn import functional as F

from llmcompressor.modeling import fuse_norm_linears
from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)
from llmcompressor.observers.min_max import MemorylessMinMaxObserver


def rotation(hidden, seed=1234):
    return make_hadamard_rotation(hidden, block_size=32, seed=seed, dtype=torch.float32)


@torch.no_grad()
def transform(weight, gain, projection, rotated=False, seed=1234):
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(projection)
    hidden = gain.numel()
    axis = 0 if projection == "down_proj" else 1
    if weight.ndim != 2 or weight.shape[axis] != hidden:
        raise ValueError("weight/gain shape mismatch")
    if projection != "down_proj":
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
        linear.weight = nn.Parameter(weight.clone(), requires_grad=False)
        norm = nn.Module()
        norm.weight = nn.Parameter(gain.clone(), requires_grad=False)
        fuse_norm_linears(norm, [linear], precision=torch.float32)
        weight = linear.weight.detach()
    else:
        weight = weight.clone()
    if rotated:
        weight = rotate_axis(
            weight, rotation(hidden, seed), axis=axis, precision=torch.float32
        )
    return weight


@torch.no_grad()
def qdq(value, activation=False, *, return_params=False):
    """Historical CT QDQ; optionally return observer qparams for diagnostics."""
    if value.ndim != 2 or value.shape[-1] % 32 or not torch.isfinite(value).all():
        raise ValueError("expected finite matrix with input dimension divisible by 32")
    args = MXFP4["input_activations" if activation else "weights"].model_copy(deep=True)
    if activation:
        # GROUP dynamic scales must retain both token and group axes. The static
        # observer reduces its observation axis and is not the CT dynamic path.
        scale, zero_point = compute_dynamic_scales_and_zp(value, args, nn.Identity())
        params = dict(scale=scale, zero_point=zero_point)
    else:
        obs = MemorylessMinMaxObserver("weight", args)
        params = obs(value).get_qparams()
    result = fake_quantize(value, args=args, **params)
    return (result, params) if return_params else result


def errors(reference, actual):
    r, a = reference.double(), actual.double()
    d = a - r
    return dict(
        max_abs=d.abs().max().item(),
        mean_abs=d.abs().mean().item(),
        nmse=(d.square().sum() / (r.square().sum() + 1e-30)).item(),
    )


def percentile(value, fraction):
    flat = value.flatten()
    return flat.kthvalue(max(1, math.ceil(flat.numel() * fraction))).values.item()


def metrics(reference, actual):
    """Actual input-axis groups; mean of ratios differs from matrix NMSE."""
    r, a = reference.double().reshape(-1, 32), actual.double().reshape(-1, 32)
    e = (a - r).square()
    b = e.sum(-1) / (r.square().sum(-1) + 1e-30)
    i = r.abs().argmax(-1, keepdim=True)  # first maximum on ties
    t = e.gather(-1, i) / (r.gather(-1, i).square() + 1e-30)
    return dict(
        weight_nmse=(e.sum() / (r.square().sum() + 1e-30)).item(),
        block_relative_mse_mean=b.mean().item(),
        block_relative_mse_p95=percentile(b, 0.95),
        top_relative_mse_mean=t.mean().item(),
        abs_mean=r.abs().mean().item(),
        abs_rms=r.square().mean().sqrt().item(),
        abs_max=r.abs().max().item(),
        abs_p50=percentile(r.abs(), 0.5),
        abs_p99=percentile(r.abs(), 0.99),
        zero_fraction=(a == 0).double().mean().item(),
    )


def expert(x, weights, mode="float"):
    """Minimal replay interface: x is already normed in this branch's coordinates."""
    if mode not in ("float", "W4A16", "W16A4", "W4A4"):
        raise ValueError(mode)

    def linear(x, w):
        return F.linear(
            qdq(x, True) if mode in ("W16A4", "W4A4") else x,
            (qdq(w) if mode in ("W4A16", "W4A4") else w).to(x.dtype),
        )

    gate, up, down = weights
    return linear(F.silu(linear(x, gate)) * linear(x, up), down)


def inverse_hidden(x, rot):
    return (x.reshape(-1, 32) @ rot.block.T.to(x)).reshape(x.shape)


@torch.no_grad()
def e0(hidden=128, intermediate=64, seed=1234):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(8, hidden, generator=rng)
    gain = (torch.rand(hidden, generator=rng) + 0.5).bfloat16()
    original = [
        (torch.randn(*shape, generator=rng) / shape[1] ** 0.5).bfloat16()
        for shape in [
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        ]
    ]
    names = ("gate_proj", "up_proj", "down_proj")
    rot = rotation(hidden, seed)
    torch.testing.assert_close(
        rot.block.T @ rot.block, torch.eye(32), atol=2e-6, rtol=2e-6
    )
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    xr = rot.apply(x)
    normed_rotated = xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normed_rotated, rot.apply(normed), atol=2e-6, rtol=2e-5)
    ref = expert(normed * gain.float(), [w.float() for w in original])
    report = {}
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16_writeback")]:
        b = [
            transform(w.to(dtype), gain.to(dtype), n).float()
            for w, n in zip(original, names)
        ]
        c = [
            transform(w.to(dtype), gain.to(dtype), n, True, seed).float()
            for w, n in zip(original, names)
        ]
        yb = expert(normed, b)
        yc = inverse_hidden(expert(rot.apply(normed), c), rot)
        report[label] = dict(
            b_vs_original=errors(ref, yb),
            c_vs_original=errors(ref, yc),
            c_vs_b=errors(yb, yc),
        )
        if dtype == torch.float32:
            torch.testing.assert_close(yb, ref, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(yc, ref, atol=3e-5, rtol=3e-4)
            assert report[label]["c_vs_b"]["nmse"] < 1e-10
        else:
            assert report[label]["c_vs_original"]["nmse"] < 3e-4
    return report
