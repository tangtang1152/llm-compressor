"""Numerical and call-count regression without a full search."""

import pytest
import torch

from .benchmark_reuse import CANDIDATES, Sections, assess, selection
from .core import Objective


@pytest.mark.parametrize("proxy", ["INT8", "MXFP4", "MXFP8"])
@pytest.mark.parametrize("activation", [False, True])
def test_reuse_exact_and_counts(proxy, activation):
    gen = torch.Generator().manual_seed(42)
    # Deliberately partial final token batch and output slab.
    x = torch.randn(19, 64, generator=gen).bfloat16()
    w = torch.randn(37, 64, generator=gen).bfloat16()
    obj = Objective(x, w, batch=8, rows=16)
    baseline, optimized = [], []
    for a, b in CANDIDATES:
        scale = obj.amax.pow(a) * obj.wmax.pow(-b)
        expected = obj.assess(scale, proxy, activation)
        for use_cache in (False, True):
            profile = Sections("cpu")
            actual = assess(obj, scale, proxy, activation, use_cache, profile)
            assert actual == expected
            assert profile.counts["activation_scale"] == (3 if use_cache else 9)
            assert profile.counts["activation_qdq"] == (
                (3 if use_cache else 9) if activation else 0
            )
            assert profile.counts["weight_qdq"] == 3
            assert profile.counts["gemm"] == 9
            assert profile.counts["output_to_cpu"] == 9
            assert profile.counts["activation_to_device"] == (3 if use_cache else 9)
        baseline.append(expected["nmse"])
        optimized.append(assess(obj, scale, proxy, activation)["nmse"])
    assert selection(baseline) == selection(optimized)
    assert selection([1, 1, 1]) == [2, 1, 0]
    # Never reuse an earlier candidate's activation cache.
    assert assess(obj, torch.ones(64, dtype=x.dtype), proxy, activation) == obj.assess(
        torch.ones(64, dtype=x.dtype), proxy, activation
    )


def test_budget_and_nonfinite():
    obj = Objective(torch.zeros(4, 32), torch.ones(8, 32), batch=2, rows=4)
    with pytest.raises(ValueError, match="cache needs"):
        assess(obj, torch.ones(32), "INT8", True, cache_mib=1e-6)
    expected = obj.assess(torch.ones(32), "INT8", True)
    assert not expected["finite"]
    assert assess(obj, torch.ones(32), "INT8", True) == expected
