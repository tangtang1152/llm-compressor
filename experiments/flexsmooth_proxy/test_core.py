import math

import pytest
import torch
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.quant_scheme import MXFP4, MXFP8
from compressed_tensors.quantization.utils import calculate_qparams

from experiments.flexsmooth_proxy.core import Objective, metrics, qdq, validate_cache
from experiments.flexsmooth_proxy.run import synthetic
from llmcompressor.modifiers.transform.flex_smooth.search import (
    _row_int8,
    _search_candidates,
    smooth_scale,
)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_e0(dtype):
    g = synthetic(16, dtype=dtype)["groups"]["model.layers.3.input_layernorm"]
    x, w = g["calibration"], g["weight"]
    x[:, 0] = 0
    w[:, 1] = 0
    s = smooth_scale(x.abs().amax(0), w.abs().amax(0), 0.4, 0.7)
    assert torch.isfinite(s).all() and (s > 0).all()
    m = metrics(x @ w.T, (x / s) @ (w * s).T)
    assert m["finite"] and m["nmse"] < 64 * torch.finfo(dtype).eps ** 2


@pytest.mark.parametrize("proxy,scheme", [("MXFP4", MXFP4), ("MXFP8", MXFP8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qdq(proxy, scheme, dtype):
    x = torch.linspace(-8, 8, 64).reshape(2, 32).to(dtype)
    x[0] = 0
    a = scheme["weights"]
    blocks = x.unflatten(-1, (-1, 32))
    s, z = calculate_qparams(blocks.amin(-1), blocks.amax(-1), a)
    explicit = dequantize(quantize(x, s, z, a), s, z, a, dtype=x.dtype)
    torch.testing.assert_close(qdq(x, proxy), explicit)
    assert torch.isfinite(explicit).all()
    assert torch.count_nonzero(explicit[0]) == 0
    with pytest.raises(ValueError):
        qdq(x[:, :31], proxy)


def test_int8_and_objectives():
    g = synthetic(16)["groups"]["model.layers.3.input_layernorm"]
    x, w = g["calibration"], g["weight"]
    s = smooth_scale(x.abs().amax(0), w.abs().amax(0), 0.3, 0.6)
    objective = Objective(x, w, batch=8, rows=16)
    for activation in (False, True):
        xd = qdq(x / s, "INT8") if activation else x / s
        expected = metrics(x @ w.T, xd @ _row_int8(w * s).T)
        assert objective.assess(s, "INT8", activation)["nmse"] == pytest.approx(
            expected["nmse"], rel=1e-5
        )
    assert not torch.isfinite(qdq(torch.zeros(2, 32), "INT8")).all()


def test_grid_ties_and_stability():
    calls = []
    r = _search_candidates(lambda a, b: calls.append((a, b)) or 1.0)
    assert (r.alpha, r.beta) == (1.0, 1.0) and len(calls) == 42
    assert calls[:2] == [(0.0, 1.0), (0.05, 0.95)]
    assert calls[21] == (1.0, 0.0)
    r = torch.full((10, 10), 1e20, dtype=torch.bfloat16)
    assert metrics(r, r)["nmse"] == 0
    assert metrics(torch.zeros(2), torch.zeros(2))["nmse"] == 0
    assert not metrics(torch.zeros(2), torch.tensor([math.inf, 0]))["finite"]


def test_seed_and_cache_contract():
    a, b = synthetic(), synthetic()
    name = next(iter(a["groups"]))
    torch.testing.assert_close(
        a["groups"][name]["calibration"], b["groups"][name]["calibration"]
    )
    validate_cache(a)
    a["groups"][name]["validation_ids"][0] = a["groups"][name]["calibration_ids"][0]
    with pytest.raises(ValueError):
        validate_cache(a)


def test_capture_and_split(monkeypatch):
    import uuid
    from pathlib import Path

    tmp_path = Path(__file__).parent / ".test-data" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True)
    import json
    from types import SimpleNamespace

    from experiments.flexsmooth_proxy import capture as cap
    from llmcompressor.modifiers.transform.flex_smooth.mappings import SmoothMapping

    class Tokenizer:
        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) % 32 for c in text]}

    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(["abcdefgh", "abcdefgh", "ijklmnop", "qrstuvwx", "yz123456"])
    )
    splits = cap.split_records(data, Tokenizer(), 42)
    assert splits == cap.split_records(data, Tokenizer(), 42)
    assert not set(r[1] for r in splits[0]) & set(r[1] for r in splits[1])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(32, 16, bias=False)
            self.b = torch.nn.Linear(32, 8, bias=False)
            self.config = SimpleNamespace()

        def forward(self, input_ids, **kwargs):
            x = torch.nn.functional.one_hot(input_ids, 32).float()
            return self.a(x), self.b(x)

    model = Model().eval()
    mapping = SmoothMapping("norm-linear", "model.layers.3.input_layernorm", ("a", "b"))
    monkeypatch.setattr(cap, "glm_mappings", lambda *args: [mapping])
    cache = cap.capture(
        model,
        splits,
        tmp_path / "cache.pt",
        layers=(3,),
        mappings=("input_layernorm",),
        tokens=8,
    )
    g = cache["groups"][mapping.source]
    assert g["weight"].shape == (24, 32) and g["calibration"].shape == (8, 32)
    assert not model.a._forward_pre_hooks
    with pytest.raises(ValueError, match="Insufficient"):
        cap.capture(
            model,
            splits,
            tmp_path / "bad.pt",
            layers=(3,),
            mappings=("input_layernorm",),
            tokens=1000,
        )
    assert not model.a._forward_pre_hooks and not (tmp_path / "bad.pt").exists()


def test_format_exact_values():
    fp4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 4).reshape(1, 32)
    torch.testing.assert_close(qdq(fp4, "MXFP4"), fp4)
    fp8 = torch.tensor([0, 1, 2, 4, 8, 16, 128, 448] * 4, dtype=torch.float32).reshape(
        1, 32
    )
    torch.testing.assert_close(qdq(fp8, "MXFP8"), fp8)
