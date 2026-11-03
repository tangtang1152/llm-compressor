import json

import pytest
import torch
from safetensors.torch import save_file

from .core import e0, errors, metrics, qdq, rotation, transform
from .run import Checkpoint


def test_e0():
    e0()


def test_orientation_and_fusion():
    torch.manual_seed(7)
    w = torch.randn(64, 128)
    gain = torch.rand(128) + 0.5
    q = rotation(128).to_dense()
    torch.testing.assert_close(transform(w, gain, "gate_proj", True), (w * gain) @ q)
    torch.testing.assert_close(transform(w.T, gain, "down_proj", True), q.T @ w.T)
    assert errors((w * gain) @ q, (w * gain) @ q.T)["nmse"] > 0.1
    with pytest.raises(ValueError):
        transform(w, gain[:64], "gate_proj", True)


def test_qdq_known_grid_zero_and_observer():
    x = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 4).reshape(1, 32)
    torch.testing.assert_close(qdq(x), x)
    torch.testing.assert_close(qdq(torch.zeros(2, 32)), torch.zeros(2, 32))
    torch.manual_seed(11)
    w = torch.randn(64, 128).bfloat16()
    # Independent grouping through CT dynamic helper must match static observer.
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_scheme import MXFP4
    from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

    args = MXFP4["input_activations"].model_copy(deep=True)
    s, z = compute_dynamic_scales_and_zp(w, args, torch.nn.Identity())
    torch.testing.assert_close(qdq(w), fake_quantize(w, s, z, args))
    with pytest.raises(ValueError):
        qdq(w[:, :33])


def test_metrics():
    r = torch.ones(2, 32)
    a = r.clone()
    a[0, 0] = 0
    m = metrics(r, a)
    assert m["weight_nmse"] == 1 / 64
    assert m["block_relative_mse_mean"] == 1 / 64
    assert m["top_relative_mse_mean"] == 0.5
    assert m["zero_fraction"] == 1 / 64


def fixture_checkpoint(path):
    torch.manual_seed(20261103)
    cfg = dict(
        _experiment_synthetic=True,
        num_hidden_layers=2,
        n_routed_experts=3,
        hidden_size=64,
        moe_intermediate_size=32,
        first_k_dense_replace=0,
        hidden_act="silu",
    )
    (path / "config.json").write_text(json.dumps(cfg))
    tensors = {}
    for layer in range(2):
        tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.ones(
            64
        ).bfloat16()
        for e in range(3):
            for p in ("gate_proj", "up_proj", "down_proj"):
                tensors[Checkpoint.key(layer, e, p)] = torch.randn(
                    (64, 32) if p == "down_proj" else (32, 64)
                ).bfloat16()
    # Two shards verify index-directed loading, without loading whole shard.
    names = list(tensors)
    index = {}
    for i in range(2):
        subset = {k: tensors[k] for k in names[i::2]}
        shard = f"weights-{i}.safetensors"
        save_file(subset, str(path / shard))
        index.update({k: shard for k in subset})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )


def test_selective_checkpoint(tmp_path):
    fixture_checkpoint(tmp_path)
    cp = Checkpoint(tmp_path, [0, 1], 2, 1234)
    assert cp.selection == Checkpoint(tmp_path, [0, 1], 2, 1234).selection
    for layer, ids in cp.selection.items():
        gain = cp.load(f"model.layers.{layer}.post_attention_layernorm.weight")
        for e in ids:
            for p in ("gate_proj", "up_proj", "down_proj"):
                w = cp.load(cp.key(layer, e, p))
                for rotated in (False, True):
                    t = transform(w, gain, p, rotated)
                    assert metrics(t, qdq(t))["weight_nmse"] >= 0
    with pytest.raises(ValueError):
        Checkpoint(tmp_path, [2], 2, 1234)
    with pytest.raises(ValueError, match="memory guard"):
        Checkpoint(tmp_path, [0], 2, 1234, max_elements=100)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    del index["weight_map"][cp.key(0, 0, "gate_proj")]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing explicit"):
        Checkpoint(tmp_path, [0], 1, 1234, experts=[0])


@pytest.fixture
def tmp_path():
    # Windows sandbox rejects pytest's mode-0700 temporary directories.
    from pathlib import Path
    from uuid import uuid4

    path = Path(__file__).resolve().parent / ".test-data" / uuid4().hex
    path.mkdir(parents=True)
    return path


def test_runner_outputs(tmp_path, monkeypatch):
    from .run import main

    fixture_checkpoint(tmp_path)
    out = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--checkpoint",
            str(tmp_path),
            "--layers",
            "0",
            "1",
            "--experts-per-layer",
            "3",
            "--output",
            str(out),
            "--e0-hidden",
            "128",
            "--e0-intermediate",
            "64",
        ],
    )
    main()
    result = json.loads((out / "results.json").read_text())
    assert result["status"] == "complete"
    assert "SYNTHETIC" in result["data_kind"]
    assert len(result["rows"]) == 18
    assert len((out / "metrics.csv").read_text().splitlines()) == 19
    assert "not measured" in (out / "report.md").read_text()


def test_replay_modes():
    from .core import expert

    torch.manual_seed(123)
    weights = [
        torch.randn(32, 64).bfloat16(),
        torch.randn(32, 64).bfloat16(),
        torch.randn(64, 32).bfloat16(),
    ]
    x = torch.randn(3, 64)
    for mode in ("float", "W4A16", "W4A4"):
        y = expert(x, weights, mode)
        assert y.shape == (3, 64) and torch.isfinite(y).all()
