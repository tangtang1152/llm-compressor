# ruff: noqa: F811
"""E1.3 regression and synthetic end-to-end checks; no real checkpoint."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import transform
from .e13 import comparisons, flat_gamma, variants
from .test_experiment import fixture_checkpoint, tmp_path  # noqa: F401


def test_flat_gamma_sign_zero_and_rms():
    gamma = torch.arange(-31, 33).bfloat16()
    flat, info = flat_gamma(gamma)
    expected = gamma.float().reshape(-1, 32).square().mean(-1).sqrt()
    torch.testing.assert_close(flat.reshape(-1, 32).square().mean(-1).sqrt(), expected)
    assert torch.equal(flat < 0, gamma < 0)
    assert (flat[gamma == 0] > 0).all()
    assert info["zero_count"] == 1 and info["negative_count"] == 31
    assert flat.dtype == torch.float32
    zero, info = flat_gamma(torch.zeros(64).bfloat16())
    assert torch.equal(zero, torch.zeros(64)) and info["all_zero_groups"] == 2
    with pytest.raises(ValueError):
        flat_gamma(torch.zeros(33))
    with pytest.raises(ValueError):
        flat_gamma(torch.full((32,), float("nan")))


def test_variants_use_actual_transforms():
    torch.manual_seed(1)
    weight = torch.randn(32, 64).bfloat16()
    gamma = torch.tensor([-2.0, 2.0] * 32).bfloat16()
    flat, _ = flat_gamma(gamma)
    values = dict(variants(weight, gamma, flat))
    assert torch.equal(values["A"], weight)
    for stage, rotated in [("B", False), ("C", True)]:
        for projection in ("gate_proj", "up_proj"):
            assert torch.equal(
                values[stage], transform(weight, gamma, projection, rotated)
            )
    assert torch.equal(values["B"], values["D"])
    assert torch.equal(values["C"], values["E"])
    delta = comparisons(dict(A=1.0, B=2.0, C=1.0, D=1.0, E=0.75))
    assert delta["B_vs_A_increase_percent"] == 100
    assert delta["rotation_gain_drop_pp"] == 25
    assert comparisons(dict.fromkeys("ABCDE", 0.0))["rotation_gain_drop_pp"] is None


def test_e13_complete_and_history_guard(tmp_path, monkeypatch):
    from .e13 import main
    from .run import main as e1

    fixture_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["num_hidden_layers"] = 4
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    extra = {}
    for key, shard in list(index["weight_map"].items()):
        layer = int(key.split(".")[2])
        new = key.replace(f"model.layers.{layer}.", f"model.layers.{layer+2}.")
        with safe_open(tmp_path / shard, framework="pt") as f:
            extra[new] = f.get_tensor(key)
        index["weight_map"][new] = "extra.safetensors"
    save_file(extra, str(tmp_path / "extra.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    prior = tmp_path / "prior"
    out = tmp_path / "e13"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(prior),
            "--layers",
            "0",
            "1",
            "2",
            "3",
            "--experts-per-layer",
            "3",
            "--e0-hidden",
            "64",
            "--e0-intermediate",
            "32",
        ],
    )
    e1()
    original = (prior / "results.json").read_bytes()
    args = [
        "e13",
        "--checkpoint",
        str(tmp_path),
        "--reuse-e1",
        str(prior / "results.json"),
        "--layers",
        "0",
        "1",
        "2",
        "3",
        "--output",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    result = json.loads((out / "results.json").read_text())
    assert result["status"] == "complete" and result["data_kind"] == "SYNTHETIC"
    assert len(result["rows"]) == 120 and len(result["paired"]) == 24
    assert len(result["layer_metrics"]) == 40 and len(result["layer_comparisons"]) == 8
    assert (prior / "results.json").read_bytes() == original
    assert len((out / "matrices.csv").read_text().splitlines()) == 121
    assert not list(out.glob("*.png"))
    with pytest.raises(FileExistsError):
        main()
    # E1.2 uses e1_rows rather than rows; preserve bytes for source provenance.
    from .diagnose import reuse
    from .e13 import HistorySource
    from .run import Checkpoint, provenance

    adapted = json.loads(original)
    adapted["e1_rows"] = adapted.pop("rows")
    path = tmp_path / "e12.json"
    path.write_text(json.dumps(adapted))
    _, selections, _ = reuse(
        [HistorySource(path)], Checkpoint(tmp_path, [], 0, 1234), provenance()
    )
    assert all(len(ids) == 3 for ids in selections.values())
    del adapted["quantization"]
    path.write_text(json.dumps(adapted))
    with pytest.raises(ValueError, match="incomplete historical schema"):
        HistorySource(path).read_text()

    bad = tmp_path / "bad-selection.json"
    bad.write_text(json.dumps({"0": [0, 1]}))
    monkeypatch.setattr(
        "sys.argv",
        args[:-1] + [str(tmp_path / "failed"), "--expert-selection", str(bad)],
    )
    with pytest.raises(ValueError, match="exactly 3"):
        main()
    assert (
        json.loads((tmp_path / "failed/results.json").read_text())["status"] == "failed"
    )
    history = json.loads(original)
    history["rows"][0]["B_weight_nmse"] += 0.1
    (prior / "results.json").write_text(json.dumps(history))
    monkeypatch.setattr("sys.argv", args[:-1] + [str(tmp_path / "mismatch")])
    with pytest.raises(ValueError, match="historical NMSE mismatch"):
        main()
    assert (
        json.loads((tmp_path / "mismatch/results.json").read_text())["status"]
        == "failed"
    )
