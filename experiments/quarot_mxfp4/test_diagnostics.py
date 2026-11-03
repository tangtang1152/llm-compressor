# ruff: noqa: F811
"""Focused regression tests for attribution and optional checkpoint MTP."""

import json

import pytest
import torch

from .core import qdq
from .diagnose import groups
from .run import Checkpoint
from .test_experiment import fixture_checkpoint, tmp_path  # noqa: F401


@pytest.mark.parametrize(
    "extra,missing,valid",
    [(None, None, True), (2, None, True), (3, None, False), (2, 0, False)],
)
def test_optional_mtp(tmp_path, extra, missing, valid):
    fixture_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors.index.json"
    data = json.loads(path.read_text())
    if missing is not None:
        data["weight_map"] = {
            k: v
            for k, v in data["weight_map"].items()
            if not k.startswith(f"model.layers.{missing}.")
        }
    if extra is not None:
        data["weight_map"][f"model.layers.{extra}.mtp.weight"] = "unused.safetensors"
    path.write_text(json.dumps(data))
    if valid:
        Checkpoint(tmp_path, [0], 1, 1234)
        with pytest.raises(ValueError, match="not a routed layer"):
            Checkpoint(tmp_path, [2], 1, 1234)
    else:
        with pytest.raises(ValueError, match="with optional MTP layer 2"):
            Checkpoint(tmp_path, [], 0, 1234)


def test_group_stats_qdq_compatibility():
    w = torch.ones(2, 32).bfloat16()
    w[0, 0] = 8
    q, params = qdq(w, return_params=True)
    assert torch.equal(q, qdq(w))
    g = groups(w, q, params["scale"])
    assert g["max_rms"][1] == 1
    assert g["max_other_rms"][0] == 8
    torch.testing.assert_close(g["sse"].sum(), (q.double() - w.double()).square().sum())
    assert "relative_mse" not in groups(w)


def test_diagnostics_reuse_and_outputs(tmp_path, monkeypatch):
    from .diagnose import main
    from .run import main as e1

    fixture_checkpoint(tmp_path)
    prior = tmp_path / "prior"
    out = tmp_path / "diagnostics"
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
            "--experts-per-layer",
            "2",
            "--e0-hidden",
            "64",
            "--e0-intermediate",
            "32",
        ],
    )
    e1()
    original = (prior / "results.json").read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(out),
            "--layers",
            "0",
            "1",
            "--experts-per-layer",
            "3",
            "--reuse-e1",
            str(prior / "results.json"),
        ],
    )
    main()
    data = json.loads((out / "diagnostics.json").read_text())
    assert data["status"] == "complete" and data["data_kind"] == "SYNTHETIC"
    assert len(data["selection"]["0"]) == 2 and len(data["selection"]["1"]) == 3
    assert len(data["e1_rows"]) == 15
    assert all(
        "relative_mse_mean" not in r for r in data["matrices"] if r["stage"] == "Raw"
    )
    assert (prior / "results.json").read_bytes() == original
    assert len(list(out.glob("groups-*.csv.gz"))) == 10
    assert (out / "layer_trends.png").stat().st_size > 1000
    assert (out / "group_error_distribution.png").stat().st_size > 1000
    # Existing destination is rejected, so previous results cannot be overwritten.
    with pytest.raises(FileExistsError):
        main()
    changed = json.loads(original)
    changed["rotation"]["seed"] = 42
    (prior / "results.json").write_text(json.dumps(changed))
    from .diagnose import reuse
    from .run import provenance

    with pytest.raises(ValueError, match="incompatible prior"):
        reuse([prior / "results.json"], Checkpoint(tmp_path, [], 0, 1234), provenance())
