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
    assert len(list(out.glob("groups-*.npz"))) == 0
    assert not (out / "layer_trends.png").exists()
    assert not (out / "group_error_distribution.png").exists()
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


def test_detail_modes_invariant(tmp_path, monkeypatch):
    import numpy as np

    from .diagnose import detail_selection, main

    fixture_checkpoint(tmp_path)
    runs = []
    for mode, files in [("0", 0), ("2", 8), ("all", 12)]:
        out = tmp_path / f"mode-{mode}"
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
                "--group-detail-experts-per-layer",
                mode,
            ],
        )
        main()
        data = json.loads((out / "diagnostics.json").read_text())
        assert data["status"] == "complete"
        assert len(list(out.glob("groups-*.npz"))) == files
        for path in out.glob("groups-*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                assert archive["Raw_energy"].size == 64
                assert np.isfinite(archive["Rotated_relative_mse"]).all()
        runs.append(data)
    for name in (
        "matrices",
        "gamma",
        "comparisons",
        "e1_rows",
        "group_correlations",
        "matrix_correlations",
    ):
        assert runs[0][name] == runs[1][name] == runs[2][name]
    assert detail_selection([5, 2, 8], 3, 2) == detail_selection([8, 5, 2], 3, 2)


def test_failure_and_preflight(tmp_path, monkeypatch):
    from . import diagnose

    fixture_checkpoint(tmp_path)

    def args(out, *extra):
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
                *extra,
            ],
        )

    out = tmp_path / "preflight"
    args(out, "--preflight-only")
    diagnose.main()
    assert (
        json.loads((out / "diagnostics.json").read_text())["status"]
        == "preflight_passed"
    )
    assert not list(out.glob("groups-*"))
    out = tmp_path / "failure"
    args(out)
    original = diagnose.qdq
    calls = 0

    def fail_later(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("injected failure")
        return original(*a, **kw)

    monkeypatch.setattr(diagnose, "qdq", fail_later)
    with pytest.raises(RuntimeError, match="injected"):
        diagnose.main()
    assert json.loads((out / "diagnostics.json").read_text())["status"] == "failed"
    assert "failed" in (out / "summary.md").read_text()
    args(tmp_path / "bad", "--group-detail-experts-per-layer", "-1")
    with pytest.raises(SystemExit):
        diagnose.main()
    with pytest.raises(ValueError, match="nonfinite"):
        diagnose.corr([1, float("nan")], [1, 2])
    assert diagnose.corr([], [])["pearson"] is None
    assert diagnose.corr([1, 1, 1], [1, 2, 3])["pearson"] is None
    with pytest.raises(ValueError, match="empty"):
        diagnose.summary(torch.tensor([]))


def test_reuse_strict(tmp_path):
    from .diagnose import reuse
    from .run import provenance

    fixture_checkpoint(tmp_path)
    cp = Checkpoint(tmp_path, [0], 1, 1234)
    import hashlib

    prov = provenance()
    data = dict(
        status="complete",
        model_config=cp.config,
        index_sha256=hashlib.sha256(cp.index_path.read_bytes()).hexdigest(),
        rotation=dict(block_size=32, seed=1234, precision="float32", shifted=False),
        quantization=dict(
            format="MXFP4 E2M1/E8M0",
            group_size=32,
            axis=-1,
            weight="static minmax",
            activation="dynamic minmax",
        ),
        provenance=prov,
        selection={"0": [0]},
        rows=[],
    )
    path = tmp_path / "prior.json"
    for mutation, match in [
        ("rows", "missing"),
        ("version", "fingerprint"),
        ("source", "fingerprint"),
        ("quant", "incompatible"),
    ]:
        copy = json.loads(json.dumps(data))
        if mutation == "version":
            copy["provenance"]["versions"]["torch"] = "wrong"
        if mutation == "source":
            k = next(iter(copy["provenance"]["implementation_sha256"]))
            copy["provenance"]["implementation_sha256"][k] = "wrong"
        if mutation == "quant":
            copy["quantization"]["weight"] = "other"
        path.write_text(json.dumps(copy))
        with pytest.raises(ValueError, match=match):
            reuse([path], cp, prov)
