"""Synthetic validation of P1/E2 stage contracts and analysis."""

import copy
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ..core import provenance
from ..run import synthetic
from .common import compatibility, git_head, identities, stage
from .e2 import smoke
from .run import token_plan


def scratch():
    root = Path(__file__).parent.parent / ".test-data" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def test_compatibility_and_tokens():
    old = synthetic(8)
    old.update(provenance=provenance(), capture_source_sha256={"capture.py": "known"})
    new = copy.deepcopy(old)
    search = {"provenance": provenance(), "reference_policy": "same"}
    assert compatibility(old, new, search, search)["strictly_comparable"]
    # Installation paths differ but source bytes do not.
    new["provenance"]["sources"] = {
        "/different/" + Path(k).name: v for k, v in old["provenance"]["sources"].items()
    }
    assert compatibility(old, new, search, search)["strictly_comparable"]
    new["groups"][next(iter(new["groups"]))]["validation_ids"][0] = "changed"
    assert not compatibility(old, new, search, search)["strictly_comparable"]
    new = copy.deepcopy(old)
    new["provenance"]["compressed_tensors"] = "other"
    assert not compatibility(old, new, search, search)["strictly_comparable"]
    missing = copy.deepcopy(old)
    missing.pop("capture_source_sha256")
    assert not compatibility(old, missing, search, search)["strictly_comparable"]
    assert token_plan(([(0, "a", [1, 2, 3])], [(1, "b", [4, 5, 6])]), 2, 3) == {
        "calibration": ["a:0", "a:1"],
        "validation": ["b:0", "b:1"],
    }
    with pytest.raises(ValueError, match="Insufficient"):
        token_plan(([], []), 2, 3)
    mismatch = copy.deepcopy(old)
    mismatch["groups"]["extra"] = copy.deepcopy(next(iter(old["groups"].values())))
    mismatch["groups"]["extra"]["calibration_ids"].reverse()
    with pytest.raises(ValueError, match="identities"):
        identities(mismatch)


def test_stage_failure_record():
    root = scratch() / "failed"
    args = SimpleNamespace(stage="test", output=root, expected_commit="wrong")
    with pytest.raises(ValueError, match="commit"):
        with stage(args):
            pass
    import json

    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert (root / "failure.csv").exists()
    assert git_head()


def test_smoke_small():
    rows = smoke(synthetic(8), batch=4, rows=16, tokens=8)
    assert len(rows) == 9 and all(r["e0_pass"] and r["finite"] for r in rows)


def test_e2_tables_and_plots():
    from .e2 import plots, tables

    name = "model.layers.3.input_layernorm"
    runs = []
    entries = []
    for p in ("INT8", "MXFP4", "MXFP8"):
        for o in ("W-only", "W+A"):
            runs.append(dict(proxy=p, objective=o, scale=torch.ones(32)))
            entries.append(
                dict(
                    proxy=p,
                    objective=o,
                    alpha=0.5,
                    beta=0.5,
                    loss=0.1,
                    search_seconds=1.0,
                    W4A4={"nmse": 0.5 if p == "MXFP4" else 1.0},
                    W8A8={"nmse": 0.25 if p == "MXFP8" else 1.0},
                )
            )
    entries.append(
        dict(
            proxy="No Smooth",
            objective="baseline",
            alpha=None,
            beta=None,
            search_seconds=0.0,
            W4A4={"nmse": 1.0},
            W8A8={"nmse": 1.0},
        )
    )
    raw, gains = tables(
        {"groups": {name: {"runs": runs}}},
        {"groups": {name: {"results": entries}}},
        "fixture",
    )
    assert len(raw) == 14 and len(gains) == 4
    assert (
        next(
            r
            for r in gains
            if r["format"] == "W4A4" and r["comparison"] == "matched_proxy"
        )["relative_gain"]
        == 0.5
    )
    path = scratch()
    plots(path, raw, gains)
    assert len(list(path.glob("*.png"))) == 3


def test_p1_no_npu_is_explicitly_blocked(monkeypatch):
    import json

    from . import p1

    monkeypatch.setattr(p1.importlib.util, "find_spec", lambda name: None)
    out = scratch()
    manifest = {}
    p1.run(SimpleNamespace(), out, manifest)
    assert manifest["status"] == "blocked"
    assert json.loads((out / "results.json").read_text())["status"] == "blocked"
    assert (out / "summary.csv").exists()


def test_search_gate_rejects_other_cache(monkeypatch):
    import json

    from . import run

    cache = synthetic(8)
    group = next(iter(cache["groups"].values()))
    cache["groups"] = {
        f"model.layers.{i}.input_layernorm": copy.deepcopy(group)
        for i in (6, 16, 60, 77)
    }
    monkeypatch.setattr(run, "load_cache", lambda path: cache)
    monkeypatch.setattr(run, "file_hash", lambda path: "actual")
    monkeypatch.setattr(run, "search", lambda *a: pytest.fail("must not search"))
    out = scratch()
    (out / "status.json").write_text(
        json.dumps(dict(stage="e2-smoke", status="complete", cache_sha256="different"))
    )
    args = SimpleNamespace(
        stage="e2-search", cache=Path("unused"), smoke=out, device="cpu"
    )
    with pytest.raises(ValueError, match="exact cache"):
        run.execute(args, out, {})
