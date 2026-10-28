# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU checks of the exact server layer A/B harness; no real checkpoint required."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from tests.flex_smooth.test_layer_l4 import fixture as fixture
from tools.glm52_layer_l4 import forward_kwargs
from tools.glm52_layer_l4_capture import capture_layer
from tools.glm52_partition_ab import (
    Measurements,
    compare_trials,
    file_hash,
    pipeline_module,
    run_trial,
)


def arguments(output, strategy, **overrides):
    return Namespace(
        **(
            dict(
                command="run",
                output=str(output),
                strategy=strategy,
                tiny=True,
                model_dir=None,
                activation_cache=None,
                synthetic_inputs=False,
                layer=1,
                max_weight_gib=1,
                samples=2,
                tokens=5,
                dtype="float32",
                device="cpu",
                offload="cpu",
                scratch=None,
                threads=1,
                seed=42,
                pairs=1,
                left=None,
                right=None,
            )
            | overrides
        )
    )


@pytest.mark.parametrize("offload", ["cpu", "disk"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_tiny_ab_lifecycle_and_counts(tmp_path, monkeypatch, offload, dtype):
    # Simulate an accelerator-visible host. The harness pins only the pipeline hook.
    detected = lambda: torch.device("cuda")  # noqa: E731
    monkeypatch.setattr(pipeline_module, "get_main_device", detected)
    reports = {}
    for strategy in ("layer", "expert"):
        reports[strategy] = run_trial(
            arguments(tmp_path / strategy, strategy, offload=offload, dtype=dtype)
        )
        assert pipeline_module.get_main_device is detected
        report = reports[strategy]
        assert report["status"] == "completed"
        assert report["flex"]["status"] == "applied"
        assert report["quarot"]["status"] == "applied"
        measure = report["measurement"]
        graphs = 2 if strategy == "layer" else 5
        assert len(measure["partitions"]) == graphs
        calls = measure["hot_calls"]
        for phase in ("calibration", "propagation"):
            assert calls[f"sequential_execution/forward.{phase}"]["calls"] == 2 * graphs
        for name in ("update", "delete"):
            assert calls[f"sequential_execution/intermediates.{name}"]["calls"] == (
                2 * (graphs - 1)
            )
        assert calls["sequential_execution/intermediates.fetch"]["calls"] == 4 * graphs
        assert len(measure["expert_calls"]) == 4
        assert all(item["calls"] == 2 for item in measure["expert_calls"].values())
        assert calls["sequential_execution/flex_search"]["calls"] == 3
        cls = "CPUCache" if offload == "cpu" else "DiskCache"
        assert any(key.endswith(f"{cls}.onload") for key in calls)
        assert not list((tmp_path / strategy).glob("glm52-ab-*"))
    compared = compare_trials(tmp_path / "layer", tmp_path / "expert")
    assert compared["comparable"]
    assert compared["exact_numerical_match"], compared
    assert not compare_trials(tmp_path / "layer", tmp_path / "layer")["comparable"]
    path = tmp_path / "expert/report.json"
    changed = json.loads(path.read_text())
    changed["input_tensor_hashes"][0]["hidden_states"] = "different input"
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not compare_trials(tmp_path / "layer", tmp_path / "expert")["comparable"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@torch.no_grad()
def test_selected_layer_streaming_and_source_unchanged(
    fixture,  # noqa: F811 -- imported pytest fixture
    tmp_path,
    cached,
    dtype,
):
    model, root, values = fixture
    cache = tmp_path / "layer.safetensors"
    if cached:
        x, indices = model.model.layers[0](
            values["hidden_states"], **forward_kwargs(values)
        )
        values = {**values, "hidden_states": x, "prev_topk_indices": indices}
        with capture_layer(
            model,
            model_dir=root,
            layer=1,
            output=cache,
            input_kind="synthetic",
            input_description="A/B test",
        ):
            model.model.layers[1](values["hidden_states"], **forward_kwargs(values))
    before = file_hash(root / "model.safetensors")
    reports = []
    for strategy in ("layer", "expert"):
        reports.append(
            run_trial(
                arguments(
                    tmp_path / strategy,
                    strategy,
                    tiny=False,
                    model_dir=str(root),
                    offload="disk",
                    dtype=dtype,
                    synthetic_inputs=not cached,
                    activation_cache=[str(cache)] if cached else None,
                )
            )
        )
    assert file_hash(root / "model.safetensors") == before
    assert reports[0]["input"] == reports[1]["input"]
    assert len(reports[0]["batch_shapes"]) == (1 if cached else 2)
    assert reports[0]["input"]["indexer_type"] == "shared"
    assert compare_trials(tmp_path / "layer", tmp_path / "expert")[
        "exact_numerical_match"
    ]


def test_boundary_sync_only_and_cpu_rss_semantics():
    calls = []
    fake = SimpleNamespace(
        synchronize=lambda device: calls.append("sync"),
        reset_peak_memory_stats=lambda device: calls.append("reset"),
        memory_allocated=lambda device: 10,
        memory_reserved=lambda device: 20,
        max_memory_allocated=lambda device: 30,
        max_memory_reserved=lambda device: 40,
    )
    meter = Measurements("cpu")
    meter.accelerator = fake
    with meter.phase("sequential_execution"):
        for _ in range(1000):
            meter.count("forward.calibration", 0.01)
    assert calls == ["sync", "reset", "sync"]
    phase = meter.report["phases"]["sequential_execution"]
    assert phase["end_memory"]["peak_allocated_bytes"] == 30
    assert phase["max_sampled_rss_bytes"] > 0
    assert meter.report["hot_calls"]["sequential_execution/forward.calibration"] == {
        "calls": 1000,
        "host_seconds": pytest.approx(10),
        "logical_bytes": 0,
    }


def test_failure_report_and_noncomparable_inputs(tmp_path, monkeypatch):
    from tools import glm52_partition_ab as ab

    def fail(*args):
        raise ValueError("bad selected metadata")

    monkeypatch.setattr(ab, "load_selected", fail)
    with pytest.raises(ValueError, match="bad selected metadata"):
        run_trial(arguments(tmp_path / "failed", "layer"))
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed"
    assert "initialize" not in report["measurement"]["phases"]
    assert compare_trials(tmp_path / "failed", tmp_path / "failed") == {
        "comparable": False,
        "reason": "one or both trials did not complete",
    }
