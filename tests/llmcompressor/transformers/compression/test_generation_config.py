# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Invalid generation metadata must fail before costly compression/save."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import GenerationConfig

from examples.glm52_precision import oneshot_profile as driver
from llmcompressor.transformers.compression import compressed_tensors_utils as saving


def greedy_config(top_p=0.95):
    config = GenerationConfig(do_sample=False)
    config.top_p = top_p
    return config


def test_driver_sanitizes_inactive_top_p_and_records_original(tmp_path):
    config = greedy_config()
    with pytest.raises(ValueError, match="top_p"):
        config.validate(strict=True)
    report = {}
    driver.prepare_generation_config(SimpleNamespace(generation_config=config), report)
    assert config.do_sample is False
    assert config.top_p is None
    config.validate(strict=True)
    config.save_pretrained(tmp_path)  # The real Transformers save validation.
    assert report["generation_config"] == {
        "adjustments": [
            {
                "field": "top_p",
                "original": 0.95,
                "value": None,
                "action": "unset_non_sampling_top_p",
                "do_sample": False,
            }
        ],
        "load_validation": "passed",
    }


@pytest.mark.parametrize("sampling,top_p", [(False, None), (False, 1.0), (True, 0.95)])
def test_driver_leaves_valid_generation_config_unchanged(sampling, top_p):
    config = GenerationConfig(do_sample=sampling, top_p=top_p)
    before, report = config.to_dict(), {}
    driver.prepare_generation_config(SimpleNamespace(generation_config=config), report)
    assert config.to_dict() == before
    assert report["generation_config"]["adjustments"] == []
    assert report["generation_config"]["load_validation"] == "passed"


def test_driver_rejects_other_invalid_flags_before_calibration(tmp_path, monkeypatch):
    config = greedy_config()
    config.temperature = 0.5  # Outside the deliberately narrow sanitation.
    model = SimpleNamespace(generation_config=config)
    model.eval = lambda: model
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneshot_profile.py",
            "--model",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--device",
            "cpu",
            "--dataset",
            str(tmp_path / "unused.json"),
        ],
    )
    monkeypatch.setattr(driver, "load_context", nullcontext)
    monkeypatch.setattr(
        driver,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: model),
    )
    tokenizer, oneshot = Mock(), Mock()
    monkeypatch.setattr(driver, "AutoTokenizer", tokenizer)
    monkeypatch.setattr(driver, "oneshot", oneshot)
    with pytest.raises(ValueError, match="temperature"):
        driver.main()
    oneshot.assert_not_called()
    assert config.do_sample is False and config.temperature == 0.5
    report = json.loads((tmp_path / "output-profile/rank-0.json").read_text())
    assert report["status"] == "failed"
    assert report["generation_config"]["load_validation"] == "failed"
    assert report["generation_config"]["adjustments"][0]["original"] == 0.95


class SavingModel:
    def __init__(self, config):
        self.generation_config = config
        self.saved = False

    def can_generate(self):
        return True

    def save_pretrained(self, *args, **kwargs):
        self.saved = True


def test_wrapper_rejects_invalid_config_without_mutation_or_compression(monkeypatch):
    model = SavingModel(greedy_config())
    before = model.generation_config.to_dict()
    factory = Mock()
    monkeypatch.setattr(saving.ModelCompressor, "from_pretrained_model", factory)
    saving.modify_save_pretrained(model)
    with pytest.raises(ValueError, match="top_p"):
        model.save_pretrained("unused", save_compressed=True)
    factory.assert_not_called()
    assert not model.saved
    assert model.generation_config.to_dict() == before


def test_wrapper_validates_strictly_before_compression(monkeypatch):
    model = SavingModel(greedy_config(None))
    events = []
    validate = model.generation_config.validate

    def validation(*, strict):
        assert strict is True
        validate(strict=strict)
        events.append("strict_validation")

    def compress(*args, **kwargs):
        events.append("compression")
        raise RuntimeError("stop after observing compression entry")

    monkeypatch.setattr(model.generation_config, "validate", validation)
    compressor = SimpleNamespace(compress_model=compress)
    monkeypatch.setattr(
        saving.ModelCompressor, "from_pretrained_model", lambda *a, **k: compressor
    )
    saving.modify_save_pretrained(model)
    with pytest.raises(RuntimeError, match="stop after"):
        model.save_pretrained("unused", save_compressed=True)
    assert events == ["strict_validation", "compression"]
    assert not model.saved
