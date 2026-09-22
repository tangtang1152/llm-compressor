# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline official GLM fixture with real calibration pipelines and export."""

import json
import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import repack_moe
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.observers import MinMaxObserver
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import DisableQuantization


@pytest.mark.parametrize("pipeline", ["basic", "sequential"])
@pytest.mark.parametrize("mixed", [False, True])
@torch.no_grad()
def test_official_glm_calibration_export(tmp_path, monkeypatch, pipeline, mixed):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = GlmMoeDsaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        index_topk=2,
        index_head_dim=64,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    config.save_pretrained(tmp_path / "input-config")
    config._name_or_path = str(tmp_path / "input-config")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokens = torch.tensor([[1, 7, 4, 5, 9], [9, 3, 5, 1, 7]])
    before = model(tokens).logits
    loader = DataLoader(
        [{"input_ids": row, "attention_mask": torch.ones_like(row)} for row in tokens],
        batch_size=1,
    )
    modifier = FlexSmoothModifier(max_tokens=32)
    recipe = [QuaRotModifier(block_size=32), modifier]
    if mixed:
        recipe = Recipe.create_instance(
            str(
                Path(__file__).resolve().parents[2]
                / "examples/glm52_precision/mixed_mxfp.yaml"
            )
        ).modifiers
        modifier = recipe[1]
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=recipe,
        pipeline=pipeline,
        sequential_targets=["GlmMoeDsaDecoderLayer"],
    )
    if mixed:
        _check_mixed_quantization(model, recipe[-1], pipeline)
    with DisableQuantization(model):
        actual = model(tokens).logits
    torch.testing.assert_close(actual, before, atol=2e-6, rtol=2e-5)
    assert len(modifier.diagnostics) == 6
    assert not modifier._hooks and not modifier._cache
    assert model.config.flex_smooth_config["status"] == "applied"
    if mixed:
        # Quantized checkpoint/kernel compatibility is a separate server gate.
        assert torch.isfinite(model(tokens).logits).all()
        return
    repack_moe(model)
    model.save_pretrained(tmp_path / "output", save_compressed=False)
    restored = GlmMoeDsaForCausalLM.from_pretrained(
        tmp_path / "output", local_files_only=True, attn_implementation="eager"
    ).eval()
    torch.testing.assert_close(restored(tokens).logits, actual, atol=2e-6, rtol=2e-5)
    assert restored.config.flex_smooth_config == model.config.flex_smooth_config
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        report = {
            "fixture_seed": 42,
            "pipeline": pipeline,
            "real_weights": False,
            "logits_max_abs_error": (actual - before).abs().max().item(),
            "reload_logits_max_abs_error": (restored(tokens).logits - actual)
            .abs()
            .max()
            .item(),
            "results": model.config.flex_smooth_config["results"],
        }
        Path(output, f"official-glm-{pipeline}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )


def _check_mixed_quantization(model, modifier, pipeline):
    checked = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        expected = None
        if ".self_attn." in name and leaf in {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
            "wq_b",
        }:
            expected = 8
        if ".mlp." in name and leaf in {"gate_proj", "up_proj", "down_proj"}:
            expected = 4 if ".experts." in name else 8
        scheme = getattr(module, "quantization_scheme", None)
        if expected is None:
            assert scheme is None, name
            continue
        assert scheme is not None, name
        assert scheme.weights.num_bits == expected
        assert scheme.input_activations.num_bits == expected
        assert scheme.weights.group_size == scheme.input_activations.group_size == 32
        assert not scheme.weights.dynamic and scheme.input_activations.dynamic
        assert module.quantization_status == QuantizationStatus.FROZEN
        assert not hasattr(module, "weight_observer")
        assert not hasattr(module, "input_scale")
        # Re-observe the final transformed weights: stale pre-transform scales fail.
        observer = MinMaxObserver("weight", scheme.weights.model_copy(deep=True))
        expected_scale = observer(module.weight).get_qparams()["scale"]
        torch.testing.assert_close(module.weight_scale, expected_scale, atol=0, rtol=0)
        # Check actual wrapped execution against an explicit dynamic input Q/DQ path.
        x = torch.linspace(-2, 2, module.in_features).repeat(2, 1)
        x[1] *= 8
        scale, zp = compute_dynamic_scales_and_zp(x, scheme.input_activations, module)
        assert not torch.equal(scale[0], scale[1])
        xq = fake_quantize(x, scale, zp, scheme.input_activations)
        wq = fake_quantize(module.weight, module.weight_scale, None, scheme.weights)
        torch.testing.assert_close(
            module(x), torch.nn.functional.linear(xq, wq, module.bias)
        )
        checked[name] = expected
    assert set(checked.values()) == {4, 8}
    assert any("shared_experts" in name for name in checked)
    assert any("indexer.wq_b" in name for name in checked)
    assert not modifier._calibration_hooks
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"mixed-mxfp-{pipeline}.json").write_text(
            json.dumps(
                {
                    "real_weights": False,
                    "pipeline": pipeline,
                    "targets": checked,
                    "post_transform_weight_scales_exact": True,
                    "dynamic_input_execution_checked": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
