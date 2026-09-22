# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline official GLM fixture with real calibration pipelines and export."""

import json
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import repack_moe
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier


@pytest.mark.parametrize("pipeline", ["basic", "sequential"])
@torch.no_grad()
def test_official_glm_calibration_export(tmp_path, monkeypatch, pipeline):
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
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=[QuaRotModifier(block_size=32), modifier],
        pipeline=pipeline,
        sequential_targets=["GlmMoeDsaDecoderLayer"],
    )
    actual = model(tokens).logits
    torch.testing.assert_close(actual, before, atol=2e-6, rtol=2e-5)
    assert len(modifier.diagnostics) == 6
    assert not modifier._hooks and not modifier._cache
    assert model.config.flex_smooth_config["status"] == "applied"
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
