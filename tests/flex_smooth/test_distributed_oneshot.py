# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The real three-modifier sequential path, with CT shared backing on two ranks."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache import DistributedCPUCache
from compressed_tensors.offload.module import remove_module_offload
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import (
    CompressedTensorsConfig,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
    PreTrainedTokenizerFast,
)

from examples.glm52_precision.oneshot_profile import Profile, quantized_tiles
from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.transform import QuaRotModifier
from llmcompressor.recipe import Recipe
from tests.flex_smooth.oneshot_trace import OneshotTrace
from tests.flex_smooth.test_transformers import _LinearizedGlm


def _fixture(directory):
    config = GlmMoeDsaConfig.from_pretrained(directory / "input", local_files_only=True)
    config._name_or_path = str(directory / "input")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).float().eval()
        linearize_moe(model)
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    return model, tokenizer


def _snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _run(model, tokenizer, tokens):
    recipe = Recipe.create_instance(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/glm52_precision/mixed_mxfp.yaml"
        )
    ).modifiers
    recipe[0].precision = torch.float32
    recipe[1].max_tokens = None
    loader = DataLoader(
        [{"input_ids": row, "attention_mask": torch.ones_like(row)} for row in tokens],
        batch_size=1,
    )
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=recipe,
        pipeline="sequential",
        sequential_targets_per_subgraph=1,
        sequential_targets=[r"re:.*\.input_layernorm$", "ExpertMLP", "GlmMoeDsaMLP"],
        propagate_error=True,
        moe_calibrate_all_experts=False,
    )
    return recipe[1].diagnostics


@torch.no_grad()
def _worker(rank, store, directory, kind):
    torch.set_num_threads(1)
    directory = Path(directory)
    expected = torch.load(directory / "reference.pt", weights_only=True)
    (directory / "offload").mkdir(exist_ok=True)
    tokens = expected["tokens"]
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    model, tokenizer = _fixture(directory)
    for module in model.modules():
        remove_module_offload(module, onload_tensors=True)
        offload_module(
            module,
            onload_device="cpu",
            offload_device=kind,
            **({"offload_dir": str(directory / "offload")} if kind == "disk" else {}),
        )
    rotation_calls = []
    with pytest.MonkeyPatch.context() as patch:
        trace = OneshotTrace(patch)
        rotate = QuaRotModifier._rotate
        start = QuaRotModifier.on_calibration_start

        def record_rotate(modifier, *args, **kwargs):
            rotation_calls.append(1)
            return rotate(modifier, *args, **kwargs)

        def check_rotated(modifier, state, event, **kwargs):
            start(modifier, state, event, **kwargs)
            actual = _snapshot(state.model)
            for name, value in expected["rotated"].items():
                if name.endswith((".weight", ".bias")):
                    torch.testing.assert_close(
                        actual[name], value, atol=0, rtol=0, msg=name
                    )

        patch.setattr(QuaRotModifier, "_rotate", record_rotate)
        patch.setattr(QuaRotModifier, "on_calibration_start", check_rotated)
        # Stand in for private accelerator residents using CPU copies of weights.
        # Quantization scale tensors retain native CPU behavior (shared backing).
        onload = DistributedCPUCache.onload

        def separate_weight_resident(cache, value):
            result = onload(cache, value)
            return (
                result.clone()
                if any(
                    cache.offloaded_values.get(key) is value
                    for key in ("weight", "bias")
                )
                else result
            )

        if kind == "cpu":
            patch.setattr(DistributedCPUCache, "onload", separate_weight_resident)
        profile = Profile("cpu")
        with profile.instrument():
            diagnostics = _run(model, tokenizer, tokens[rank : rank + 1])
        assert profile.stages["flex_search"]["calls"] == 6
        assert profile.stages["flex_apply"]["calls"] == 6
        assert profile.stages["calibration_forward"]["calls"] == 7
        assert profile.stages["propagation"]["calls"] == 7
        assert bool(rotation_calls) == (rank == 0)
        assert trace.events.index("quarot.applied") < trace.events.index("flex.start")
        assert trace.events.index("flex.start") < trace.events.index("quant.start")
        assert trace.captures > 0
        # OneshotTrace asserts post-Flex resident weights inside weight observe.
        observed = [None, None]
        dist.all_gather_object(observed, sorted(trace.observed))
        assert len(set(observed[0]) | set(observed[1])) >= 6
        actual = _snapshot(model)
        for name, value in expected["final"].items():
            torch.testing.assert_close(actual[name], value, atol=2e-6, rtol=2e-5)
        for name, value in diagnostics.items():
            baseline = expected["diagnostics"][name]
            assert (value["alpha"], value["beta"]) == (
                baseline["alpha"],
                baseline["beta"],
            )
            torch.testing.assert_close(
                value["scale"], baseline["scale"], atol=2e-6, rtol=2e-5
            )
            assert value["tokens_used"] == tokens.numel()
        with disable_offloading():
            logits = model(tokens).logits
        tiles = quantized_tiles(model)
        torch.testing.assert_close(logits, expected["logits"], atol=2e-6, rtol=2e-5)
        rotary = model.model.rotary_emb.inv_freq.detach().clone()
        dist.barrier()
        model.save_pretrained(directory / "compressed", save_compressed=True)
        torch.testing.assert_close(
            model.model.rotary_emb.inv_freq, rotary, atol=0, rtol=0
        )
    dist.destroy_process_group()
    restored, loading = _LinearizedGlm.from_pretrained(
        directory / "compressed",
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        output_loading_info=True,
    )
    assert not any(loading.values()), loading
    restored.float().eval()
    actual_tiles = quantized_tiles(restored, dequantized=True)
    for name in tiles:
        for key in ("weight", "scale"):
            torch.testing.assert_close(
                actual_tiles[name][key], tiles[name][key], atol=0, rtol=0
            )
    torch.testing.assert_close(restored(tokens).logits, logits, atol=0, rtol=0)
    report = {
        "rank": rank,
        "backing": kind,
        "real_weights": False,
        "rotation_calls": len(rotation_calls),
        "events": trace.events,
        "global_flex_parameters_match_single_rank": True,
        "post_flex_observer": True,
        "shared_weights_match_single_rank": True,
        "compressed_reload_exact": True,
        "profile": profile.report,
    }
    (directory / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("kind", ["cpu", "disk"])
@torch.no_grad()
def test_distributed_sequential_roundtrip(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    GlmMoeDsaConfig(
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
    ).save_pretrained(tmp_path / "input")
    model, tokenizer = _fixture(tmp_path)
    tokens = torch.tensor([[1, 7, 4, 5, 9], [9, 3, 5, 1, 7]])
    rotated = {}
    start = QuaRotModifier.on_calibration_start

    def record_rotated(modifier, state, event, **kwargs):
        start(modifier, state, event, **kwargs)
        rotated.update(_snapshot(state.model))

    with monkeypatch.context() as patch:
        patch.setattr(QuaRotModifier, "on_calibration_start", record_rotated)
        diagnostics = _run(model, tokenizer, tokens)
    with disable_offloading():
        logits = model(tokens).logits
    torch.save(
        {
            "rotated": rotated,
            "final": _snapshot(model),
            "diagnostics": diagnostics,
            "tokens": tokens,
            "logits": logits,
        },
        tmp_path / "reference.pt",
    )
    context = mp.spawn(
        _worker,
        args=((tmp_path / "store").as_uri(), str(tmp_path), kind),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Tiny distributed oneshot exceeded 180s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"distributed-oneshot-{kind}.json").write_text(
            json.dumps(reports, indent=2)
        )
