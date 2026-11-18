"""Exercise the actual sequential graph and CT disk backing on tiny GLM."""

import uuid
from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import dispatch_with_map

from .capture import capture
from .offload import capture_prefix, loader_options


def scratch():
    path = Path(__file__).parent / ".test-data" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_loader_controls(monkeypatch):
    path = scratch()
    options = loader_options(
        torch.bfloat16, path / "offload", {"0": "2GiB", "cpu": "8GiB"}
    )
    assert options["device_map"] == "auto_offload"
    assert options["max_memory"] == {0: "2GiB", "cpu": "8GiB"}
    with pytest.raises(FileExistsError):
        loader_options(torch.bfloat16, path / "offload")
    with pytest.raises(ValueError, match="offload-dir"):
        loader_options(torch.bfloat16, None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single process"):
        loader_options(torch.bfloat16, path / "multi")
    assert not (path / "multi").exists()


@pytest.mark.parametrize(
    "layers,mappings",
    [
        ((1,), ("input_layernorm",)),
        ((0, 1), ("input_layernorm", "q_a_layernorm")),
    ],
)
def test_tiny_glm_disk_sequential_matches_direct(layers, mappings):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    from llmcompressor.modeling.moe.linearize import linearize_moe

    torch.manual_seed(42)
    config = GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        vocab_size=128,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        index_head_dim=32,
        index_n_heads=2,
        index_topk=4,
    )
    config._attn_implementation = "eager"
    config.dtype = torch.bfloat16
    model = GlmMoeDsaForCausalLM(config).bfloat16().eval()
    linearize_moe(model)
    path = scratch()
    splits = (
        [(0, "cal0", list(range(1, 9))), (1, "cal1", list(range(9, 17)))],
        [(2, "val0", list(range(17, 25))), (3, "val1", list(range(25, 33)))],
    )
    options = dict(layers=layers, mappings=mappings, tokens=16, context=8)
    direct = capture(model, splits, path / "direct.pt", **options)
    # Real CT disk-backed parameters/buffers, not mocks of the execution engine.
    from compressed_tensors.offload.module import remove_module_offload

    for module in model.modules():
        remove_module_offload(module, onload_tensors=True)
    (path / "backing").mkdir()
    dispatch_with_map(
        model,
        {name: (torch.device("cpu"), "disk") for name, _ in model.named_modules()},
        offload_dir=str(path / "backing"),
    )

    from compressed_tensors.offload.cache import DiskCache

    assert isinstance(model.model.layers[1].self_attn.q_a_proj._parameters, DiskCache)
    assert list((path / "backing").rglob("*.safetensors"))

    def forbidden(*args):
        raise AssertionError("executed later layer or LM head")

    handles = [
        model.model.layers[2].input_layernorm.register_forward_hook(forbidden),
        model.lm_head.register_forward_hook(forbidden),
    ]
    try:
        sequential = capture(
            model, splits, path / "sequential.pt", sequential=True, **options
        )
    finally:
        for handle in handles:
            handle.remove()
    for name, expected in direct["groups"].items():
        actual = sequential["groups"][name]
        for field in ("weight", "calibration", "validation"):
            torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
        assert actual["calibration_ids"] == expected["calibration_ids"]
    assert all(e["subgraphs"] > 1 for e in sequential["execution"])
    assert not model.model.layers[1].self_attn.q_a_proj._forward_pre_hooks
    with pytest.raises(ValueError, match="call_module"):
        capture_prefix([], [])
