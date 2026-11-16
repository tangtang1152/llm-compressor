import copy
import json
import subprocess
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.core import State
from llmcompressor.modifiers.transform.quarot import QuaRotModifier
from llmcompressor.modifiers.transform.quarot.mappings import build_glm_plan

from .no_hidden_fixture import TinyGLM
from .no_hidden_run import LOCAL, inventory, prepare, verify
from .test_experiment import tmp_path  # noqa: F401


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def apply(model, spaces=None, cls=QuaRotModifier):
    mod = cls(
        block_size=32,
        precision=torch.float32,
        **({"enabled_rotation_spaces": spaces} if cls is QuaRotModifier else {}),
    )
    state = State(model=model)
    mod.on_initialize(state)
    matrices = dict(mod._matrices)
    mod.on_calibration_start(state, None)
    return mod, state, matrices


def test_plan_and_full_legacy_regression():
    torch.manual_seed(1234)
    model = TinyGLM()
    full = build_glm_plan(model)
    local = build_glm_plan(model, tuple(LOCAL))
    assert local.fusions == full.fusions and local.embedding is None
    assert local.operations == tuple(op for op in full.operations if op.space != "rot")
    assert [s.name for s in local.spaces] == LOCAL
    assert all(op.space != "rot" for op in local.operations)
    for spaces in ((), ("unknown",), ("rot_uv", "rot_uv")):
        with pytest.raises(ValueError):
            build_glm_plan(model, spaces)
    # Execute pinned old Full implementation as regression oracle, not a rewrite.
    source = subprocess.check_output(
        [
            "git",
            "show",
            "89d814b2:src/llmcompressor/modifiers/transform/quarot/base.py",
        ],
        text=True,
    )
    module = types.ModuleType("llmcompressor.modifiers.transform.quarot._legacy_test")
    module.__package__ = "llmcompressor.modifiers.transform.quarot"
    sys.modules[module.__name__] = module
    exec(compile(source, "historical_base.py", "exec"), module.__dict__)
    reference = copy.deepcopy(model)
    apply(reference, cls=module.QuaRotModifier)
    apply(model)
    for name, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name


@pytest.mark.parametrize("spaces", [(s,) for s in LOCAL] + [tuple(LOCAL)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_compensation(spaces, dtype):
    torch.manual_seed(1234)
    model = TinyGLM().to(dtype)
    reference = copy.deepcopy(model)
    embedding = model.model.embed_tokens.weight.clone()
    mod, state, matrices = apply(model, spaces)
    assert set(matrices) == set(spaces) and "rot" not in matrices
    assert torch.equal(embedding, model.model.embed_tokens.weight)
    assert model.config.quarot_config["enabled_rotation_spaces"] == [
        s for s in LOCAL if s in spaces
    ]
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        a, at = reference(ids)
        b, bt = model(ids)

    def close(x, y):
        nmse = (x.double() - y.double()).square().sum() / x.double().square().sum()
        assert nmse < (3e-4 if dtype == torch.bfloat16 else 1e-10), nmse.item()

    close(a, b)
    for x, y in zip(at, bt):
        for name in (
            "q",
            "k",
            "probabilities",
            "hidden",
            "index_scores",
            "router_logits",
        ):
            if x.get(name) is not None:
                close(x[name], y[name])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mod.on_calibration_start(state, None)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="already"):
        QuaRotModifier(enabled_rotation_spaces=spaces).on_initialize(state)


def artifact(root, spaces):
    root.mkdir()
    meta = dict(
        status="applied",
        seed=1234,
        block_size=32,
        precision="float32",
        topology="glm_mla",
        enabled_rotation_spaces=spaces,
    )
    config = dict(
        quarot_config=meta,
        quantization_config={"fixture": True},
        flex_smooth_config={"fixture": True},
    )
    (root / "config.json").write_text(json.dumps(config))
    save_file(
        {"weight": torch.ones(2, 32, dtype=torch.bfloat16)},
        str(root / "model.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}})
    )


def test_artifact_and_driver(tmp_path):  # noqa: F811
    from argparse import Namespace

    full, candidate = tmp_path / "full", tmp_path / "candidate"
    artifact(full, ["rot", *LOCAL])
    artifact(candidate, LOCAL)
    assert verify(full, candidate)["tensor_count"] == 1
    original = tmp_path / "original"
    original.mkdir()
    (original / "config.json").write_text("{}")
    data = tmp_path / "data.json"
    data.write_text('["fixture"]')
    args = Namespace(
        checkpoint=original,
        baseline_full=full,
        run_dir=tmp_path / "run",
        output=tmp_path / "new",
        offload_dir=tmp_path / "offload",
        dataset=data,
        max_memory=None,
    )
    prepare(args)
    code = (args.run_dir / "oneshot_profile.py").read_text()
    compile(code, "snapshot", "exec")
    assert "enabled_rotation_spaces" in code and "recipe[1].max_tokens = None" in code
    assert "--samples 42" in (args.run_dir / "quantize.sh").read_text()
    with pytest.raises(FileExistsError):
        prepare(args)
    index = candidate / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"missing": "model.safetensors"}}))
    with pytest.raises(ValueError, match="keys mismatch"):
        inventory(candidate)


def test_actual_glm_bf16_no_hidden():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    from llmcompressor.modeling.moe.linearize import linearize_moe

    torch.manual_seed(1234)
    config = GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
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
        q_lora_rank=64,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        index_head_dim=64,
        index_n_heads=2,
        index_topk=4,
    )
    config._attn_implementation = "eager"
    config.dtype = torch.bfloat16
    model = GlmMoeDsaForCausalLM(config).bfloat16().eval()
    linearize_moe(model)
    reference = copy.deepcopy(model)
    apply(model, tuple(LOCAL))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        a = reference(ids, use_cache=False).logits
        b = model(ids, use_cache=False).logits
    nmse = (a.double() - b.double()).square().sum() / a.double().square().sum()
    assert nmse < 3e-4, nmse.item()
