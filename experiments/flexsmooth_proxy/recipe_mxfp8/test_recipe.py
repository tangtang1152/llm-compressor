import copy
from types import SimpleNamespace

import pytest
import torch

from llmcompressor.modifiers.transform.flex_smooth.mappings import (
    apply_scales,
    glm_mappings,
)
from llmcompressor.modifiers.transform.flex_smooth.search import ov_scales

from ..core import Objective, metrics
from ..distributed_capture import capture_split, merge_shards, partition
from ..test_distributed_capture import tiny_inputs, tiny_model
from ..test_offload import scratch
from .objective import RecipeObjective, select_group
from .run import ab, parity


def test_checkpoint_audit_and_formal_evaluation():
    import json

    from safetensors.torch import save_file

    from .run import audit, complete

    names = [
        "model.layers.6.self_attn." + s
        for s in (
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "indexer.wk",
            "indexer.weights_proj",
        )
    ]
    native = {
        n + ".weight": torch.randn(4, 32).to(
            torch.float32 if n.endswith("weights_proj") else torch.bfloat16
        )
        for n in names
    }
    path = scratch()
    save_file(native, str(path / "tiny.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map={k: "tiny.safetensors" for k in native}))
    )
    x = torch.randn(8, 32).bfloat16().float()
    g = dict(
        weight=torch.cat(list(native.values())),
        consumers=names,
        consumer_rows=[4] * 4,
        calibration=x,
        validation=x.flip(0),
        calibration_ids=[f"c:{i}" for i in range(8)],
        validation_ids=[f"v:{i}" for i in range(8)],
    )
    cache = dict(
        schema=1, kind="real", seed=42, groups={"model.layers.6.input_layernorm": g}
    )
    torch.save(cache, path / "cache.pt")
    groups = audit(cache, path)
    args = SimpleNamespace(
        old_search=None,
        cache=path / "cache.pt",
        device="cpu",
        batch=4,
        rows=4,
        cache_mib=1,
    )
    complete(groups, cache, args, path)
    result = json.loads((path / "results.json").read_text())["groups"][
        "model.layers.6.input_layernorm"
    ]
    assert (
        result["consumers"] == names[:2] and result["compensation_consumers"] == names
    )
    assert set(result["metrics"]) == {"No Smooth", "W-only", "W+A"}
    assert (path / "mxfp8_e2_completed.csv").exists()
    g["weight"][0, 0] += 1
    with pytest.raises(ValueError, match="Checkpoint/cache mismatch"):
        audit(cache, path)


def test_qa_reuse_without_search(monkeypatch):
    from ..run import search, synthetic
    from .run import complete, reuse_qa

    cache = synthetic(tokens=8, dtype=torch.bfloat16)
    g = next(iter(cache["groups"].values()))
    g["consumers"] = [
        "model.layers.6.self_attn.q_b_proj",
        "model.layers.6.self_attn.indexer.wq_b",
    ]
    name = "model.layers.6.self_attn.q_a_layernorm"
    cache["groups"] = {name: g}
    args = SimpleNamespace(
        device="cpu", batch=4, rows=16, cache_mib=1, proxies=["MXFP8"]
    )
    old = search(cache, args)
    selected = select_group(name, g, {n: "torch.bfloat16" for n in g["consumers"]})
    assert reuse_qa(name, selected, cache, old) == old["groups"][name]["runs"]
    path = scratch()
    torch.save(cache, path / "cache.pt")
    torch.save(old, path / "search.pt")
    args.cache = path / "cache.pt"
    args.old_search = path / "search.pt"

    def forbidden(*args):
        raise AssertionError("QA search must not rerun")

    monkeypatch.setattr(RecipeObjective, "search", forbidden)
    complete({name: selected}, cache, args, path)
    corrupted = copy.deepcopy(old)
    corrupted["reference_policy"] = "changed"
    with pytest.raises(ValueError, match="reference/loss"):
        reuse_qa(name, selected, cache, corrupted)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(42)


@pytest.mark.parametrize("geometry", [None, dict(heads=2, value_dim=16, key_dim=16)])
@pytest.mark.parametrize("activation", [False, True])
def test_full_search_parity(geometry, activation):
    x, w = torch.randn(8, 32).bfloat16(), torch.randn(48, 32).bfloat16()
    baseline = RecipeObjective(
        x, w, batch=4, rows=16, optimized=False, geometry=geometry
    )
    optimized = RecipeObjective(x, w, batch=4, rows=16, geometry=geometry)
    a, b = baseline.search("MXFP8", activation), optimized.search("MXFP8", activation)
    assert len(a["candidates"]) == 42 and parity(a, b)
    if geometry:
        expected = ov_scales(baseline.amax, baseline.wmax, a["alpha"], a["beta"], 2, 2)[
            0
        ]
        assert torch.equal(a["scale"], expected)
    else:
        assert parity(a, Objective(x, w, batch=4, rows=16).search("MXFP8", activation))
    optimized.cache_mib = 1e-9
    with pytest.raises(ValueError, match="Activation cache"):
        optimized.assess(torch.ones(32), "MXFP8", True)


def test_mixed_cache_excludes_fp32_consumers():
    names = [
        "a.q_a_proj",
        "a.kv_a_proj_with_mqa",
        "a.indexer.wk",
        "a.indexer.weights_proj",
    ]
    x = torch.randn(8, 32).bfloat16().float()
    w = torch.randn(16, 32).bfloat16().float()
    w[12:] = torch.randn(4, 32)  # FP32 exception must never enter selected objective.
    g = dict(
        consumers=names,
        consumer_rows=[4] * 4,
        weight=w,
        calibration=x,
        validation=x.clone(),
    )
    d = {n: "torch.bfloat16" for n in names}
    d[names[-1]] = "torch.float32"
    result = select_group("model.layers.6.input_layernorm", g, d)
    assert result["consumers"] == names[:2]
    assert result["compensation_consumers"] == names
    assert result["weight"].dtype == torch.bfloat16 and len(result["weight"]) == 8
    assert torch.equal(g["weight"], w)
    d[names[1]] = "torch.float32"
    with pytest.raises(ValueError, match="mixed target dtype"):
        select_group("model.layers.6.input_layernorm", g, d)


@pytest.mark.parametrize("general_scale", [False, True])
def test_native_attention_ov_value_compensation(general_scale):
    model = tiny_model()
    mapping = next(
        m
        for m in glm_mappings(model, ("ov",))
        if m.source.startswith("model.layers.0.")
    )
    attn = model.model.layers[0].self_attn
    x = torch.randn(1, 8, 64).bfloat16()
    pos = torch.arange(8).unsqueeze(0)
    kwargs = dict(
        position_embeddings=model.model.rotary_emb(x, pos),
        position_ids=pos,
        attention_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool).tril(),
        past_key_values=None,
    )
    before = attn(x, **kwargs)
    original = attn.kv_b_proj.weight.detach().clone()
    scale = 2.0 ** torch.arange(mapping.heads * mapping.value_dim).remainder(3).sub(1)
    if general_scale:
        scale = ov_scales(
            torch.rand_like(scale) + 0.1,
            attn.o_proj.weight.abs().amax(0),
            0.55,
            0.7,
            mapping.heads,
            mapping.heads,
        )[0]
    apply_scales(model, mapping, scale)
    after = attn(x, **kwargs)
    shape = (mapping.heads, mapping.key_dim + mapping.value_dim, -1)
    assert torch.equal(
        original.reshape(shape)[:, : mapping.key_dim],
        attn.kv_b_proj.weight.reshape(shape)[:, : mapping.key_dim],
    )
    torch.testing.assert_close(
        attn.kv_b_proj.weight.reshape(shape)[:, mapping.key_dim :],
        original.reshape(shape)[:, mapping.key_dim :]
        * scale.to(original.dtype)
        .reshape(mapping.heads, mapping.value_dim, 1)
        .reciprocal(),
        rtol=0,
        atol=0,
    )
    assert torch.equal(before[2], after[2])
    assert metrics(before[0], after[0])["nmse"] < (1e-4 if general_scale else 1e-5)


@pytest.mark.parametrize("suffix", ["input_layernorm", "q_a_layernorm"])
def test_full_norm_compensation_includes_unquantized_fp32(suffix):
    model = tiny_model()
    model.model.layers[0].self_attn.indexer.weights_proj.float()
    mapping = next(
        m
        for m in glm_mappings(model, ("norm-linear",))
        if m.source.startswith("model.layers.0.") and m.source.endswith("." + suffix)
    )
    norm = model.get_submodule(mapping.source)
    x = torch.randn(1, 8, len(norm.weight)).bfloat16()
    value = norm(x)
    modules = [model.get_submodule(n) for n in mapping.consumers]
    dtypes = [m.weight.dtype for m in modules]
    before = [m(value.to(m.weight.dtype)) for m in modules]
    scale = 2.0 ** torch.arange(len(norm.weight)).remainder(3).sub(1)
    apply_scales(model, mapping, scale)
    after = [m(norm(x).to(m.weight.dtype)) for m in modules]
    assert dtypes == [m.weight.dtype for m in modules]
    assert (
        model.model.layers[0].self_attn.indexer.weights_proj.weight.dtype
        == torch.float32
    )
    assert all(metrics(a, b)["nmse"] < 1e-6 for a, b in zip(before, after))


def test_ov_capture_geometry_and_original_inputs():
    from llmcompressor.modeling.moe.linearize import linearize_moe

    model = tiny_model()
    linearize_moe(model)
    splits, source = tiny_inputs()
    source.update(
        mapping_suffixes=["self_attn.kv_b_proj"], rank_plan=partition(splits, 16, 8, 2)
    )
    shards = [
        capture_split(
            model,
            splits[r],
            r,
            source,
            [1],
            ["kv_b_proj"],
            16,
            8,
            "cpu",
            mapping_kind="ov",
        )
        for r in (0, 1)
    ]
    cache = merge_shards(shards)
    g = cache["groups"]["model.layers.1.self_attn.kv_b_proj"]
    assert g["geometry"] == dict(heads=2, value_dim=16, key_dim=16)
    assert g["consumers"] == ["model.layers.1.self_attn.o_proj"]
    assert g["calibration"].shape == (16, 32)


def test_ab_outputs_and_parity_rejects_change():
    g = dict(
        calibration=torch.randn(8, 32).bfloat16(), weight=torch.randn(48, 32).bfloat16()
    )
    out = scratch()
    torch.save({}, out / "source.pt")
    args = SimpleNamespace(
        device="cpu", batch=4, rows=16, cache_mib=1, repeats=1, cache=out / "source.pt"
    )
    ab(g, "test", args, out)
    assert (out / "p1_candidate_parity.csv").exists()
    o = RecipeObjective(g["calibration"], g["weight"])
    a = o.search("MXFP8", True)
    b = copy.deepcopy(a)
    b["candidates"][0]["loss"] += 1e-10
    assert not parity(a, b)
