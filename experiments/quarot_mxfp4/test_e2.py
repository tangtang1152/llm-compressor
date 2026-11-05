# ruff: noqa: F811
import json

import pytest
import torch
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm

from . import core
from .e2 import evaluate, replay, run
from .e2_capture import RoutedCapture
from .test_experiment import fixture_checkpoint, tmp_path  # noqa: F401


def test_dynamic_token_group_and_modes(monkeypatch):
    x = torch.tensor([[1.0] * 32 + [8.0] * 32, [64.0] * 32 + [0.125] * 32]).bfloat16()
    q, p = core.qdq(x, True, return_params=True)
    assert p["scale"].shape == (2, 2)
    assert len(torch.unique(p["scale"])) == 4
    assert torch.equal(q[:1], core.qdq(x[:1], True))
    weights = [
        torch.randn(32, 64).bfloat16(),
        torch.randn(32, 64).bfloat16(),
        torch.randn(64, 32).bfloat16(),
    ]
    original = core.qdq
    calls = []

    def spy(value, activation=False, **kw):
        calls.append(activation)
        return original(value, activation, **kw)

    monkeypatch.setattr(core, "qdq", spy)
    for mode, expected in [
        ("float", (0, 0)),
        ("W4A16", (3, 0)),
        ("W16A4", (0, 3)),
        ("W4A4", (3, 3)),
    ]:
        calls.clear()
        core.expert(x, weights, mode)
        assert (calls.count(False), calls.count(True)) == expected


def test_replay_direction_and_down_input(monkeypatch):
    from . import e2

    torch.manual_seed(7)
    z = torch.randn(8, 64)
    g = torch.rand(64) + 0.5
    ws = [torch.randn(32, 64) / 8, torch.randn(32, 64) / 8, torch.randn(64, 32) / 6]
    outputs, _, _ = evaluate(z, z * g, g, ws, torch.float32)
    assert max(r["common_nmse"] for r in outputs if r["mode"] == "FP") < 1e-10
    rot = core.rotation(64)
    assert not torch.allclose(core.inverse_hidden(z, rot), rot.apply(z))
    quant = [core.qdq(w) for w in ws]
    inputs = []
    original = e2.qdq

    def spy(x, activation=False, **kw):
        if activation:
            inputs.append(x.clone())
        return original(x, activation, **kw)

    monkeypatch.setattr(e2, "qdq", spy)
    replay(z, ws, quant, "W4A4", torch.float32)
    expected = torch.nn.functional.silu(original(z, True) @ quant[0].T) * (
        original(z, True) @ quant[1].T
    )
    assert len(inputs) == 3
    torch.testing.assert_close(inputs[2], expected)
    ideal = torch.nn.functional.silu(z @ ws[0].T) * (z @ ws[1].T)
    assert not torch.allclose(inputs[2], ideal)


def make_capture(order=(0, 1)):
    collector = RoutedCapture({0: [0, 1, 2], 1: [0, 1, 2]}, target=128, minimum=128)
    norm = GlmMoeDsaRMSNorm(64).bfloat16()
    batches = []
    for layer in (0, 1):
        for sample in order:
            gen = torch.Generator().manual_seed(10 + sample + layer * 2)
            x = torch.randn(96, 64, generator=gen).bfloat16()
            ids = torch.stack([torch.arange(96) % 3, (torch.arange(96) + 1) % 3], -1)
            weights = torch.full((96, 2), 0.5).bfloat16()
            with torch.no_grad():
                y = norm(x)
            collector.add(
                layer,
                sample,
                norm,
                x,
                y,
                ids,
                weights,
                torch.ones(96, dtype=torch.bool),
            )
            batches.append(
                (
                    layer,
                    sample,
                    norm,
                    x,
                    y,
                    ids,
                    weights,
                    torch.ones(96, dtype=torch.bool),
                )
            )
    return collector, batches


def test_capture_and_e2_end_to_end(tmp_path):
    fixture_checkpoint(tmp_path)
    collector, batches = make_capture()
    reverse, _ = make_capture((1, 0))
    for key in collector.buffers:
        assert [(r[0], r[1], r[2]) for r in collector.buffers[key]] == [
            (r[0], r[1], r[2]) for r in reverse.buffers[key]
        ]
    with pytest.raises(ValueError, match="duplicate"):
        collector.add(*batches[0])
    capture = tmp_path / "capture"
    collector.save(capture, {"kind": "synthetic_test"})
    original = (capture / "capture_manifest.json").read_bytes()
    result = run(tmp_path, capture, tmp_path / "output", synthetic=True)
    assert result["status"] == "complete"
    assert (
        len(result["outputs"]) == 72
        and len(result["weights"]) == 54
        and len(result["activations"]) == 216
    )
    assert all(
        r["common_nmse"] == 0
        for r in result["outputs"]
        if r["branch"] == "A" and r["mode"] == "FP"
    )
    assert max(r["common_nmse"] for r in result["outputs"] if r["mode"] == "FP") < 0.003
    assert (capture / "capture_manifest.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        run(tmp_path, capture, tmp_path / "output", synthetic=True)
    with pytest.raises(ValueError, match="real capture"):
        run(tmp_path, capture, tmp_path / "not-real")
    short = RoutedCapture({0: [0]}, target=128, minimum=128)
    with pytest.raises(ValueError, match="insufficient"):
        short.save(tmp_path / "short", {"kind": "synthetic_test"})
    assert (
        json.loads((tmp_path / "short/capture_manifest.json").read_text())["status"]
        == "incomplete"
    )
    manifest = json.loads(original)
    path = capture / manifest["experts"][0]["file"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, capture, tmp_path / "corrupt", synthetic=True)
    assert (
        json.loads((tmp_path / "corrupt/results.json").read_text())["status"]
        == "failed"
    )
    assert not (tmp_path / "corrupt/summary.md").exists()


def test_data_preflight_no_fallback(tmp_path):
    from .e2_preflight import inspect_data

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == dict(
                truncation=True,
                max_length=2688,
                add_special_tokens=False,
                return_token_type_ids=False,
            )
            return dict(input_ids=[1] * len(text), attention_mask=[1] * len(text))

    path = tmp_path / "data.json"
    path.write_text(json.dumps(["abc", "de"]))
    r = inspect_data(path, Tokenizer(), samples=2)
    assert r["valid_tokens"] == 5 and r["global_sample_ids"] == [0, 1]
    with pytest.raises(ValueError, match="not enough"):
        inspect_data(path, Tokenizer(), samples=3)


def test_actual_tiny_glm_sequential_capture_and_merge(tmp_path):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    from llmcompressor.modeling.moe.linearize import linearize_moe

    from .e2_sequential import collect, merge_shards

    torch.manual_seed(1234)
    config = GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
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
    batches = [
        dict(
            input_ids=torch.randint(1, 128, (1, 12)),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
        )
        for _ in range(2)
    ]
    batches[1]["attention_mask"][0, -2:] = 0
    selection = {1: [0, 1, 2, 3], 2: [0, 1, 2, 3]}
    direct = RoutedCapture(selection, target=8, minimum=1)
    # Direct original model route/norm values provide an independent oracle.
    pending, handles, current = {}, [], [0]

    def norm_hook(layer):
        def hook(module, args, output):
            pending[layer] = (module, args[0].reshape(-1, 64), output.reshape(-1, 64))

        return hook

    def route_hook(layer):
        def hook(module, args, output):
            norm, x, y = pending.pop(layer)
            direct.add(
                layer,
                current[0],
                norm,
                x,
                y,
                output[2],
                output[1],
                batches[current[0]]["attention_mask"].reshape(-1).bool(),
            )

        return hook

    for layer in selection:
        handles.append(
            model.model.layers[layer].post_attention_layernorm.register_forward_hook(
                norm_hook(layer)
            )
        )
        handles.append(
            model.model.layers[layer].mlp.gate.register_forward_hook(route_hook(layer))
        )
    with torch.no_grad():
        for i, batch in enumerate(batches):
            current[0] = i
            model(**batch, use_cache=False)
    for handle in handles:
        handle.remove()

    # Prefix execution must not reach the following decoder or LM head.
    def forbidden(*args):
        raise AssertionError("capture executed beyond last requested layer")

    handles = [
        model.model.layers[3].input_layernorm.register_forward_hook(forbidden),
        model.lm_head.register_forward_hook(forbidden),
    ]
    source = dict(kind="synthetic_test")
    paths = []
    for rank in range(2):
        part = RoutedCapture(selection, target=8, minimum=1)
        report = collect(model, [batches[rank]], [rank], part)
        assert report["verified_dispatches"] == 8
        path = tmp_path / f"rank{rank}"
        part.save(path, source, rank_shard=True)
        paths.append(path)
    for handle in handles:
        handle.remove()
    merged = merge_shards(paths, tmp_path / "merged", source)
    for key, rows in direct.buffers.items():
        assert direct.counts[key] == merged.counts[key]
        for actual, expected in zip(merged.buffers[key], rows):
            assert actual[:3] == expected[:3]
            for x, y in zip(actual[3:], expected[3:]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match="overlapping"):
        merge_shards([paths[0], paths[0]], tmp_path / "badmerge", source)


def test_capture_norm_zero_negative_gamma_and_graph_failure():
    from .e2_sequential import capture_prefix

    norm = GlmMoeDsaRMSNorm(64).bfloat16()
    with torch.no_grad():
        norm.weight[0] = 0
        norm.weight[1] = -2
        x = torch.randn(4, 64).bfloat16()
        y = norm(x)
    collector = RoutedCapture({0: [0]}, target=4, minimum=1)
    args = (
        0,
        1,
        norm,
        x,
        y,
        torch.zeros(4, 1, dtype=torch.int64),
        torch.ones(4, 1),
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
    )
    collector.add(*args)
    assert collector.counts[(0, 0)] == 2
    assert collector.norm_errors[0]["nmse"] == 0
    assert collector.gammas[0][0] == 0 and collector.gammas[0][1] == -2
    with pytest.raises(ValueError, match="call_module"):
        capture_prefix([], {0: [0]})
