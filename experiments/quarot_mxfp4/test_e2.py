# ruff: noqa: F811
import json

import pytest
import torch
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm

from . import core
from .e2 import (
    combine_batches,
    error_stats,
    evaluate,
    replay,
    run,
    select_tokens,
    summarize,
)
from .e2_capture import RoutedCapture, file_hash, read_manifest, read_part
from .test_experiment import fixture_checkpoint, tmp_path  # noqa: F401


@pytest.fixture(autouse=True, scope="module")
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


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
    outputs, _ = evaluate(z, z * g, g, ws, torch.float32)
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


def make_capture(order=(0, 1), length=480, output=None):
    collector = RoutedCapture({0: [0, 1, 2], 1: [0, 1, 2]}, output=output)
    norm = GlmMoeDsaRMSNorm(64).bfloat16()
    batches = []
    for layer in (0, 1):
        for sample in order:
            gen = torch.Generator().manual_seed(10 + sample + layer * 2)
            x = torch.randn(length, 64, generator=gen).bfloat16()
            ids = torch.stack(
                [torch.arange(length) % 3, (torch.arange(length) + 1) % 3], -1
            )
            weights = torch.full((length, 2), 0.5).bfloat16()
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
                torch.ones(length, dtype=torch.bool),
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
                    torch.ones(length, dtype=torch.bool),
                )
            )
    return collector, batches


def archive_rows(root):
    result = {}
    for entry in read_manifest(root)["experts"]:
        for part in entry["parts"]:
            data, positions = read_part(root, part)
            for i, identity in enumerate(positions):
                key = (entry["layer"], entry["expert"], *identity)
                assert key not in result
                result[key] = tuple(
                    data[k][i].clone() for k in ("z", "z_gamma", "router_weight")
                )
    return result


def test_capture_and_e2_end_to_end(tmp_path):
    fixture_checkpoint(tmp_path)
    capture = tmp_path / "capture"
    collector, batches = make_capture(output=capture)
    assert all(
        not chunks for chunks in collector.buffers.values()
    )  # streamed, not resident
    with pytest.raises(ValueError, match="duplicate"):
        collector.add(*batches[0])
    manifest = collector.save(capture, dict(kind="synthetic_test"))
    assert all(e["retained"] == e["available"] == 640 for e in manifest["experts"])
    assert manifest["archive_file_bytes"] == sum(
        p.stat().st_size
        for p in (capture / "captured_activations").glob("*.safetensors")
    )
    original = (capture / "capture_manifest.json").read_bytes()
    reverse, _ = make_capture((1, 0))
    reverse.save(tmp_path / "reverse", dict(kind="synthetic_test"))
    for entry, other in zip(
        manifest["experts"], read_manifest(tmp_path / "reverse")["experts"]
    ):
        chosen = select_tokens(
            capture, entry, torch.ones(64).bfloat16(), set(collector.seen)
        )
        reordered = select_tokens(
            tmp_path / "reverse", other, torch.ones(64).bfloat16(), set(reverse.seen)
        )
        assert len(chosen) == 512
        assert [r[:3] for r in chosen] == [r[:3] for r in reordered]
    result = run(tmp_path, capture, tmp_path / "output", synthetic=True)
    assert result["status"] == "complete"
    assert len(result["outputs"]) == 72 and len(result["activations"]) == 108
    assert all(
        r["available_tokens"] == 640 and r["replay_tokens"] == 512
        for r in result["replay_selection"]
    )
    assert all(
        r["common_mse"] == r["common_nmse"] == 0
        for r in result["outputs"]
        if r["branch"] == "A" and r["mode"] == "FP"
    )
    assert max(r["common_nmse"] for r in result["outputs"] if r["mode"] == "FP") < 0.003
    assert {p.name for p in (tmp_path / "output").iterdir()} == {
        "results.json",
        "activations.csv",
        "outputs.csv",
        "activation_summary.csv",
        "layer_summary.csv",
        "summary.md",
    }
    assert {r["mode"] for r in result["activations"]} == {"W16A4", "W4A4"}
    assert not {"weights", "associations"}.intersection(result)
    assert (capture / "capture_manifest.json").read_bytes() == original
    full = run(tmp_path, capture, tmp_path / "all", synthetic=True, max_replay_tokens=0)
    assert all(r["replay_tokens"] == 640 for r in full["replay_selection"])
    assert all(r["tokens"] == 640 for r in full["outputs"])
    assert all(r["common_elements"] == 640 * 64 for r in full["outputs"])
    with pytest.raises(FileExistsError):
        run(tmp_path, capture, tmp_path / "output", synthetic=True)
    with pytest.raises(ValueError, match="real capture"):
        run(tmp_path, capture, tmp_path / "not-real")
    with pytest.raises(ValueError, match="max_replay_tokens"):
        run(
            tmp_path,
            capture,
            tmp_path / "invalid",
            synthetic=True,
            max_replay_tokens=127,
        )
    path = capture / manifest["experts"][0]["parts"][0]["file"]
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
    direct = RoutedCapture(selection)
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
        part = RoutedCapture(selection)
        report = collect(model, [batches[rank]], [rank], part)
        assert report["verified_dispatches"] == 8
        path = tmp_path / f"rank{rank}"
        part.save(path, source, rank_shard=True)
        paths.append(path)
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match="overlapping"):
        merge_shards([paths[0], paths[0]], tmp_path / "badmerge", source)
    merged = merge_shards(paths, tmp_path / "merged", source)
    assert merged["cleanup_status"] == "complete"
    assert not any(p.exists() for p in paths)
    assert all(
        e["retained"] == e["available"] == direct.counts[(e["layer"], e["expert"])]
        for e in merged["experts"]
    )
    direct.save(tmp_path / "direct", source)
    actual, expected = (
        archive_rows(tmp_path / "merged"),
        archive_rows(tmp_path / "direct"),
    )
    assert actual.keys() == expected.keys()
    for key, rows in expected.items():
        for x, y in zip(actual[key], rows):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_capture_norm_zero_negative_gamma_and_graph_failure():
    from .e2_sequential import capture_prefix

    norm = GlmMoeDsaRMSNorm(64).bfloat16()
    with torch.no_grad():
        norm.weight[0] = 0
        norm.weight[1] = -2
        x = torch.randn(4, 64).bfloat16()
        y = norm(x)
    collector = RoutedCapture({0: [0]})
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


@pytest.mark.parametrize(
    "failure", [None, "truncated", "duplicate", "overlap", "missing", "unrelated"]
)
def test_full_rank_merge_and_cleanup(tmp_path, failure):
    from safetensors.torch import load_file, save_file

    from .e2_sequential import merge_shards

    source = dict(kind="synthetic_test", global_sample_ids=[0, 1], world_size=2)
    paths = [tmp_path / "rank0", tmp_path / "rank1"]
    counts = {}
    for rank, path in enumerate(paths):
        cap, _ = make_capture((0 if failure == "overlap" else rank,), output=path)
        manifest = cap.save(path, source, rank_shard=True)
        for e in manifest["experts"]:
            key = (e["layer"], e["expert"])
            counts[key] = counts.get(key, 0) + e["available"]
    expected = archive_rows(paths[0]) | archive_rows(paths[1])
    if failure in ("truncated", "duplicate"):
        path = paths[0] / "capture_manifest.json"
        m = json.loads(path.read_text())
        entry = m["experts"][0]
        if failure == "truncated":
            entry["retained"] -= 1
        else:
            part = entry["parts"][0]
            file = paths[0] / part["file"]
            data = load_file(str(file))
            data["token_position"][1] = data["token_position"][0]
            save_file(data, str(file))
            part.update(sha256=file_hash(file), file_bytes=file.stat().st_size)
        path.write_text(json.dumps(m))
    if failure == "unrelated":
        (paths[0] / "keep.txt").write_text("unrelated file")
    if failure:
        with pytest.raises(ValueError):
            merge_shards(
                paths[:1] if failure == "missing" else paths,
                tmp_path / "merged",
                source,
            )
        assert all(path.exists() for path in paths)
        marker = tmp_path / "merged/capture_manifest.json"
        assert (
            not marker.exists()
            or json.loads(marker.read_text())["status"] != "complete"
        )
        return
    merged = merge_shards(paths[::-1], tmp_path / "merged", source)
    assert merged["status"] == merged["cleanup_status"] == "complete"
    assert all(not path.exists() for path in paths)
    for e in merged["experts"]:
        assert (
            e["retained"] == e["available"] == counts[(e["layer"], e["expert"])] == 640
        )
    actual = archive_rows(tmp_path / "merged")
    assert actual.keys() == expected.keys()
    for key in actual:
        for x, y in zip(actual[key], expected[key]):
            assert torch.equal(x, y)


def test_single_sample_over512_and_insufficient_replay(tmp_path):
    cap, _ = make_capture((0,), length=900, output=tmp_path / "large")
    manifest = cap.save(tmp_path / "large", dict(kind="synthetic_test"))
    assert all(e["available"] == e["retained"] == 600 for e in manifest["experts"])
    small, _ = make_capture((0,), length=96)
    small_manifest = small.save(tmp_path / "small", dict(kind="synthetic_test"))
    with pytest.raises(ValueError, match="insufficient"):
        select_tokens(
            tmp_path / "small",
            small_manifest["experts"][0],
            torch.ones(64).bfloat16(),
            set(small.seen),
        )


def test_metrics_and_aggregation():
    a = error_stats(torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 4.0]]))
    b = error_stats(
        torch.tensor([[3.0, 3.0], [3.0, 3.0]]), torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    )
    assert a["mse"] == 2.5 and a["nmse"] == 1
    assert b["mse"] == 1 and b["nmse"] == pytest.approx(1 / 9)
    rows = [dict(a, layer=0, tokens=1), dict(b, layer=0, tokens=2)]
    agg = summarize(rows, ("layer",), ("",))[0]
    assert agg["pooled_mse"] == pytest.approx(9 / 6)
    assert agg["pooled_nmse"] == pytest.approx(9 / 41)
    assert agg["mean_expert_nmse"] == pytest.approx((1 + 1 / 9) / 2)
    combined = combine_batches(rows, ("layer",), ("",))[0]
    direct = error_stats(
        torch.tensor([[1.0, 2.0], [3.0, 3.0], [3.0, 3.0]]),
        torch.tensor([[2.0, 4.0], [4.0, 4.0], [4.0, 4.0]]),
    )
    for key in ("mse", "nmse", "sse", "elements", "token_nmse_p50", "token_nmse_p95"):
        assert combined[key] == pytest.approx(direct[key])
    zero = error_stats(torch.zeros(2, 2), torch.ones(2, 2))
    assert zero["nmse"] is None and zero["token_nmse_p50"] is None
    assert not {
        "max_abs",
        "mean_abs",
        "rmse",
        "cosine",
        "token_nmse_p90",
        "token_nmse_p99",
    }.intersection(a)


def test_activation_metrics_and_shared_token_inputs(monkeypatch):
    from . import e2

    torch.manual_seed(10)
    z = torch.randn(7, 64)
    gamma = torch.rand(64) + 0.5
    ws = [torch.randn(32, 64) / 8, torch.randn(32, 64) / 8, torch.randn(64, 32) / 6]
    original = e2.replay
    seen = []

    def spy(x, weights, quantized, mode, dtype):
        seen.append(x.clone())
        return original(x, weights, quantized, mode, dtype)

    monkeypatch.setattr(e2, "replay", spy)
    _, acts = evaluate(z, z * gamma, gamma, ws, torch.float32)
    for group, expected in enumerate((z * gamma, z, core.rotation(64).apply(z))):
        for x in seen[group * 4 : (group + 1) * 4]:
            torch.testing.assert_close(x, expected)
    assert len(acts) == 18
    quant = [core.qdq(w) for w in ws]
    for mode in ("FP", "W4A16", "W16A4", "W4A4"):
        _, records = original(z, ws, quant, mode, torch.float32)
        assert len(records) == (3 if mode in ("W16A4", "W4A4") else 0)
        if records:
            expected = error_stats(z, core.qdq(z, True))
            assert records[0]["mse"] == expected["mse"]
            assert records[1]["nmse"] == expected["nmse"]
