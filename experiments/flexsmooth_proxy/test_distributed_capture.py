"""CPU contracts; actual shared-backing two-process test additionally needs Linux."""

import copy
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from .capture import capture
from .core import digest, provenance, validate_cache
from .distributed_capture import (
    capture_split,
    compare_reference,
    merge_shards,
    partition,
    prepare,
    split_for_rank,
)
from .e2_collect import backing_signature, phase, runtime_contract
from .run import evaluate, search, synthetic
from .test_offload import scratch


def parts():
    original = synthetic(tokens=8, dtype=torch.bfloat16)
    source = dict(
        seed=42,
        model_path="tiny",
        layers=[3],
        mapping_suffixes=["input_layernorm"],
        provenance=provenance(),
        capture_sources={"test": "synthetic"},
        world_size=2,
        token_ids={
            s: next(iter(original["groups"].values()))[s + "_ids"]
            for s in ("calibration", "validation")
        },
    )
    source["rank_plan"] = {
        str(r): dict(
            split=split_for_rank(r),
            batch_indices=[0],
            positions=list(range(8)),
            token_ids=source["token_ids"][split_for_rank(r)],
        )
        for r in (0, 1)
    }
    shards = []
    for rank in (0, 1):
        split = split_for_rank(rank)
        groups = {}
        for name, g in original["groups"].items():
            groups[name] = dict(
                weight=g["weight"].clone(),
                weight_hash=digest(g["weight"]),
                consumers=g["consumers"].copy(),
                consumer_rows=g["consumer_rows"].copy(),
                consumer_dtypes=[str(g["weight"].dtype)] * len(g["consumers"]),
                activation=g[split].clone(),
                activation_hash=digest(g[split]),
                token_ids=g[split + "_ids"].copy(),
            )
        shards.append(
            dict(
                rank=rank,
                split=split,
                groups=groups,
                status="complete",
                source=copy.deepcopy(source),
                capture_seconds=rank + 1,
                active=True,
                batch_indices=[0],
                positions=list(range(8)),
            )
        )
    return shards


def test_rank_and_preparation():
    assert [split_for_rank(r) for r in (0, 1)] == ["calibration", "validation"]
    for rank, world in [(0, 1), (0, 3), (2, 2)]:
        with pytest.raises(ValueError):
            split_for_rank(rank, world)
    batches, ids = prepare([(0, "a", [1, 2, 3, 4]), (1, "b", [5, 6, 7])], 5, 3)
    assert ids == ["a:0", "a:1", "a:2", "b:0", "b:1"]
    assert [b["input_ids"].tolist() for b in batches] == [[[1, 2, 3]], [[5, 6]]]
    with pytest.raises(ValueError):
        prepare([(0, "a", [1])], 2, 3)
    with pytest.raises(ValueError):
        prepare([(0, "a", [1]), (1, "a", [2])], 2, 3)


@pytest.mark.parametrize(
    "world,lengths", [(8, [8]), (8, [2, 2, 2, 2]), (4, [1, 3, 1, 3])]
)
def test_whole_prefill_partition_merge_order_and_idle(world, lengths):
    base = parts()
    splits = [
        [(i, f"{split}-{i}", list(range(1, n + 1))) for i, n in enumerate(lengths)]
        for split in ("cal", "val")
    ]
    plan = partition(splits, 8, 8, world)
    source = copy.deepcopy(base[0]["source"])
    source.update(
        world_size=world,
        rank_plan=plan,
        token_ids={split_for_rank(i): prepare(splits[i], 8, 8)[1] for i in (0, 1)},
    )
    shards = []
    for rank in range(world):
        s = copy.deepcopy(base[rank % 2])
        assignment = plan[str(rank)]
        s.update(
            rank=rank,
            source=source,
            active=bool(assignment["positions"]),
            positions=assignment["positions"],
            batch_indices=assignment["batch_indices"],
        )
        for g in s["groups"].values():
            g["activation"] = g["activation"][assignment["positions"]]
            g["activation_hash"] = digest(g["activation"])
            g["token_ids"] = assignment["token_ids"]
        shards.append(s)
    merged = merge_shards(list(reversed(shards)))
    g = next(iter(merged["groups"].values()))
    for i, split in enumerate(("calibration", "validation")):
        torch.testing.assert_close(
            g[split],
            next(iter(base[i]["groups"].values()))["activation"],
            rtol=0,
            atol=0,
        )
    assert sum(s["active"] for s in shards) == 2 * min(world // 2, len(lengths))
    if len(lengths) == 1:
        assert {s["rank"] for s in shards if s["active"]} == {0, 1}
    shards[0]["positions"] = [100]
    with pytest.raises(ValueError, match="assignment"):
        merge_shards(shards)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate_rank",
        "failed",
        "split",
        "mapping",
        "both_mapping",
        "order",
        "rows",
        "dtype",
        "weights",
        "activation",
        "duplicate_token",
        "overlap",
        "source",
    ],
)
def test_merge_rejects(fault):
    shards = parts()
    group = next(iter(shards[1]["groups"].values()))
    if fault == "missing":
        shards.pop()
    elif fault == "duplicate_rank":
        shards[1]["rank"] = 0
    elif fault == "failed":
        shards[1]["status"] = "failed"
    elif fault == "split":
        shards[1]["split"] = "calibration"
    elif fault == "mapping":
        shards[1]["groups"].clear()
    elif fault == "both_mapping":
        for s in shards:
            s["source"]["layers"].append(40)
    elif fault == "order":
        group["consumers"].reverse()
    elif fault == "rows":
        group["consumer_rows"].reverse()
    elif fault == "dtype":
        group["consumer_dtypes"][0] = "torch.float32"
    elif fault == "weights":
        group["weight"][0, 0] += 1
        group["weight_hash"] = digest(group["weight"])
    elif fault == "activation":
        group["activation"][0, 0] += 1
    elif fault in ("overlap", "duplicate_token"):
        group["token_ids"][0] = "cal:0" if fault == "overlap" else group["token_ids"][1]
        for s in shards:
            s["source"]["token_ids"]["validation"] = group["token_ids"].copy()
    elif fault == "source":
        shards[1]["source"]["seed"] += 1
    with pytest.raises(ValueError):
        merge_shards(shards)


def test_schema_and_existing_search_evaluate():
    path = scratch()
    cache = validate_cache(merge_shards(list(reversed(parts()))))
    torch.save(cache, path / "cache.pt")
    loaded = torch.load(path / "cache.pt", weights_only=True)
    rows = compare_reference(loaded, cache, 0, 0)
    assert all(r["passed"] and r["bitwise_equal"] for r in rows)
    # Tiny matrix only: exercises unchanged E1 consumer of the merged schema.
    args = SimpleNamespace(
        device="cpu", batch=4, rows=32, proxies=["INT8"], output=path
    )
    found = search(loaded, args)
    evaluate(loaded, found, args)
    assert (path / "evaluation.json").is_file()
    assert (path / "summary.csv").is_file()


def test_comparison_reports_numeric_difference():
    cache = merge_shards(parts())
    ref = copy.deepcopy(cache)
    next(iter(ref["groups"].values()))["validation"][0, 0] += 0.5
    rows = compare_reference(cache, ref, 0, 0)
    assert rows[0]["passed"] and not rows[1]["passed"]
    assert rows[1]["max_abs"] > 0 and not rows[1]["bitwise_equal"]


def test_failure_handshake(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda errors, error: errors.__setitem__(slice(None), [error, "rank1 failed"]),
    )
    with pytest.raises(RuntimeError, match="rank1 failed"):
        phase(lambda: "local success is insufficient")
    assert len(runtime_contract()) == 3


def tiny_model():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    torch.manual_seed(42)
    cfg = GlmMoeDsaConfig(
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
    cfg._attn_implementation = "eager"
    cfg.dtype = torch.bfloat16
    return GlmMoeDsaForCausalLM(cfg).bfloat16().eval()


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_one_model_smoke_then_formal(monkeypatch, fail_smoke):
    from . import e2_collect as collector

    path = scratch()
    model = object()
    calls = []
    reference = merge_shards(parts())
    if fail_smoke:
        next(iter(reference["groups"].values()))["validation"].add_(10)
    monkeypatch.setattr(collector, "phase", lambda fn: fn())
    monkeypatch.setattr(collector, "load_cache", lambda p: reference)

    def fake_capture(current, records, rank, source, layers, mappings, tokens, *args):
        assert current is model
        calls.append((layers, tokens))
        root = path / ("capture-smoke" if tokens == 128 else "capture")
        shards = parts()
        for s in shards:
            s.update(forward_seconds=1, trace_seconds=0, weight_snapshot_seconds=0)
        torch.save(shards[1], root / "rank1.pt")
        return shards[0]

    monkeypatch.setattr(collector, "capture_split", fake_capture)
    splits = ([(0, "cal", [1] * 2048)], [(1, "val", [2] * 2048)])

    def run():
        return collector.run_captures(
            model, splits, {}, "cpu", path, path / "old.pt", 0, 2048, lambda: None, 0, 0
        )

    if fail_smoke:
        with pytest.raises(ValueError, match="Smoke differs"):
            run()
        assert not (path / "capture").exists()
    else:
        assert len(run()) == 2
        assert calls == [([3], 128), ([6, 16, 60, 77], 2048)]
        assert (path / "capture/cache.pt").exists()
    assert "max_memory" not in collector.loading_options(path)


def tiny_inputs():
    splits = (
        [(0, "cal0", list(range(1, 9))), (1, "cal1", list(range(9, 17)))],
        [(2, "val0", list(range(17, 25))), (3, "val1", list(range(25, 33)))],
    )
    source = dict(
        seed=42,
        model_path="tiny",
        layers=[1],
        mapping_suffixes=["input_layernorm", "self_attn.q_a_layernorm"],
        provenance=provenance(),
        capture_sources={"test": "tiny"},
        world_size=2,
        rank_plan=partition(splits, 16, 8, 2),
        token_ids={
            split_for_rank(i): prepare(s, 16, 8)[1] for i, s in enumerate(splits)
        },
    )
    return splits, source


@pytest.mark.parametrize("world", [2, 8])
def test_tiny_split_capture_matches_legacy_and_stops(world):
    from llmcompressor.modeling.moe.linearize import linearize_moe

    torch.set_num_threads(1)
    model = tiny_model()
    linearize_moe(model)
    splits, source = tiny_inputs()
    source.update(world_size=world, rank_plan=partition(splits, 16, 8, world))
    path = scratch()
    reference = capture(
        model, splits, path / "legacy.pt", layers=[1], tokens=16, context=8
    )

    def forbidden(*args):
        # FX tracing visits Python with symbolic proxies before prefix trimming.
        if isinstance(args[1][0], torch.Tensor):
            raise AssertionError("Capture ran beyond target attention")

    handles = [
        m.register_forward_hook(forbidden)
        for m in (
            model.model.layers[1].mlp,
            model.model.layers[2].input_layernorm,
            model.lm_head,
        )
    ]
    try:
        shards = [
            capture_split(
                model,
                splits[r % 2],
                r,
                source,
                [1],
                ["input_layernorm", "q_a_layernorm"],
                16,
                8,
                "cpu",
            )
            for r in range(world)
        ]
    finally:
        for h in handles:
            h.remove()
    cache = merge_shards(shards)
    assert all(r["passed"] for r in compare_reference(cache, reference, 0, 0))
    assert all(not m._forward_pre_hooks for m in model.modules())


def shared_worker(rank, path):
    """Real CT source/meta loading and Gloo orchestration, no accelerator needed."""
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils import load_context

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(path / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        with load_context():
            model = AutoModelForCausalLM.from_pretrained(
                path / "model",
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto_offload",
                max_memory={"cpu": 64 * 2**20},
                offload_folder=str(path / "offload"),
            ).eval()
        back = backing_signature(model)
        peers = [None, None]
        dist.all_gather_object(peers, back)
        assert peers[0] == peers[1]
        splits, source = tiny_inputs()
        shard = phase(
            lambda: capture_split(
                model,
                splits[rank],
                rank,
                source,
                [1],
                ["input_layernorm", "q_a_layernorm"],
                16,
                8,
                "cpu",
            )
        )
        torch.save(shard, path / f"rank{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform != "linux", reason="CT shared backing requires Linux /dev/shm"
)
def test_two_process_shared_loader_and_capture():
    from llmcompressor.modeling.moe.linearize import linearize_moe

    path = scratch().resolve()
    (path / "offload").mkdir()
    model = tiny_model()
    model.save_pretrained(path / "model")
    # Use the same from_pretrained dtype exceptions as distributed workers.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path / "model", dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    linearize_moe(model)
    reference = capture(
        model, tiny_inputs()[0], path / "reference.pt", layers=[1], tokens=16, context=8
    )
    workers = torch.multiprocessing.spawn(
        shared_worker, args=(path,), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Two-process shared loader exceeded 180s")
    finally:
        for worker in workers.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
    shards = [torch.load(path / f"rank{r}.pt", weights_only=True) for r in (0, 1)]
    assert all(
        r["passed"]
        for r in compare_reference(merge_shards(shards), reference, 1e-5, 0.125)
    )
