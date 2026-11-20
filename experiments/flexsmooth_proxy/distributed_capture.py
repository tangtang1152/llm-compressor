"""Single-split norm-linear capture; schema-1 merge, no experimental transforms."""

import time

import torch
from compressed_tensors.offload import align_module_device

from llmcompressor.modifiers.transform.flex_smooth.mappings import glm_mappings

from .core import digest, metrics, sync, validate_cache
from .offload import SequentialRunner

SPLITS = ("calibration", "validation")


def split_for_rank(rank, world=2):
    if world not in (2, 4, 6, 8) or not 0 <= rank < world:
        raise ValueError("Use 2/4/6/8 ranks: even calibration, odd validation")
    return SPLITS[rank % 2]


def prepare(records, tokens, context):
    if tokens < 1 or context < 1:
        raise ValueError("Positive tokens/context required")
    batches, ids = [], []
    for _, key, values in records:
        values = values[: min(context, tokens - len(ids))]
        if values:
            x = torch.tensor([values])
            batches.append(dict(input_ids=x, attention_mask=torch.ones_like(x)))
            ids.extend(f"{key}:{i}" for i in range(len(values)))
        if len(ids) == tokens:
            break
    if len(ids) != tokens or len(set(ids)) != tokens:
        raise ValueError("Insufficient/duplicate split token identities")
    return batches, ids


def partition(splits, tokens, context, world):
    """Assign whole original prefills, greedy by tokens; never split a context."""
    split_for_rank(0, world)
    plan = {
        str(r): dict(
            split=split_for_rank(r, world), batch_indices=[], positions=[], token_ids=[]
        )
        for r in range(world)
    }
    for split_index, records in enumerate(splits):
        batches, ids = prepare(records, tokens, context)
        ranks = list(range(split_index, world, 2))
        offset = 0
        for batch_index, batch in enumerate(batches):
            size = batch["input_ids"].shape[1]
            rank = min(ranks, key=lambda r: (len(plan[str(r)]["positions"]), r))
            item = plan[str(rank)]
            item["batch_indices"].append(batch_index)
            item["positions"].extend(range(offset, offset + size))
            item["token_ids"].extend(ids[offset : offset + size])
            offset += size
    return plan


@torch.no_grad()
def capture_split(
    model,
    records,
    rank,
    source,
    layers,
    mappings,
    tokens,
    context,
    device,
    check=lambda: None,
):
    sync(device)
    capture_start = time.perf_counter()
    world = source["world_size"]
    split = split_for_rank(rank, world)
    if model.training or any(
        getattr(model.config, k, None)
        for k in ("quantization_config", "quarot_config", "flex_smooth_config")
    ):
        raise ValueError("Original eval model required")
    all_batches, all_ids = prepare(records, tokens, context)
    if all_ids != source["token_ids"][split]:
        raise ValueError("Prepared identities changed")
    assignment = source["rank_plan"][str(rank)]
    batches = [all_batches[i] for i in assignment["batch_indices"]]
    ids = assignment["token_ids"]
    if ids != [all_ids[i] for i in assignment["positions"]]:
        raise ValueError("Rank assignment identities changed")
    # Verify whole-batch boundaries too; this is not prefix replay/token slicing.
    offsets = [0]
    for batch in all_batches:
        offsets.append(offsets[-1] + batch["input_ids"].shape[1])
    expected_positions = [
        p
        for i in assignment["batch_indices"]
        for p in range(offsets[i], offsets[i + 1])
    ]
    if expected_positions != assignment["positions"]:
        raise ValueError("Rank assignment splits an original prefill")
    plan = [
        m
        for m in glm_mappings(model, ("norm-linear",))
        if int(m.source.split(".")[2]) in layers and m.source.split(".")[-1] in mappings
    ]
    if len(plan) != len(layers) * len(mappings):
        raise ValueError("Incomplete mapping plan")
    groups = {}
    for m in plan:
        weights = []
        for name in m.consumers:
            module = model.get_submodule(name)
            if getattr(module, "quantization_scheme", None):
                raise ValueError("Quantized consumer")
            with align_module_device(module, torch.device("cpu")):
                weights.append(module.weight.detach().cpu().clone())
        weight = torch.cat(weights, dim=0)
        groups[m.source] = dict(
            consumers=list(m.consumers),
            consumer_rows=[len(w) for w in weights],
            consumer_dtypes=[str(w.dtype) for w in weights],
            weight=weight,
            weight_hash=digest(weight),
        )
    weight_seconds = time.perf_counter() - capture_start
    trace_start = time.perf_counter()
    runner = SequentialRunner(model, batches, plan, device) if batches else None
    sync(device)
    trace_seconds = time.perf_counter() - trace_start
    values = {m.source: [] for m in plan}
    seen = set()
    current = [None]

    def before(index):
        check()
        current[0] = index

    def hook(name):
        def receive(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            key = (name, current[0])
            if (
                key in seen
                or x.ndim != 3
                or x.shape[:2] != batches[current[0]]["input_ids"].shape
            ):
                raise ValueError("Duplicate/incorrect mapping prefill")
            seen.add(key)
            values[name].append(x.detach().reshape(-1, x.shape[-1]).cpu().clone())

        return receive

    handles = []
    sync(device)
    start = time.perf_counter()
    try:
        for m in plan:
            handles.append(
                model.get_submodule(m.consumers[0]).register_forward_pre_hook(
                    hook(m.source), with_kwargs=True
                )
            )
        if runner:
            runner.run(batches, before)
        sync(device)
        check()
    finally:
        for h in handles:
            h.remove()
    if len(seen) != len(plan) * len(batches):
        raise ValueError("Incomplete split capture")
    for name, g in groups.items():
        x = (
            torch.cat(values[name]).to(g["weight"].dtype)
            if batches
            else g["weight"].new_empty((0, g["weight"].shape[1]))
        )
        g.update(activation=x, token_ids=ids, activation_hash=digest(x))
    return dict(
        rank=rank,
        split=split,
        status="complete",
        active=bool(batches),
        positions=assignment["positions"],
        batch_indices=assignment["batch_indices"],
        source=source,
        groups=groups,
        capture_seconds=time.perf_counter() - capture_start,
        forward_seconds=time.perf_counter() - start,
        weight_snapshot_seconds=weight_seconds,
        trace_seconds=trace_seconds,
        subgraphs=len(runner.subgraphs) if runner else 0,
    )


def merge_shards(shards):
    if not shards:
        raise ValueError("Missing rank shards")
    world = shards[0]["source"]["world_size"]
    split_for_rank(0, world)
    if len(shards) != world or {s["rank"] for s in shards} != set(range(world)):
        raise ValueError("Missing or duplicate rank")
    shards = sorted(shards, key=lambda s: s["rank"])
    a = shards[0]
    for rank, s in enumerate(shards):
        if (
            s["status"] != "complete"
            or s["split"] != split_for_rank(rank, world)
            or s["source"] != a["source"]
        ):
            raise ValueError("Incomplete/inconsistent shard metadata")
    if any(set(s["groups"]) != set(a["groups"]) for s in shards) or not a["groups"]:
        raise ValueError("Mapping sets differ")
    source = a["source"]
    expected = {
        f"model.layers.{i}.{suffix}"
        for i in source["layers"]
        for suffix in source["mapping_suffixes"]
    }
    if set(a["groups"]) != expected:
        raise ValueError("Missing mapping in both shards")
    for s in shards:
        assignment = source["rank_plan"][str(s["rank"])]
        if (
            s["positions"] != assignment["positions"]
            or s["batch_indices"] != assignment["batch_indices"]
            or s["active"] != bool(s["positions"])
            or assignment["split"] != s["split"]
        ):
            raise ValueError("Rank assignment mismatch")
    for split in SPLITS:
        positions = [i for s in shards if s["split"] == split for i in s["positions"]]
        if sorted(positions) != list(range(len(source["token_ids"][split]))):
            raise ValueError("Missing/duplicate token position across rank shards")
    groups = {}
    for name, left in a["groups"].items():
        for s in shards:
            right = s["groups"][name]
            for field in ("consumers", "consumer_rows", "consumer_dtypes"):
                if left[field] != right[field]:
                    raise ValueError("Consumer order/geometry/dtypes differ")
            if (
                digest(right["weight"]) != left["weight_hash"]
                or right["weight_hash"] != left["weight_hash"]
                or right["weight"].shape != left["weight"].shape
                or right["weight"].dtype != left["weight"].dtype
            ):
                raise ValueError("Weight hash mismatch")
        if len(set(left["consumers"])) != len(left["consumers"]) or any(
            r <= 0 for r in left["consumer_rows"]
        ):
            raise ValueError("Invalid consumer geometry")
        g = {
            k: left[k]
            for k in ("weight", "consumers", "consumer_rows", "consumer_dtypes")
        }
        for s in shards:
            part = s["groups"][name]
            split = s["split"]
            if (
                digest(part["activation"]) != part["activation_hash"]
                or part["token_ids"]
                != [source["token_ids"][split][i] for i in s["positions"]]
                or len(part["activation"]) != len(s["positions"])
                or part["activation"].ndim != 2
                or part["activation"].shape[1] != left["weight"].shape[1]
                or part["activation"].dtype != left["weight"].dtype
            ):
                raise ValueError("Activation hash/token identity mismatch")
        for split in SPLITS:
            selected = [s for s in shards if s["split"] == split]
            positions = torch.tensor([i for s in selected for i in s["positions"]])
            g[split] = torch.cat([s["groups"][name]["activation"] for s in selected])[
                positions.argsort()
            ]
            g[split + "_ids"] = source["token_ids"][split]
        groups[name] = g
    cache = dict(
        schema=1,
        kind="real",
        seed=source["seed"],
        model_path=source["model_path"],
        provenance=source["provenance"],
        capture_backend="multi-rank-whole-prefill-parallel",
        capture_source_sha256=source["capture_sources"],
        distributed_source=source,
        capture_seconds=max(s["capture_seconds"] for s in shards),
        groups=groups,
    )
    validate_cache(cache)
    for g in groups.values():
        g["hashes"] = {
            key: digest(g[key]) for key in ("weight", "calibration", "validation")
        }
    return cache


def compare_reference(cache, reference, nmse_tol, max_abs_tol):
    validate_cache(reference)
    rows = []
    for name, g in cache["groups"].items():
        ref = reference["groups"][name]
        if (
            g["consumers"] != ref["consumers"]
            or digest(g["weight"]) != digest(ref["weight"])
            or g["weight"].shape != ref["weight"].shape
            or g["weight"].dtype != ref["weight"].dtype
        ):
            raise ValueError("Reference original weights/consumer order differ")
        for split in SPLITS:
            n = len(g[split])
            x = ref[split][:n]
            if g[split + "_ids"] != ref[split + "_ids"][:n]:
                raise ValueError("Reference token identity mismatch")
            m = metrics(x, g[split])
            rows.append(
                dict(
                    mapping=name,
                    split=split,
                    bitwise_equal=digest(x) == digest(g[split]),
                    **m,
                    passed=m["finite"]
                    and m["nmse"] <= nmse_tol
                    and m["max_abs"] <= max_abs_tol,
                )
            )
    return rows
