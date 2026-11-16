"""Capture contract for an existing unquantized baseline; no model loader required."""

import hashlib
import json
import random
import time
from pathlib import Path

import torch
from compressed_tensors.offload import align_module_device

from llmcompressor.modifiers.transform.flex_smooth.mappings import glm_mappings

from .core import digest, memory, provenance, sync, validate_cache


def split_records(path, tokenizer, seed=42):
    """Reuse local JSON list[str]/JSONL text/messages; deduplicate tokenized documents.

    Whole documents go to only one split, including their truncated contexts.
    No padding is used. Original row indices and token hashes are retained.
    """
    text = Path(path).read_text(encoding="utf-8")
    records = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(s) for s in text.splitlines() if s.strip()]
    )
    unique = {}
    for index, row in enumerate(records):
        if isinstance(row, str):
            row = {"text": row}
        if "messages" in row:
            ids = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
        else:
            ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        if not ids:
            continue
        key = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        unique.setdefault(key, (index, key, ids))
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    if len(rows) < 2:
        raise ValueError("Need at least two distinct tokenized documents")
    return rows[::2], rows[1::2]


@torch.no_grad()
def capture(
    model,
    splits,
    output,
    layers=(3, 40),
    mappings=("input_layernorm", "q_a_layernorm"),
    tokens=2048,
    context=2048,
    device="cpu",
    model_path="",
    seed=42,
):
    if tokens < 1 or context < 1:
        raise ValueError("Positive tokens/context required")
    if model.training or any(
        getattr(model.config, k, None)
        for k in ("quarot_config", "flex_smooth_config", "quantization_config")
    ):
        raise ValueError("Capture requires eval, untransformed, unquantized baseline")
    plan = [
        m
        for m in glm_mappings(model, ("norm-linear",))
        if int(m.source.split(".")[2]) in layers
        and m.source.rsplit(".", 1)[-1] in mappings
    ]
    if len(plan) != len(set(layers)) * len(set(mappings)):
        raise ValueError("Requested layers/mappings were not resolved completely")
    if set(r[1] for r in splits[0]) & set(r[1] for r in splits[1]):
        raise ValueError("Calibration/validation document overlap")
    groups = {}
    for m in plan:
        weights = []
        for name in m.consumers:
            module = model.get_submodule(name)
            if getattr(module, "quantization_scheme", None):
                raise ValueError("Quantized consumer")
            with align_module_device(module):
                weights.append(module.weight.detach().cpu().clone())
        groups[m.source] = dict(
            consumers=list(m.consumers),
            consumer_rows=[len(w) for w in weights],
            weight=torch.cat(weights),
        )
    sync(device)
    start = time.perf_counter()
    for split, records in zip(("calibration", "validation"), splits):
        collected = {m.source: [] for m in plan}
        identities = []
        current = [0]
        handles = []

        def hook(key):
            def receive(module, args, kwargs):
                x = args[0] if args else kwargs["input"]
                if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != current[0]:
                    raise ValueError("Capture expects one unpadded prefill")
                collected[key].append(x.detach().reshape(-1, x.shape[-1]).cpu().clone())

            return receive

        try:
            for m in plan:
                handles.append(
                    model.get_submodule(m.consumers[0]).register_forward_pre_hook(
                        hook(m.source), with_kwargs=True
                    )
                )
            for row, key, ids in records:
                remaining = tokens - len(identities)
                if not remaining:
                    break
                ids = ids[: min(context, remaining)]
                current[0] = len(ids)
                input_ids = torch.tensor([ids], device=device)
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                )
                identities.extend(f"{key}:{pos}" for pos in range(len(ids)))
            if len(identities) != tokens:
                raise ValueError(
                    f"Insufficient independent {split} tokens: {len(identities)}/{tokens}"  # noqa: E501
                )
        finally:
            for h in handles:
                h.remove()
        for m in plan:
            groups[m.source][split] = torch.cat(collected[m.source]).to(
                groups[m.source]["weight"].dtype
            )
            groups[m.source][split + "_ids"] = identities
    sync(device)
    cache = dict(
        schema=1,
        kind="real",
        groups=groups,
        seed=seed,
        model_path=str(model_path),
        capture_seconds=time.perf_counter() - start,
        memory=memory(device),
        provenance=provenance(),
    )
    validate_cache(cache)
    for g in groups.values():
        g["hashes"] = {k: digest(g[k]) for k in ("weight", "calibration", "validation")}
    torch.save(cache, output)
    return cache
