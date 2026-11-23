"""Incremental o_proj-only hook; reuse historical two-rank loader and runner."""

import torch

from ..distributed_capture import capture_split, merge_shards, partition, prepare
from ..e2_collect import phase


def run_ov(
    model,
    splits,
    source,
    device,
    output,
    reference,
    rank,
    context,
    check,
    nmse_tol,
    max_abs_tol,
):
    root = output / "capture"
    phase(lambda: root.mkdir() if rank == 0 else None)
    metadata = dict(
        source,
        layers=[6, 16, 60, 77],
        mapping_suffixes=["self_attn.kv_b_proj"],
        rank_plan=partition(splits, 2048, context, 2),
        token_ids={
            s: prepare(records, 2048, context)[1]
            for s, records in zip(("calibration", "validation"), splits)
        },
    )
    shard = phase(
        lambda: capture_split(
            model,
            splits[rank],
            rank,
            metadata,
            [6, 16, 60, 77],
            ["kv_b_proj"],
            2048,
            context,
            device,
            check,
            mapping_kind="ov",
        )
    )
    timing = {
        k: shard[k]
        for k in (
            "capture_seconds",
            "forward_seconds",
            "trace_seconds",
            "weight_snapshot_seconds",
        )
    }

    def save():
        tmp = root / f"rank{rank}.tmp"
        torch.save(shard, tmp)
        tmp.replace(root / f"rank{rank}.pt")

    phase(save)
    shard = None

    def merge():
        if rank == 0:
            check()
            result = merge_shards(
                [
                    torch.load(
                        root / f"rank{i}.pt", map_location="cpu", weights_only=True
                    )
                    for i in (0, 1)
                ]
            )
            tmp = root / "pending-cache.pt"
            torch.save(result, tmp)
            tmp.replace(root / "cache.pt")

    phase(merge)
    return [dict(stage="o_proj_incremental", **timing)]
