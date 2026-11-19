"""Independent P1 benchmark and E2 layer-extension pipeline."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from ..capture import capture, split_records
from ..core import provenance
from ..offload import loader_options
from ..run import evaluate, search, write_json
from . import e2, p1
from .common import csv_rows, file_hash, identities, load_cache, stage


def token_plan(splits, tokens, context):
    result = {}
    for name, records in zip(("calibration", "validation"), splits):
        ids = []
        for _, key, values in records:
            count = min(context, len(values), tokens - len(ids))
            ids.extend(f"{key}:{i}" for i in range(count))
            if len(ids) == tokens:
                break
        if len(ids) != tokens:
            raise ValueError("Insufficient independent tokens")
        result[name] = ids
    return result


def capture_extension(args, out, manifest):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from llmcompressor.utils import load_context

    old = load_cache(args.old_cache)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if config.model_type != "glm_moe_dsa" or config.num_hidden_layers <= 77:
        raise ValueError("Expected GLM-5.2 native decoder layout through L77")
    if Path(old.get("model_path", "")).resolve() != Path(args.model_path).resolve():
        raise ValueError(
            "Model checkpoint path differs from original capture; verify before extending"  # noqa: E501
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    splits = split_records(args.data, tokenizer, args.seed)
    plan = token_plan(splits, args.tokens, args.context)
    old_ids = identities(old)
    if args.seed != old.get("seed") or any(
        plan[s] != old_ids[s][: args.tokens] for s in plan
    ):
        raise ValueError(
            "New tokenization/split differs from original ordered token identities"
        )
    if args.tokens not in (128, 2048):
        raise ValueError("Use 128-token smoke or 2048-token formal capture")
    options = loader_options(torch.bfloat16, args.offload_dir, args.max_memory)
    manifest.update(
        data_sha256=file_hash(args.data),
        old_cache_sha256=file_hash(args.old_cache),
        prefix_stop="L77 attention; earlier native MoE required, no expert quantization",  # noqa: E501
    )
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **options).eval()
    if getattr(model.config, "num_hidden_layers", 0) <= 77:
        raise ValueError("L77 must be an ordinary decoder layer, not MTP")
    destination = out / "capture"
    destination.mkdir()
    cache = capture(
        model,
        splits,
        destination / "cache.pt",
        layers=(6, 16, 60, 77),
        tokens=args.tokens,
        context=args.context,
        device=args.device,
        model_path=args.model_path,
        seed=args.seed,
        sequential=True,
        loading={
            k: str(v) if isinstance(v, torch.dtype) else v for k, v in options.items()
        },
    )
    if identities(cache) != plan:
        raise ValueError("Captured token identities differ from preflight plan")
    manifest["capture_sha256"] = file_hash(destination / "cache.pt")
    csv_rows(
        out / "capture_counts.csv",
        [
            dict(
                mapping=n,
                calibration=len(g["calibration"]),
                validation=len(g["validation"]),
            )
            for n, g in cache["groups"].items()
        ],
    )
    (out / "report.md").write_text(
        "# E2 capture complete\n\nOnly L6/L16/L60/L77 captured in one sequential prefix per split. Run e2-smoke next. Original L3/L40 cache untouched.\n",  # noqa: E501
        encoding="utf-8",
    )


def execute(args, out, manifest):
    required = {
        "p1": ("cache",),
        "e2-capture": ("old_cache", "model_path", "data", "offload_dir"),
        "e2-smoke": ("cache",),
        "e2-search": ("cache", "smoke"),
        "e2-evaluate": ("cache", "search"),
        "e2-analyze": ("old", "new"),
    }
    missing = [key for key in required[args.stage] if getattr(args, key, None) is None]
    if missing:
        raise ValueError("Missing stage arguments: " + ", ".join(missing))
    if args.stage == "p1":
        return p1.run(args, out, manifest)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(torch.device(args.device))
    if args.stage == "e2-capture":
        return capture_extension(args, out, manifest)
    if args.stage == "e2-analyze":
        return e2.analyze(args, out, manifest)
    cache = load_cache(args.cache)
    if {int(n.split(".")[2]) for n in cache["groups"]} != e2.NEW_LAYERS:
        raise ValueError("E2 work is restricted to newly selected L6/L16/L60/L77")
    if args.stage == "e2-smoke":
        rows = e2.smoke(cache, args.device, args.batch, args.rows)
        manifest.update(cache_sha256=file_hash(args.cache), status="complete")
        write_json(out / "results.json", dict(metadata=manifest, results=rows))
        csv_rows(out / "summary.csv", rows)
        (out / "report.md").write_text(
            "# E2 smoke passed\n\nAll new mappings: first 128 calibration tokens, three scales, E0 and three W+A proxies finite. No full search.\n",  # noqa: E501
            encoding="utf-8",
        )
    elif args.stage == "e2-search":
        import json

        gate = json.loads((args.smoke / "status.json").read_text(encoding="utf-8"))
        if (
            gate.get("stage") != "e2-smoke"
            or gate.get("status") != "complete"
            or gate.get("cache_sha256") != file_hash(args.cache)
        ):
            raise ValueError("Search requires a passing smoke of this exact cache")
        if (
            gate["git_commit"] != manifest["git_commit"]
            or gate["provenance"] != provenance()
        ):
            raise ValueError("Smoke/search code or runtime changed")
        if any(
            len(g["calibration"]) != 2048 or len(g["validation"]) != 2048
            for g in cache["groups"].values()
        ):
            raise ValueError("Formal search requires 2048/2048 tokens")
        found = search(
            cache,
            SimpleNamespace(
                device=args.device,
                batch=args.batch,
                rows=args.rows,
                proxies=["INT8", "MXFP4", "MXFP8"],
            ),
        )
        torch.save(found, out / "search.pt")
        write_json(out / "search.json", found)
        csv_rows(
            out / "summary.csv",
            [
                dict(
                    mapping=n,
                    proxy=r["proxy"],
                    objective=r["objective"],
                    alpha=r["alpha"],
                    beta=r["beta"],
                    loss=r["loss"],
                    seconds=r["search_seconds"],
                )
                for n, g in found["groups"].items()
                for r in g["runs"]
            ],
        )
        (out / "report.md").write_text(
            "# E2 search complete\n\nSix original search arms per mapping, unchanged coordinate grid and losses. See search.json for all candidates and scales.\n",  # noqa: E501
            encoding="utf-8",
        )
    elif args.stage == "e2-evaluate":
        found = torch.load(args.search, map_location="cpu", weights_only=True)
        evaluate(cache, found, args)
        import json

        ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
        if not all(
            r["e0"]["pass"] for g in ev["groups"].values() for r in g["results"]
        ):
            raise ValueError("Full selected-scale E0 failed; do not aggregate gains")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "stage",
        choices=[
            "p1",
            "e2-capture",
            "e2-smoke",
            "e2-search",
            "e2-evaluate",
            "e2-analyze",
        ],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--old-cache", type=Path)
    p.add_argument("--search", type=Path)
    p.add_argument("--smoke", type=Path)
    p.add_argument("--old", type=Path)
    p.add_argument("--new", type=Path)
    p.add_argument("--model-path")
    p.add_argument("--data", type=Path)
    p.add_argument("--offload-dir", type=Path)
    p.add_argument("--max-memory", type=__import__("json").loads)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--cache-mib", type=float, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (
        min(
            args.batch,
            args.rows,
            args.cache_mib,
            args.threads,
            args.tokens,
            args.context,
        )
        <= 0
    ):
        p.error("All sizes must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    with stage(args) as (out, manifest):
        if args.stage == "p1" and not args.device.startswith("npu"):
            raise ValueError("P1 requires explicit --device npu:0")
        execute(args, out, manifest)


if __name__ == "__main__":
    main()
