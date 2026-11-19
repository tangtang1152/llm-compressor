"""Run records and conservative compatibility checks for existing artifacts."""

import contextlib
import csv
import hashlib
import subprocess
import traceback
from pathlib import Path

import torch

from ..core import provenance, validate_cache
from ..run import cache_hashes, write_json

ROOT = Path(__file__).resolve().parents[3]


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_cache(path, real=True):
    cache = validate_cache(torch.load(path, map_location="cpu", weights_only=True))
    if real and cache.get("kind") != "real":
        raise ValueError("This stage requires real captures; synthetic is tests only")
    return cache


def source_signature(p):
    # Absolute installation paths are not a compatibility property.
    return {Path(k.replace("\\", "/")).name: v for k, v in p.get("sources", {}).items()}


def runtime_signature(p):
    return dict(
        torch=p.get("torch"),
        compressed_tensors=p.get("compressed_tensors"),
        sources=source_signature(p),
    )


def identities(cache):
    groups = list(cache["groups"].values())
    first = {s: groups[0][s + "_ids"] for s in ("calibration", "validation")}
    if any(g[s + "_ids"] != first[s] for g in groups for s in first):
        raise ValueError("Mappings do not share identical ordered token identities")
    return first


def compatibility(old, new, old_search=None, new_search=None):
    reasons = []
    if identities(old) != identities(new):
        reasons.append("ordered calibration/validation token identities differ")
    if old.get("seed") != new.get("seed"):
        reasons.append("seed differs")
    for label, a, b in [("capture", old, new), ("search", old_search, new_search)]:
        if a is None or b is None:
            reasons.append(f"{label} provenance absent")
            continue
        if not a.get("provenance") or not b.get("provenance"):
            reasons.append(f"{label} provenance absent")
        elif any(
            not p.get(k)
            for p in (a["provenance"], b["provenance"])
            for k in ("torch", "compressed_tensors", "sources")
        ):
            reasons.append(f"{label} incomplete runtime/source evidence")
        elif runtime_signature(a["provenance"]) != runtime_signature(b["provenance"]):
            reasons.append(f"{label} quantization/search source or runtime differs")
    # A recorded identical seed alone does not establish preprocessing identity.
    for key in ("capture_source_sha256",):
        if not old.get(key) or not new.get(key):
            reasons.append(f"{key} unavailable: preprocessing equivalence unverified")
        elif old[key] != new[key]:
            reasons.append(f"{key} differs")
    if old_search and new_search:
        if old_search.get("reference_policy") != new_search.get("reference_policy"):
            reasons.append("reference/loss policy differs")
    return dict(
        strictly_comparable=not reasons,
        reasons=reasons,
        grouping="paired layer extension"
        if not reasons
        else "separate experimental batches",
    )


def check_search(cache, found):
    if set(cache["groups"]) != set(found["groups"]):
        raise ValueError("Search/cache mapping coverage differs")
    if cache_hashes(cache) != found.get("hashes"):
        raise ValueError("Search and original cache hashes differ")
    for name, group in found["groups"].items():
        runs = group["runs"]
        keys = {(r["proxy"], r["objective"]) for r in runs}
        if len(runs) != 6 or keys != {
            (p, o) for p in ("INT8", "MXFP4", "MXFP8") for o in ("W-only", "W+A")
        }:
            raise ValueError(f"{name}: missing/duplicate search arms")


def csv_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def stage(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        stage=args.stage,
        status="running",
        git_commit=git_head(),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        seed=getattr(args, "seed", 42),
        provenance=provenance(),
        experiment_sources={
            str(p.relative_to(ROOT)): file_hash(p)
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    write_json(out / "status.json", manifest)
    try:
        if args.expected_commit != manifest["git_commit"]:
            raise ValueError(
                "Checkout commit differs from --expected-commit; synchronize explicitly"
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                "experiments/flexsmooth_proxy",
            ],
            check=True,
            capture_output=True,
        )
        yield out, manifest
        if manifest["status"] == "running":
            manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        (out / "failure.md").write_text(
            f"# {args.stage} failed\n\n{exc}\n\nNo automatic retry or changed batching. Preserve this attempt; retry with explicit smaller --batch/--rows in a new directory if OOM.\n",  # noqa: E501
            encoding="utf-8",
        )
        csv_rows(out / "failure.csv", [dict(status="failed", reason=repr(exc))])
        raise
    finally:
        write_json(out / "status.json", manifest)
