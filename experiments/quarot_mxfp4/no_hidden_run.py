"""Prepare the archived production driver; inspect No-Hidden artifacts by headers."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from safetensors import safe_open

from .diagnose import atomic_json

LOCAL = ["rot_b_proj", "rot_uv", "rot_kv_b_proj"]
HISTORY = "97eabf56"


def inventory(root):
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    total = 0
    for name in sorted(set(index.values())):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("unsafe shard path")
        total += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if keys != {k for k, v in index.items() if v == name}:
                raise ValueError(f"index/file keys mismatch: {name}")
            for key in keys:
                value = f.get_slice(key)
                tensors[key] = (value.get_shape(), value.get_dtype())
    if {p.name for p in root.glob("*.safetensors")} != set(index.values()):
        raise ValueError("unindexed safetensors shard")
    return config, tensors, total


def verify(full, candidate):
    a, at, ab = inventory(full)
    b, bt, bb = inventory(candidate)
    meta = b.get("quarot_config", {})
    if meta.get("status") != "applied" or meta.get("enabled_rotation_spaces") != LOCAL:
        raise ValueError("candidate must record exactly three local spaces")
    old = a.get("quarot_config", {})
    if old.get("status") != "applied" or set(
        old.get("enabled_rotation_spaces", ["rot", *LOCAL])
    ) != {"rot", *LOCAL}:
        raise ValueError("baseline must be applied Full QuaRot")
    for key in ("seed", "block_size", "precision", "topology"):
        if meta.get(key) != old.get(key):
            raise ValueError(f"QuaRot configuration mismatch: {key}")
    for key in ("quantization_config", "flex_smooth_config"):
        if not a.get(key) or a[key] != b.get(key):
            raise ValueError(f"missing/changed {key}")
    ignored = {"quarot_config", "_name_or_path", "transformers_version"}
    if {k: v for k, v in a.items() if k not in ignored} != {
        k: v for k, v in b.items() if k not in ignored
    }:
        raise ValueError("other model configuration changed")
    if at != bt:
        raise ValueError("tensor key/shape/dtype layout differs from Full baseline")
    return dict(
        status="header_validation_passed",
        full_bytes=ab,
        no_hidden_bytes=bb,
        tensor_count=len(bt),
        quantization_config_equal=True,
        limitation=(
            "Headers/layout only; driver reloads representative MXFP4/8 tiles. "
            "Server load and GPQA still required."
        ),
    )


def prepare(args):
    source = Path(args.checkpoint)
    config = json.loads((source / "config.json").read_text())
    if config.get("quarot_config") or config.get("quantization_config"):
        raise ValueError("requires original BF16 checkpoint")
    if any(p.exists() for p in (args.run_dir, args.output, args.offload_dir)):
        raise FileExistsError("run, output and offload directories must be new")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    baseline_config, _, baseline_bytes = inventory(args.baseline_full)
    meta = baseline_config.get("quarot_config", {})
    if meta.get("status") != "applied" or set(
        meta.get("enabled_rotation_spaces", ["rot", *LOCAL])
    ) != {"rot", *LOCAL}:
        raise ValueError("requires an applied Full QuaRot baseline")
    if any(
        meta.get(k) != v
        for k, v in dict(seed=1234, block_size=32, precision="float32").items()
    ):
        raise ValueError("baseline does not match historical B32/1234/FP32 recipe")
    if not baseline_config.get("quantization_config") or not baseline_config.get(
        "flex_smooth_config"
    ):
        raise ValueError("baseline quantization/FlexSmooth metadata missing")
    parent = args.output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < 2 * baseline_bytes:
        raise ValueError(
            "output filesystem needs at least twice baseline checkpoint size free"
        )
    repo = Path(__file__).resolve().parents[2]

    def history(name):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                f"{HISTORY}:examples/glm52_precision/{name}",
            ]
        ).decode()

    original = history("oneshot_profile.py")
    needle = "        recipe[0].precision = torch.float32\n"
    if original.count(needle) != 1:
        raise ValueError("unexpected historical driver; inspect before adaptation")
    adapted = original.replace(
        needle,
        needle + f"        recipe[0].enabled_rotation_spaces = {tuple(LOCAL)!r}\n",
    )
    # Historical git provenance assumed examples/<dir>/<script>; the exported
    # snapshot resides in a run directory, so point that query at this checkout.
    adapted = adapted.replace(
        "cwd=Path(__file__).resolve().parents[2],", f"cwd={str(repo)!r},"
    )
    recipe = history("mixed_mxfp.yaml")
    args.run_dir.mkdir(parents=True)
    driver = args.run_dir / "oneshot_profile.py"
    driver.write_text(adapted, encoding="utf-8")
    (args.run_dir / "mixed_mxfp.yaml").write_text(recipe, encoding="utf-8")
    command = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=8",
        str(driver),
        "--model",
        str(source),
        "--output",
        str(args.output),
        "--device",
        "npu",
        "--device-map",
        "auto_offload",
        "--dataset",
        str(args.dataset),
        "--samples",
        "42",
        "--sequence-length",
        "2688",
        "--batch-size",
        "1",
        "--targets-per-subgraph",
        "1",
        "--offload-dir",
        str(args.offload_dir),
        "--report-dir",
        str(args.run_dir / "profile"),
    ]
    if args.max_memory:
        json.loads(args.max_memory)
        command += ["--max-memory", args.max_memory]
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(repo)) + "\n"
    )
    for path in (args.output, args.offload_dir):
        script += "test ! -e " + shlex.quote(str(path)) + "\n"
    script += (
        "python -m pytest experiments/quarot_mxfp4/test_no_hidden.py "
        "-q -p no:cacheprovider\n"
    )
    script += (
        shlex.join(command)
        + " 2>&1 | tee "
        + shlex.quote(str(args.run_dir / "quantize.log"))
        + "\n"
    )
    script += (
        shlex.join(
            [
                "python",
                "-m",
                "experiments.quarot_mxfp4.no_hidden_run",
                "verify",
                "--baseline-full",
                str(args.baseline_full),
                "--candidate",
                str(args.output),
                "--report",
                str(args.run_dir / "artifact_validation.json"),
            ]
        )
        + "\n"
    )
    (args.run_dir / "quantize.sh").write_text(script, encoding="utf-8")
    atomic_json(
        args.run_dir / "preparation.json",
        dict(
            status="prepared_not_executed",
            history=HISTORY,
            original_driver_sha256=hashlib.sha256(original.encode()).hexdigest(),
            adapted_driver_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
            recipe_sha256=hashlib.sha256(recipe.encode()).hexdigest(),
            arguments={k: str(v) for k, v in vars(args).items()},
            output_free_bytes=free,
            baseline_bytes=baseline_bytes,
            warning=(
                "Historical driver remounts /dev/shm to1200G after loading. "
                "Requires original mount privileges/capacity. "
                "Check offload disk and L4 gate before execution."
            ),
        ),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare")
    for name in ("checkpoint", "baseline-full", "run-dir", "output", "offload-dir"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument(
        "--dataset", type=Path, default=Path("/mnt/share/datasets/qwen3_cot_w4a4.json")
    )
    q.add_argument("--max-memory")
    q = sub.add_parser("verify")
    for name in ("baseline-full", "candidate", "report"):
        q.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a)
    else:
        if a.report.exists():
            raise FileExistsError(a.report)
        try:
            report = verify(a.baseline_full, a.candidate)
        except Exception as exc:
            atomic_json(a.report, dict(status="failed", error=repr(exc)))
            raise
        atomic_json(a.report, report)


if __name__ == "__main__":
    main()
