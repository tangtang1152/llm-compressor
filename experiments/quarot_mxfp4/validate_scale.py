"""Reproducible synthetic end-to-end capacity check, no model checkpoint."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--layers", type=int, default=12)
parser.add_argument("--experts", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--intermediate", type=int, default=64)
a = parser.parse_args()
if (
    min(a.layers, a.experts, a.hidden, a.intermediate) < 1
    or a.hidden % 32
    or a.intermediate % 32
):
    parser.error("positive dimensions/counts required, dimensions divisible by 32")
root = a.output
root.mkdir(parents=True, exist_ok=False)
config = dict(
    _experiment_synthetic=True,
    num_hidden_layers=a.layers,
    n_routed_experts=a.experts,
    hidden_size=a.hidden,
    moe_intermediate_size=a.intermediate,
    first_k_dense_replace=0,
    hidden_act="silu",
)
(root / "config.json").write_text(json.dumps(config))
rng = torch.Generator().manual_seed(1234)
index = {}
for layer in range(a.layers):
    tensors = {
        f"model.layers.{layer}.post_attention_layernorm.weight": (
            torch.rand(a.hidden, generator=rng) + 0.5
        ).bfloat16()
    }
    for expert in range(a.experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shape = (
                (a.hidden, a.intermediate)
                if proj == "down_proj"
                else (a.intermediate, a.hidden)
            )
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"] = (
                torch.randn(*shape, generator=rng).bfloat16()
            )
    shard = f"layer-{layer}.safetensors"
    save_file(tensors, str(root / shard))
    index.update({k: shard for k in tensors})
(root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=index)))
cmd = [
    sys.executable,
    "-m",
    "experiments.quarot_mxfp4.diagnose",
    "--checkpoint",
    str(root),
    "--output",
    str(root / "output"),
    "--layers",
    *map(str, range(a.layers)),
    "--experts-per-layer",
    str(a.experts),
    "--threads",
    "4",
]
started = time.perf_counter()
with (root / "execution.log").open("w") as log:
    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    proc = psutil.Process(child.pid)
    peak = 0
    while child.poll() is None:
        try:
            peak = max(
                peak,
                sum(
                    p.memory_info().rss
                    for p in [proc, *proc.children(recursive=True)]
                    if p.is_running()
                ),
            )
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.1)
assert child.returncode == 0, (root / "execution.log").read_text()
r = json.loads((root / "output/diagnostics.json").read_text())
assert (
    r["status"] == "complete"
    and len(r["e1_rows"]) == 3 * a.layers * a.experts
    and len(r["matrices"]) == 6 * a.layers * a.experts
)
assert not list((root / "output").glob("groups-*"))
summary = dict(
    layers=a.layers,
    experts_per_layer=a.experts,
    hidden=a.hidden,
    intermediate=a.intermediate,
    formal_matrices=len(r["e1_rows"]),
    stage_summaries=len(r["matrices"]),
    elapsed_seconds=time.perf_counter() - started,
    peak_rss_bytes=peak,
    output_bytes=sum(p.stat().st_size for p in (root / "output").iterdir()),
    groups=r["budget"]["groups_all_gate_up"],
)
(root / "validation.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
