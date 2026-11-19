# Candidate-local activation reuse benchmark (2026-11-19)

This module is isolated from core.py and the normal capture/search/evaluate CLI.
No production Modifier changes, GLM loading, capture, full search or custom kernels.
`benchmark_reuse.assess(..., optimized=False/True)` selects the diagnostic variant.
Headline Baseline calls the existing `Objective.assess()` directly. The optimized
variant caches activation token batches on the execution device for ONE candidate.

For R=ceil(output_rows/rows), T=ceil(tokens/batch), each candidate has:

| Operation | Baseline | Optimized |
|---|---:|---:|
| Activation CPU scaling + `.to(device)` | R*T | T |
| Activation QDQ (W+A) | R*T | T |
| Activation QDQ (W-only) | 0 | 0 |
| Weight CPU scaling + `.to(device)` + QDQ | R | R |
| GEMM and output `.cpu()` | R*T | R*T |
| Explicit synchronization inside uninstrumented assess | 0 | 0 |

With batch=128, rows=256: Small R=2/T=1 (A QDQ 2→1), Medium R=8/T=4
(32→4), Large R=16/T=8 (128→8). Device-to-CPU output transfers implicitly
block on accelerator; CPU `.to(cpu)` and `.cpu()` are no-ops. Finite checks and
three scalar `.item()` extractions per output tile run on CPU. Reference GEMMs
and maxima are computed once outside candidate timing; candidate scales are also
prepared once for both variants. There are no cross-candidate transformed caches.

The optimization preserves original CPU BF16 scaling, CT per-token/per-row
32-element MXFP blocks, output GEMM tile shapes/order, and CPU FP64 subtraction,
square, sum and scalar accumulation order. Under deterministic QDQ/GEMM, this
preserves each candidate's loss and hence the existing two-stage <= tie-breaking
selection. Measurements check only the three representative pairs (0,1),
(0.5,0.5), (1,0); neither a full final alpha/beta nor a 42-candidate search is
claimed. Tests also check a repeated candidate after another scale, partial
final batches/slabs, all proxies/objectives and the all-zero INT8 nonfinite case.

Cache payload: Small 0.125 MiB; Medium 2 MiB; Large 8 MiB. It is built lazily,
one batch at a time during the first weight slab. Default --cache-mib=64 is a
hard payload limit; larger requests fail with required bytes, rather than silently
switch loops/reductions or spill. Raise the limit only with available memory.
QDQ temporaries are still batch/slab bounded. This explicitly trades up to one
activation matrix's memory for fewer QDQ calls and input copies, while preserving
weight-QDQ count. No caching of reconstructed outputs or all candidate results.

Run from work/llm-compressor in the existing environment:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experiments/flexsmooth_proxy/test_benchmark_reuse.py -q -p no:cacheprovider
python -m experiments.flexsmooth_proxy.benchmark_reuse \
  --sizes Small Medium --threads 1 --batch 128 --rows 256 \
  --output /path/to/new/flex-reuse-cpu
# Optional higher-cost synthetic case, not needed for this first diagnosis:
python -m experiments.flexsmooth_proxy.benchmark_reuse \
  --sizes Large --output /path/to/new/flex-reuse-large
```

Each case uses seed 42 and BF16. Each version has one warmup sweep, then three
un-instrumented sweeps with alternating version order; median seconds/candidate
is the primary timing. A separate set of three instrumented sweeps collects
section medians and actual operation counts. The optimized uninstrumented path
has lightweight Python counters/null contexts but no section clocks or extra
synchronization. Profiled synchronization overhead and dispatch overhead mean
section sums need not equal primary timing. Activation/weight sections include
CPU scaling, input transfer, CT scale calculation and QDQ. Output section includes
copy, FP64 conversion, finite checks and error/energy/max statistics. It does not
separately isolate pure data-transfer time. Report and JSON state these limits.
CPU peak RSS/device allocator peaks are lifetime counters, not per-version peaks;
cache payload provides the explicit incremental memory budget.

Only on an environment that already has a local NPU, an optional existing-cache
run can use the same machinery without model loading or a new capture:

```bash
python -m experiments.flexsmooth_proxy.benchmark_reuse --device npu:0 \
  --cache /mnt/share/t50065065/results/flexsmooth/full/capture/cache.pt \
  --mapping model.layers.3.self_attn.q_a_layernorm \
  --output /path/to/new/flex-reuse-npu
```

Use `model.layers.3.input_layernorm` instead if its cached tensors are BF16. Mixed
consumer groups can have FP32 promoted caches; the benchmark refuses implicit
casting. This optional command also evaluates only three candidate scales. No
remote connection or NPU invocation was attempted on this CPU-only Windows host.
CPU speedups and ratios are not estimates of NPU performance.

Outputs: report.md (summary and diagnostic table), results.json (all candidate
NMSEs/rankings, three timing samples, counters, sections, memory and provenance),
summary.csv. Output directory must be new. Existing experimental results are
never overwritten. Local results: workspace experiments/flexsmooth-reuse-20261119.

Branch verified first: flexsmooth-experiments, base
b6bc19c1d0eb43a50b470cd41d59fef87da8e88b. Reference/work fetch attempts failed due to
SSH/DNS restrictions; not synced. Pinned upstream/main remains
f8a3abb5358765a1864783deb3e38d3e2452912a. QuaRot experiments/results untouched.

## Local measured CPU result

7 new tests passed; Small and Medium each covered all three proxies, both
objectives and both versions, three representative scales, warmup and three
repetitions. Every measured NMSE difference was exactly zero and rankings agreed.
Large was skipped to keep diagnosis lightweight (Medium MXFP4 W+A baseline was
already about one second/candidate). Local torch is CPU-only; NPU skipped.

| Medium proxy | W+A baseline ms | W+A optimized ms | Speedup | W+A/W-only before | after |
|---|---:|---:|---:|---:|---:|
| INT8 | 280.542 | 278.204 | 1.01x | 1.03 | 1.03 |
| MXFP4 | 1001.809 | 598.041 | 1.68x | 2.04 | 1.22 |
| MXFP8 | 476.034 | 369.367 | 1.29x | 1.39 | 1.08 |

Medium activation QDQ calls dropped 32→4; weight QDQ remained 8 and GEMM 32.
Small activation QDQ dropped 2→1, W+A speedups 1.02x/1.15x/1.09x respectively.
Weight-only timings were essentially unchanged (Small includes slight overhead).
CPU output/loss is 1.7%–5.3% of optimized W+A instrumented time across cases:
low priority for further CPU work. NPU transfer/reduction must be measured there,
not inferred from these CPU results or compared directly to the reported 3.0–3.4x.
