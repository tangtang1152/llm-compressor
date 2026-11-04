# E1.2: bounded Group attribution diagnostics

## Defaults and data contract

The server computes all group statistics for every measured gate/up matrix, but
**defaults to summaries only and produces no plots**. Formal B/C metrics retain
E1's schema. Raw is distribution-only. Fusion, B32 seed 1234, FP32 transform
arithmetic, BF16 write-back and CT MXFP4 QDQ remain unchanged. E0 is unchanged.

| Parameter | Meaning |
|---|---|
| `--group-detail-experts-per-layer 0` | Default: no group files |
| `--group-detail-experts-per-layer 2` | Save 2 deterministically chosen experts per layer |
| `--group-detail-experts-per-layer all` | Save every selected expert |
| `--preflight-only` | Validate headers, reuse compatibility, selections and disk budget without loading weights |
| `--threads 4` | CPU thread limit, default 4 |

Detail IDs are selected from the measured IDs using Random(1234+layer), before
observing weights/errors. All summary statistics and correlations use **all**
measured groups, regardless of detail mode. Later rerun selected layers into a
fresh directory to obtain optional details. This recomputes those layers' new
diagnostics; it does not read or rewrite earlier result directories.

## Resource design

For hidden=6144, intermediate=2048: 393216 groups per gate/up matrix. At 12 layers
x16 experts x2 projections, all statistics cover 150994944 paired groups. Group
vectors live only for the current matrix. FP64 elementwise group statistics are
chunked into at most 65536 groups, exact nearest-rank quantiles use NumPy partition
on current-matrix vectors, and correlations are immediately reduced to scalar
records. No corpus-sized arrays/DataFrames/Python float lists are retained.
QDQ runs once per Fused/Rotated stage; fused BF16 weights are reused for rotation.
Each measured matrix is loaded once. Config/index is read once and selections are
validated from cached metadata before the first matrix. Plots read summaries only.

Optional detail storage uses uncompressed `.npz` (26 FP64 group vectors, original
shape, and gamma CV vector). It avoids per-row CSV conversion and gzip CPU cost.
Approximate numeric payload: default 0 bytes; 2 experts/layer 3.66 GiB; all 16
29.25 GiB. Additional NPZ headers and summaries are small. Each detail file is
written to `.partial` and renamed on completion. No lossy float32 conversion.

Preflight prints actual group counts and disk bytes from checkpoint dimensions
and resolved expert selections. Disk checks reserve 10% above detail payload plus
16 MiB for summaries. A conservative working-array estimate is 100 bytes per
matrix element (1.17 GiB at 6144x2048), **plus Python/PyTorch, allocator retention,
OS page cache and file mappings**. This is a planning estimate, not a hard RSS cap.
Existing exact E1 element percentiles still need current-matrix temporaries.
Allow at least 4 GiB free RAM initially and measure server peak RSS; checkpoint
shard mmap page-cache behavior and CPU/NPU backend performance are not locally
validated. CPU-only execution, no accelerator/full-model forward.

## Reuse and failure behavior

Pass every relevant old `results.json` explicitly through `--reuse-e1`. Existing
layer selections remain exactly their previous union; new layers use the original
Random(1234+layer) sampler and requested expert count. If historical layers had
fewer than 16 experts, the run intentionally preserves that count: inspect
preflight.json rather than assuming 12x16. Complete triplets and selected IDs,
config/index hash, full quantization policy, rotation settings, recorded source
hashes and package versions must agree. Git commit/source paths may differ.
Historical versions must be retained; do not upgrade CT to bypass a mismatch.
Only source files fingerprinted in historical E1 can be checked retrospectively;
new reports additionally fingerprint experiment source files.

Existing formal metrics are copied. Gate/up diagnostic QDQ is necessary for new
statistics; its NMSE is checked against history (rtol 1e-6, atol 1e-12). Existing
down rows skip loading/QDQ entirely. Missing formal rows use the E1 metric helper.
A differing quantizer, conflicting/missing rows or measured NMSE mismatch fails.

Output directories must be new. diagnostics.json starts as running; a complete
status is published atomically only after final outputs finish. Exceptions and
interrupts record failed status where the filesystem allows it. SIGKILL/power
loss can leave running/finalizing or .partial files, never a success marker.
Only status=complete is an experiment result. A failed summary explicitly says
failed. No auto-resume and no overwriting; use a new directory on retry.

## Safe Git synchronization

Windows: commit is new, not amended; publish normally with `git push origin
quarot-experiments`. On the server first inspect without changing working files:

```bash
git status --short
git fetch origin
git log --oneline --left-right HEAD...origin/quarot-experiments
```

If preserving server work, use a fresh checkout/worktree directory, leaving both
tracked modifications and untracked results untouched:

```bash
git worktree add --detach /mnt/share/t50065065/llmc-e12-run origin/quarot-experiments
cd /mnt/share/t50065065/llmc-e12-run
```

Alternatively, **only if local tracked changes/commits are explicitly disposable**:

```bash
git switch quarot-experiments
git reset --hard origin/quarot-experiments
```

Do not blindly pull divergent rewritten history. Never run git clean. Keep result
directories outside the checkout and do not delete untracked experiment files.

## Server preflight and execution

Use the historical Python environment, with current checkout's source first on
PYTHONPATH. No matplotlib is needed for the server run. Set prior paths to actual
existing JSON files; filenames below are placeholders, not discovered results.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import llmcompressor,compressed_tensors; print(llmcompressor.__file__); print(compressed_tensors.__file__)"
git rev-parse HEAD
free -h
df -h /mnt/weight/GLM-5.2 /mnt/share/t50065065/results/quarot
prior=(/absolute/path/to/previous/results.json /absolute/path/to/other/results.json)
run=/mnt/share/t50065065/results/quarot/e12-$(date +%Y%m%d-%H%M%S)
python -m experiments.quarot_mxfp4.diagnose \
  --checkpoint /mnt/weight/GLM-5.2 --reuse-e1 "${prior[@]}" \
  --layers 3 4 5 6 7 8 9 10 12 16 40 77 --experts-per-layer 16 \
  --group-detail-experts-per-layer 0 --threads 4 \
  --preflight-only --output "${run}-preflight"
# Continue only after preflight passes and selections, fingerprints and RAM fit.
/usr/bin/time -v python -m experiments.quarot_mxfp4.diagnose \
  --checkpoint /mnt/weight/GLM-5.2 --reuse-e1 "${prior[@]}" \
  --layers 3 4 5 6 7 8 9 10 12 16 40 77 --experts-per-layer 16 \
  --group-detail-experts-per-layer 0 --threads 4 --output "$run" \
  >"${run}.log" 2>&1
python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete"; n=sum(map(len,r["selection"].values())); assert len(r["e1_rows"])==3*n; assert len(r["matrices"])==6*n; assert len(r["comparisons"])==2*n; print(n,"experts",r["budget"],r["elapsed_seconds"])' "$run/diagnostics.json"
```

Start with `--layers 6 --experts-per-layer 1` in a separate output directory to
measure real per-matrix time/RSS before the full selection. Choose a new layer
without reused selections; existing layer IDs intentionally override count.
No real server run has been executed by Codex.

## Outputs and local plots

- preflight.json: validated selection/detail selection, provenance and resource budget.
- diagnostics.json: status, elapsed time, configuration, source hashes, selections,
  budget, detail file manifest, all summaries/correlations and formal rows.
- gamma.csv, matrices.csv, comparisons.csv: gamma/three-stage/paired matrix statistics.
- group_correlations.csv: correlations over every group per matrix (not disk samples).
- matrix_correlations.csv: within-layer and pooled per-projection associations.
- e1_rows.csv: compatible formal B/C metrics; summary.md: table and interpretation limits.
- groups-L*-E*-*.npz: optional complete paired vectors. Use `np.load(...,
  allow_pickle=False)`; `shape` gives matrix shape. For flattened group index i,
  row=i//(shape[1]//32), input_group=i%(shape[1]//32). gamma_abs_cv is indexed by
  input_group. The manifest gives group count and actual bytes.

Copy only diagnostics.json for local plotting (or the CSVs for other analysis):

```bash
python -m experiments.quarot_mxfp4.report --input /local/diagnostics.json --output /local/new-figures
```

Local plotting needs matplotlib. It rejects incomplete runs and existing output
directories. Generates layer_trends.png and group_error_distribution.png from
matrix summaries only. Distribution figure shows each matrix's exact Group
P50/P90/P95/P99, not pooled raw-group histograms.

Group RMS=sqrt(mean(w^2)), Max/other-RMS removes one maximum element. Gamma CV uses
population std(abs(gamma))/mean(abs(gamma)). Zero/degenerate denominators use 1e-30;
constant/too-small correlations return null, nonfinite inputs fail. Scale log2
is decoded CT scale, not uint8 storage bytes. Matrix NMSE is sum(SSE)/sum(energy),
not the mean of group-relative errors. Correlations are descriptive: repeated
gamma/groups/experts are dependent, and no causal or W4A4 conclusion follows.

## Validation performed on 2026-11-04

- Relevant pytest suite: 16 passed (13.07 s); Ruff and git diff --check passed.
- Detail modes 0/2/all produce identical full-group summaries and correlations;
  deterministic selection, mixed historical/new layers, strict config/version/
  source checks, missing rows, invalid inputs, existing directory rejection,
  injected mid-run failure and preflight-only mode covered.
- Reproducible synthetic complete run: 12 layers x16 experts, hidden=128,
  intermediate=64, 12 shards, 98304 gate/up groups; 576 formal matrix rows,
  1152 stage summaries; no group files and no plots. Elapsed 13.72 seconds,
  sampled process-tree peak RSS 439836672 bytes (~419 MiB), output 4626053 bytes
  (~4.41 MiB). Child-process RSS was included (Windows Python launcher alone is
  not a valid measurement). Sampling interval 100 ms; brief peaks may be missed.
- Separate realistic-size synthetic single gate matrix [2048,6144]: Raw/Fused/
  Rotated transformations, native QDQ, original formal metrics, chunked group
  statistics, quantiles and paired correlation all completed (393216 groups).
  Elapsed 11.14 s including process startup; sampled process-tree peak RSS
  1199128576 bytes (~1.12 GiB), sampled at 50 ms. This is a matrix-path check,
  not a full 12x16 run at real dimensions or a real-checkpoint benchmark.
- Both checks used local Windows CPU and four torch threads. No real checkpoint
  was searched/loaded; server I/O, runtime, complete-model topology and historical
  server-result compatibility remain unverified until server preflight/run.

Reproduce the full synthetic capacity check without generating plots (requires
psutil only for this validation utility):

```bash
python -m experiments.quarot_mxfp4.validate_scale --output experiments/quarot_mxfp4/.test-data/new-capacity-check
```

The utility writes generated shards, execution.log, validation.json and output/
under the new directory. It checks row counts, success status and absence of group
files. Optional --layers/--experts/--hidden/--intermediate control validation size;
no real model is involved. Small dimensions validate record counts and end-to-end
flow; they do not establish full-size throughput.
