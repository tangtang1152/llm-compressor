# Recipe-correct MXFP8 E2 completion

E3 is cancelled for this round. Scope: L6/16/60/77; input_layernorm (corrected),
q_a_layernorm (audited reuse), OV/o_proj (incremental capture). No MXFP4, Experts,
QuaRot, model export, GPQA or whole P1 benchmark. Old artifacts remain read-only.

## Implementation

- `run.py audit` (internal function) reads only named consumer weights from the
  original checkpoint safetensors and checks exact cache/checkpoint agreement.
  Native dtypes are recorded. This does not instantiate the model.
- Input objective keeps q_a_proj and kv_a_proj_with_mqa. Both original FP32-promoted
  weights and activations must round-trip exactly to their native target dtype.
  The complete compensation list still contains wk/weights_proj. FP32 indexer
  behavior is preserved. Mixed target dtypes and bias-bearing consumers fail
  explicitly instead of being silently concatenated into a different objective.
  Statistics for this corrected objective use the selected quantized weights;
  input scales therefore belong to the new objective and cannot be relabeled as
  old E2 scales. Full-group compensation remains a separate graph contract.
- QA consumers must match the full old group, dtype, tensors, reference policy,
  search/QDQ source and runtime. Scales are read directly from old search.pt.
  Unknown server source revisions or changed runtime require audit; no implicit
  re-search or relabeling of old results. Known old core SHA is accepted because
  the only core change extracts `applied_scale()` without changing its behavior.
- `RecipeObjective.search` inherits the entire old 21-alpha + 21-beta search,
  later-tie-wins, reference and raw candidate formula. Only assess dispatch uses
  `benchmark_reuse.assess`, with a candidate-local bounded cache. Old E1/E2 entry
  points remain baseline; this directory is the new formal recipe search path.
- OV uses original o_proj X/W. GLM's expanded heads are resolved by glm_mappings;
  the production search uses raw channel statistics and applies `ov_scales` after
  choosing alpha/beta with heads=kv_heads. We preserve both raw-candidate and
  applied-scale diagnostics. kv_b_proj never receives QDQ. Native tiny Attention
  tests verify only Value rows change, Key rows/top-k stay unchanged, and the
  output compensation agrees (including non-power-of-two scales).
- Validation always uses MXFP8 W+A for No Smooth/W-only/W+A. Results include all
  candidates, alpha/beta, scales, native dtype, full compensation and QDQ lists.
  Generic local no-QDQ NMSE must be finite and <=1e-3 before publishing; native
  tiny OV tests use tighter bounds. These are local tests, not E3 Block claims.

## Full-search A/B

Each objective: one complete warmup per version, three alternating-order measured
repetitions by default. Baseline calls the original Objective.assess; Optimized
calls the tested P1 implementation. Every candidate loss and coordinate, selected
alpha/beta and applied scale must match exactly. A tiny difference is a failure,
even if it looks harmless; inspect ranking/ties and dtype rather than relax it.

Search seconds cover 42 candidates, excluding reference initialization. End-to-end
includes initialization and inherited selected-raw/applied calibration diagnostics
(two additional assessments). Per-version reference/end-to-end times and memory
are saved. A separate instrumented replay of all 42 candidates records actual QDQ
calls and activation/weight/GEMM/output-loss timing, excluded from headline timing.
Cache bytes report activation payload, not total allocator peak. Device name is
recorded; CPU tests must not be presented as Ascend A5 measurements.

## Server sequence (not executed on this Windows/CPU host)

Inspect actual branch and installed sources before integrating changes. This host
is on flexsmooth-experiments, not proof of the server branch. No server execution
connection or real E2 artifacts were available locally. Fetch failed; not synced.

```bash
cd /mnt/share/t50065065/flexsmooth/llm-compressor
git branch --show-current
git status --short
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1  # only idle devices; check npu-smi first
OLD=/mnt/share/t50065065/results/flexsmooth/next/layer_extension/distributed/20261120-2rank-r2/formal
MODEL=/mnt/weight/GLM-5.2
DATA=/mnt/share/datasets/qwen3_cot_w4a4.json
NEW=/mnt/share/t50065065/results/flexsmooth/next/recipe_mxfp8/$(date +%Y%m%d-%H%M%S)
mkdir -p "$NEW/logs"
python -m pytest experiments/flexsmooth_proxy/recipe_mxfp8/test_recipe.py -q -p no:cacheprovider

# No model loading: corrected L6 input objective, complete search A/B.
python -m experiments.flexsmooth_proxy.recipe_mxfp8.run ab \
 --cache "$OLD/capture/cache.pt" --checkpoint "$MODEL" \
 --mapping model.layers.6.input_layernorm --device npu:0 \
 --batch 128 --rows 256 --cache-mib 64 --output "$NEW/input-ab" \
 2>&1 | tee "$NEW/logs/input-ab.log"

# Corrected input searches; QA reuse is validated against original search.pt.
python -m experiments.flexsmooth_proxy.recipe_mxfp8.run complete \
 --cache "$OLD/capture/cache.pt" --checkpoint "$MODEL" \
 --old-search "$OLD/search/search.pt" --ab "$NEW/input-ab/parity.json" \
 --device npu:0 --batch 128 --rows 256 --cache-mib 64 --output "$NEW/norm" \
 2>&1 | tee "$NEW/logs/norm.log"

# One additional original-model load per rank is necessary: old cache lacks o_proj X.
# Same E2 split identities, only o_proj hooks, no repeat norm capture/L3 capture.
torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
 --log-dir "$NEW/logs/ov-capture" --redirects 3 --tee 3 \
 -m experiments.flexsmooth_proxy.e2_collect --ov-only \
 --model-path "$MODEL" --data "$DATA" --old-cache "$OLD/capture/cache.pt" \
 --output "$NEW/ov-capture" --offload-dir "$NEW/ov-offload"

python -m experiments.flexsmooth_proxy.recipe_mxfp8.run ab \
 --cache "$NEW/ov-capture/capture/cache.pt" --checkpoint "$MODEL" \
 --mapping model.layers.6.self_attn.kv_b_proj --device npu:0 \
 --batch 128 --rows 256 --cache-mib 64 --output "$NEW/ov-ab" \
 2>&1 | tee "$NEW/logs/ov-ab.log"
python -m experiments.flexsmooth_proxy.recipe_mxfp8.run complete \
 --cache "$NEW/ov-capture/capture/cache.pt" --checkpoint "$MODEL" \
 --ab "$NEW/ov-ab/parity.json" --device npu:0 \
 --batch 128 --rows 256 --cache-mib 64 --output "$NEW/ov" \
 2>&1 | tee "$NEW/logs/ov.log"
python -m experiments.flexsmooth_proxy.recipe_mxfp8.report \
 --norm "$NEW/norm" --ov "$NEW/ov" --norm-ab "$NEW/input-ab" --ov-ab "$NEW/ov-ab" \
 --output "$NEW/analysis"
```

Use `set -euo pipefail` for scripted execution. All output directories must be new.
If parity or QA reuse audit fails, preserve the output and diagnose before continuing.
The A/B acceptance record is the requested parity prerequisite, not another model
load gate. Existing e2_collect loader/3-hour collectives/offload are unchanged.

Final analysis contains p1_search_ab.csv/md, p1_candidate_parity.csv and
mxfp8_e2_completed.csv/md. Detailed JSON/PT retain parameters and all candidate
curves. Norm/OV aggregation checks identical token identities, seed and runtime.
Inspect sections_w/wa.json for bottlenecks; do not infer NPU speedup from CPU or
the previous three-candidate P1 run. No real A5 gains are available until execution.
