# FlexSmooth E0/E1: local mapping proxy ablation

Scope: only original floating norm-linear mapping tensors, E0 and E1. No
QuaRot, OV, expert cache, weight export, full-model quantization, Block or GPQA.
Layer indices are zero-based (3 and 40). This experiment does not change the
production Modifier or claim actual MX kernels have been validated.

## Source audit (2026-11-16)

Branch checked first: `flexsmooth-experiments`, HEAD
`824b9c37030c8413c551b0d5bd34cc8cc6917dca`. User's current branch instruction
supersedes the older workspace branch preference. All three fetches failed
(network proxy / Git signal-pipe restriction); **not synced**. Pinned local
upstream/main: `f8a3abb5358765a1864783deb3e38d3e2452912a`;
reference llm-compressor: `de16f385705239c10bc54add79d4881242ddbb8c`;
ModelSlim: `1c6391745db71912bb145e7f139db1cff8cad9e1`.
No reference sources were edited or copied.

- `search.py`: candidate S = max(abs(X), tokens)^alpha *
  max(abs(W), output_rows)^(-beta), in storage dtype, without clamps.
  Applied S reuses `smooth_scale`: W maxima FP32-clamped to 1e-5 then cast back;
  ratio computed in storage dtype, FP32-clamped to 1e-5 then cast back.
- `_search_candidates` is reused directly: 21 ascending alpha candidates,
  beta=1-alpha; 21 ascending beta candidates with fixed alpha. `<=` means later
  ties win. Stage two retains stage one's loss threshold (and the production
  default beta=0 behavior). Nonfinite candidates cannot win. All-invalid raises.
  `selected_raw_loss` and `applied_calibration_loss` expose any discrepancy
  between recorded threshold, selected pair and clamped application.
- INT8 is production `_row_int8`: per-row absmax/127, round, clamp [-127,127],
  dequantize, all in storage dtype. All-zero rows produce nonfinite values,
  deliberately retained rather than silently changing the proxy.
- Production `reconstruction_loss` also squares/means in storage dtype.
  This experiment intentionally casts BEFORE error subtraction/squaring and
  accumulates CPU FP64. Thus it preserves search geometry and proxy QDQ, but
  does not promise bit-identical BF16 production selections. GEMMs retain the
  cache dtype, including BF16 reference GEMMs. Reference is cached per split.
- `base.py` concatenates every consumer weight along output rows. `mappings.py`
  includes q_a_proj, kv_a_proj_with_mqa and, when present, indexer.wk and
  indexer.weights_proj under input_layernorm. q_a_layernorm includes q_b_proj
  AND indexer.wq_b when present. Never split these into independent searches.
- `distributed.py` reduces maxima and moments across ranks, retains storage
  dtype square behavior. This experiment is single-process, not a new
  distributed calibration implementation.
- Installed compressed-tensors is **0.19.1a20260919** (site-packages, distinct
  from work/compressed-tensors HEAD a66733fcb18d52861ad4ab0e395f621577d51f02).
  It has usable `calculate_qparams` and `fake_quantize`; no custom MX adapter
  or copied rounding implementation is required. Each run records imported
  source paths, SHA256 fingerprints and version. Search/evaluate reject runtime
  changes. Both MX presets use groups of 32 on the LAST/input-channel axis,
  independently per weight row or activation token, symmetric zero-point 0,
  E8M0 power-of-two scale metadata. CT derives exponents using its bit-level
  `round_to_power_2`, offset 2 (E2M1) or 8 (E4M3), and exponent bias 127.
  Do not replace that rule with absmax/6, absmax/448 or an assumed ceil(log2).
  CT handles saturation to +/-6 or +/-448 and its FP4 / FP8 rounding.
  Non-multiple-of-32 widths fail explicitly, no invented padding rule.
- Inspected existing complete-layer capture (`work/l5-as-run/tools/`), existing
  profiling JSON/text/messages input conventions, and CT MX utils tests.
  Decoder-layer/MoE caches do not establish this cache's norm-linear contract.
  A new minimal input hook is therefore supplied, not an expert-cache adapter.

## Capture contract and resource limits

Use local existing baseline weights only. No downloads. `capture.capture()` can
be embedded in an existing baseline loader/offload driver: pass its eval model
and two lists of `(original_row, tokenized_document_sha256, token_ids)` records.
`split_records` accepts existing JSON list[str], JSON list[dict], or JSONL with
text/messages. It deduplicates tokenized documents, seeded-shuffles, then assigns
whole documents to alternating splits. Each selected document is an unpadded,
cache-free batch=1 prefill, capped at context length; final sample is truncated
to the exact requested token budget. Calibration/validation token identities
are disjoint document-hash/position pairs (not vocabulary token IDs). Insufficient
independent tokens is an error. The same cache serves every proxy and objective.
There is no automatic reuse of differently shaped older caches.

The default CLI loader now uses the QuaRot E2 path: `load_context()` with
`device_map="auto_offload"`, a required NEW `--offload-dir`, and optional
`--max-memory` JSON (numeric device keys become integers). CT manages CPU/disk
weight backing; `load_context` also enables the existing MoE linearization.
`set_onload_device` selects the execution device. LLMC sequential execution
keeps only the current subgraph weights resident across its samples, then releases
them; `IntermediatesCache` keeps intervening activations on CPU. Attention remains
an opaque subgraph, with input norms and individual experts as additional
boundaries. The graph stops immediately after the last selected layer's attention,
before its MoE, subsequent layers and LM head. Earlier layers' original MoE
forwards are necessary to obtain real Layer 40 inputs; no experts are smoothed or
quantized. Internal consumer hooks still capture complete norm-linear groups.
Tracing occurs before capture hooks are installed; duplicate or missing mapping /
sample calls fail. Failure removes hooks and does not write a completed cache.

This is single-process CPU/disk offload, NOT E2's eight-rank data partition and
shared-backing path. `WORLD_SIZE>1` is rejected; use `python`, not `torchrun`.
Use `--device-map auto` only as an explicit direct-forward fallback; it rejects
offload-specific options. No full-model `.to(device)` or `.bfloat16()` is applied.
Native dtype exceptions are preserved by the loader (the existing cache's shared
weight concatenation can promote mixed consumer dtypes). The model path must be
local. Reuse the historical server's capacity-checked max-memory budget rather
than guessing device/RAM capacities; reserve room for attention workspace and
CPU intermediate/reference caches. Disk backing is an additional model-size
storage requirement and is never automatically removed. This is not tensor
parallelism and does not guarantee the server memory budget will fit.

Inspected source: `quarot-experiments` commit
`492b1e109117f1eccedf67059b2669aada339102`, `e2_collect.py`, `e2_sequential.py` and
`E2_IMPLEMENTATION.md`. Nothing in that branch or its results was modified.
2026-11-18 fetches failed again (SSH process / DNS restrictions); not synced.
Tests use synthetic tiny GLM only. Real full checkpoint loading/NPU remain untested.

Original cache and BF16 references reside on CPU. Only one weight-row slab and
one token batch/output reside on device. `--batch` and `--rows` tune memory;
changing GEMM shapes can change rounding. Timing includes transfers and CPU
FP64 reductions: it measures end-to-end experiment cost, not isolated kernel
throughput. Search excludes reference preparation (recorded separately) and
post-search diagnostics. Capture time includes graph tracing, hook forwards and copying, not
model loading/tokenization/initial weight snapshot. CPU peak RSS is process
lifetime; device allocated/reserved peaks are process-lifetime allocator counters,
not stage-exclusive peaks. NPU/CUDA timers synchronize before and after work.
Short smoke timings are noisy; formal cost conclusions need repeated warm runs.
NPU FP8/QDQ operator support is untested: errors propagate, with no silent CPU
fallback. Run CPU QDQ tests on the server environment first, then NPU smoke.

## Commands (server, from work/llm-compressor)

Use the existing environment with this checkout on PYTHONPATH, installed torch,
compressed-tensors, transformers, accelerate, psutil, pytest; torch_npu for NPU.

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experiments/flexsmooth_proxy -q -p no:cacheprovider

# Recommended first real GLM invocation: one mapping, 128 tokens per split.
python -m experiments.flexsmooth_proxy.run capture \
  --model-path /server/models/GLM-5.2 --data /server/data/calibration.json \
  --layers 3 --mappings input_layernorm --tokens 128 --context 128 \
  --seed 42 --dtype bfloat16 --device npu:0 --device-map auto_offload \
  --offload-dir /server/results/flex-e01-smoke-offload \
  --output /server/results/flex-e01-smoke/capture
python -m experiments.flexsmooth_proxy.run search \
  --cache /server/results/flex-e01-smoke/capture/cache.pt --device npu:0 \
  --output /server/results/flex-e01-smoke/search
python -m experiments.flexsmooth_proxy.run evaluate \
  --cache /server/results/flex-e01-smoke/capture/cache.pt \
  --search /server/results/flex-e01-smoke/search/search.pt --device npu:0 \
  --output /server/results/flex-e01-smoke/evaluation

# After smoke passes: both layers, both mappings, 2048 effective tokens per split.
python -m experiments.flexsmooth_proxy.run capture \
  --model-path /server/models/GLM-5.2 --data /server/data/calibration.json \
  --layers 3 40 --mappings input_layernorm q_a_layernorm \
  --tokens 2048 --context 2048 --seed 42 --dtype bfloat16 --device npu:0 \
  --offload-dir /server/results/flex-e01-offload \
  --output /server/results/flex-e01/capture
python -m experiments.flexsmooth_proxy.run search \
  --cache /server/results/flex-e01/capture/cache.pt --device npu:0 \
  --proxies INT8 MXFP4 MXFP8 --batch 128 --rows 256 \
  --output /server/results/flex-e01/search
python -m experiments.flexsmooth_proxy.run evaluate \
  --cache /server/results/flex-e01/capture/cache.pt \
  --search /server/results/flex-e01/search/search.pt --device npu:0 \
  --output /server/results/flex-e01/evaluation
```

All output directories must be empty. Model/data paths are server placeholders.
For CPU synthetic smoke: `synthetic --tokens 128 --dtype bfloat16 --output CACHE`,
then the same search/evaluate commands with `--device cpu` and the new cache.
`--tokens`, `--layers`, `--mappings` apply at capture, not retroactively at search.

## Outputs and decision rule

- cache.pt: original weights/activations, ordered consumers and row boundaries,
  effective token identities, seed, model path, capture timing and provenance.
- search.pt / search.json: six selected scales per mapping, all 42 candidates
  per run (including null JSON for nonfinite losses), stage, alpha, beta, NRMSE,
  per-candidate seconds, reference/search time and memory counters.
- evaluation.json: E0 MSE/NMSE/max_abs/finite/pass for every selected scale;
  target MSE/NMSE/normalized RMSE, reductions versus No Smooth; pairwise alpha,
  beta, log2-scale RMS/max, identical parameters, boundary flags, W+A/W-only
  gain and time ratio; stage flatness (relative range and number within 1%).
- summary.csv, candidates.csv, report.md, loss_curves.svg: requested summary
  table, No Smooth, complete candidate curve data and quick visual curves.
  Curves use candidate order (21 ascending steps) and per-panel loss ranges.

Hypothesis: including activation QDQ can alter scales and improve held-out
matched-format NMSE, at extra cost. Minimal real experiment is the command above.
Pass criteria: exact token counts, disjoint identities, full mapping consumers,
finite target metrics, all E0 passes (NMSE <= 64*dtype_eps^2), six runs and
252 candidate entries per mapping. Quality benefit is positive validation NMSE
reduction, separately for each target and proxy, compared with both No Smooth
and its W-only counterpart. A search optimum does NOT imply validation optimum;
all-candidate validation/regret is intentionally deferred, so that consistency
cannot be claimed from this first round. Cross-proxy minimum losses are never
ranked. Report whether INT8's selected scale actually performs well under each
common target, across mappings; it is not reliable merely because INT8 loss is
small. Only consider a future Block NMSE proposal after real E0/E1 evidence is
repeatable and relevant gains justify time cost. This script never launches it.

Local evidence: 13 small tests passed (four dtypes E0 including zero channels,
CT fused-vs-separate QDQ, exact MX values, INT8 W-only/W+A objective, safe metrics,
ties/grid, reproducibility, cache split rejection, capture/cleanup). 128-token
BF16 synthetic smoke completed six searches and both targets, all E0 passed.
Synthetic examples are plumbing evidence only, not GLM-5.2 quality conclusions.
No real checkpoint, NPU, ModelSlim differential, Block or full-model run was used.

2026-11-18 offload validation adds loader control tests and tiny GLM direct-vs-CT
disk-backed sequential comparison, including exact original weights, activations
and token identities and forbidden later-layer/LM-head hooks. No real model run.

Final local check: all 16 tests passed, Ruff and git diff --check passed.
