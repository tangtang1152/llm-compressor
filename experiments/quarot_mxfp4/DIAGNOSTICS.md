# Norm Fusion / MXFP4 attribution diagnostics

Run from the llm-compressor repository root using the same Python/CT environment
as the existing real E1 results. Install matplotlib if absent (`python -m pip
install matplotlib`). This is an independent entry point; the original E1 runner,
result schema and saved results remain available. No E0 expansion or server run
is performed by this change. B/C definitions, BF16 writes, FP32 transform precision,
CT QDQ and B32 seed=1234 are unchanged.

```bash
python -m experiments.quarot_mxfp4.diagnose \
  --checkpoint /path/to/original-bf16-checkpoint \
  --reuse-e1 /path/to/previous-e1/results.json /path/to/other-e1/results.json \
  --layers 3 4 5 6 7 8 9 10 12 16 40 77 \
  --experts-per-layer 16 \
  --output /path/to/new/norm-fusion-diagnostics
```

Pass all relevant prior E1 JSON files explicitly. Existing layer selections are
reused exactly (union across compatible files); layers absent from these files
sample 16 experts with `Random(1234 + layer).sample`, as in E1. Thus new transition
layers 6/7/9/10/12 get 16 experts. `--experts-per-layer` does not replace old layer
selections. If no prior JSON is supplied, all requested layers are sampled anew.
Duplicate conflicting rows are rejected. Index hash, model config, rotation,
group axis/size, package versions and implementation hashes must match. Paths and
git commit may differ. A version mismatch requires running in the original
historical environment; there is no silent compatibility override.

Existing formal B/C metrics are copied without changes into e1_rows.csv; missing
rows are calculated with the existing E1 metric function. Existing down_proj rows
need no weight load/QDQ. New down_proj rows get only formal E1 measurements.
Gate/up weights are re-read because the old aggregates cannot reconstruct group
distributions. Each Fused/Rotated stage performs QDQ once for the newly requested
group statistics. Reused matrix NMSE is cross-checked against this QDQ at rtol 1e-6,
atol 1e-12; a mismatch fails the run. Raw is never quantized in this diagnostic.

Outputs (all in a fresh directory, rejected if it already exists):

- diagnostics.json: provenance, selections, source hashes, status/error, all summaries.
- gamma.csv: signed min/max/mean/RMS, AbsMax/RMS, absolute P50/P90/P99 and
  within-contiguous-32 absolute coefficient-of-variation distribution.
- matrices.csv: Raw/Fused/Rotated matrix and group summaries for gate/up.
- comparisons.csv: paired matrix changes and formal weight NMSE gain.
- group_correlations.csv: per-matrix Pearson associations of paired group changes,
  including gamma group CV versus Fusion Group Max/RMS change.
- matrix_correlations.csv: per-projection pooled and within-layer correlations of
  Fusion change, rotation Max/RMS change, zero-fraction change, Block P95 change
  with Weight NMSE gain. Constant/insufficient samples return null.
- groups-L*-E*-*_proj.csv.gz: **every** paired group, no error-driven sampling.
  Row/input_group identifies output channel and contiguous input block; repeated
  gamma_abs_cv joins gamma to each weight group. Raw has no QDQ fields.
- e1_rows.csv: compatible formal B/C E1 rows, both reused and newly measured.
- layer_trends.png, group_error_distribution.png, summary.md.

Definitions: Group RMS=sqrt(mean(w^2)); Max/RMS=max(abs(w))/RMS;
Max/other-RMS removes one maximum-energy element and computes RMS of the other 31.
Gamma dispersion is population std(abs(gamma))/mean(abs(gamma)). Denominator
regularizer is 1e-30 throughout (all-zero ratios are zero; single nonzero groups
can have huge Max/other-RMS). Quantiles use E1's exact nearest-rank convention.
Group relative MSE=SSE/energy; matrix NMSE=sum(SSE)/sum(energy), never the average
of group ratios. QDQ zeros include pre-existing zeros. Scale log2 is derived from
the **decoded** power-of-two scale returned by CT get_qparams, not mislabeled as
raw uint8 storage. Histogram counts and per-group exponents are both retained.

The error distribution figure shows each matrix's exact P50/P90/P95/P99 group
quantiles, not a pooled group histogram. Layer trends average matrix summaries;
use paired CSVs and within-layer associations for interpretation. Gate/up use
separate matrix correlations. Groups within a matrix and experts within a layer
are dependent; no IID significance or causal claim is made. Gamma repeats across
experts and rows, so a large group count does not add independent gamma evidence.

Memory is bounded to one expert matrix's transforms/group vectors plus summaries,
but writing all groups costs time and disk. Gzip CSV retains O(number of groups)
data per matrix; for 6144x2048 there are 393216 rows per gate/up matrix. Allow several
GB of CPU RAM and budget disk according to the number of selected matrices;
no accelerator or full checkpoint forward is needed. Interrupted runs have failed
status and may retain partial group files; start a fresh output directory to rerun.

Research interpretation: the user reports 21-27% early-layer gate/up gains, ~4.3%
at layer 8, small later gains, and little down benefit. These server observations
have not been independently checked locally. Test whether Fusion increases group
imbalance more in early layers, whether rotation reverses that change, and whether
scale/zero/error changes track gains within layers and matrices. Synchronous
gate/up gains and shared gamma support a correlation hypothesis only. Weight
content, scale boundaries and BF16 rounding are alternative contributors. No
claim that gamma causes the effect, or that W4A4 improves, follows from E1 alone.

Topology fix: config n=78 permits layer IDs exactly 0..77 or 0..78. MTP is only
accepted as optional index metadata; it remains excluded from selection. Missing
main layers and any extra layer beyond n are rejected with explicit actual IDs.
