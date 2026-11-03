# QuaRot / MXFP4 routed-expert experiment

Only experimental files change. B is norm fusion, C is the same fusion followed by
actual LLMC Hidden B32 rotation (seed 1234, FP32 arithmetic, BF16 write-back after
fusion and after rotation). No FlexSmooth, BPB, router or shared-expert measurement.

## Source audit and limits

`quarot/mappings.py` includes post_attention_layernorm gain in routed gate/up
columns; down has no gain fusion. `rotate_axis` computes gate/up WQ and down Q^T W.
The signed 32x32 block is repeated 192 times at hidden size 6144; blocks do not
have independently sampled signs. Q is generally not symmetric. Expert input is
unit-gain RMSNorm(x) Q and output must be multiplied by Q^T before comparison.
The actual whole-model modifier also transforms other consumers; this experiment
isolates expert algebra and does not replace the whole-model modifier lifecycle.

The checkpoint adapter validates config layer count against index layer IDs,
selected expert IDs, every selected tensor's header shape and BF16 dtype. Only
explicit `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` is supported;
packed/fused layouts are rejected with the missing key. Corresponding gain key is
`model.layers.L.post_attention_layernorm.weight`. Actual server naming and topology
remain unverified until this preflight runs. No checkpoint was sought locally.

QDQ uses CT's MXFP4 preset, LLMC MemorylessMinMaxObserver and CT fake_quantize.
Weights are static; future inputs dynamic. Grouping is contiguous input-axis 32
values, E2M1 grid +/-{0,.5,1,1.5,2,3,4,6}, nearest with even ties. CT's MX scale
routine adds the mantissa rounding increment, masks exponent bits, subtracts the
E2M1 exponent offset (2), and stores biased E8M0. This is **not** simply ceil(log2
(absmax/6)). Source BF16 is preserved through observer/QDQ; metrics reduce in FP64.
No INT4, ModelSlim RTN or scale optimization is substituted. Local CT is installed
0.19.1a20260919, not the adjacent CT checkout. Version, source paths and hashes are
recorded per run. Match the historical server environment before combining runs;
the historical server's exact package/source fingerprint is not available here.

Paper motivation: [Egiazarian et al., v3](https://arxiv.org/html/2509.23202v3).
We measure matrix SSE/energy, arithmetic mean and p95 of per-block SSE/energy,
and mean relative squared error of the first maximum-absolute element per block.
Epsilon is 1e-30; all-zero blocks contribute zero. Magnitude percentiles use nearest
rank. These empirical, energy-normalized metrics are not the paper's analytical
expected-error normalization. Gate/up axes align with MX groups; down output-axis
rotation mixes rows and does not align with MX input-axis groups. Gains are signed
100*(B-C)/B, undefined (JSON null) if B=0. No projection metric establishes W4A4 gain.

## Run on Linux

Use the existing historical LLMC/CT environment, with this branch's LLMC source
installed editable. No dependency upgrade or checkpoint conversion is needed.
From the llm-compressor repository root:

```bash
python -m pytest experiments/quarot_mxfp4 -q -p no:cacheprovider
python -m experiments.quarot_mxfp4.run --output /path/to/new/e0-run
python -m experiments.quarot_mxfp4.run \
  --checkpoint /path/to/original-bf16-checkpoint \
  --layers 3 40 77 --experts-per-layer 3 --seed 1234 \
  --output /path/to/new/e1-run
```

Output directory must not already exist. Optionally specify `--experts 0 7 12` to
use those IDs at each layer. Otherwise Python Random(seed+layer) samples without
replacement before observing data. Run results include selected IDs and shard keys.
The runner always gates E1 on E0. CPU-only: no accelerator checkpoint loading.
One matrix and one norm at a time; both branches measured serially. Metric temporary
arrays dominate memory: conservatively budget 100 bytes per matrix element plus
Python/PyTorch runtime and mapped shard pages (not a measured RSS guarantee).
Default --max-elements 64000000 rejects larger matrices before loading. At
6144x2048, allow roughly 1.3 GB working memory plus runtime; 4 GB free CPU RAM is a
reasonable initial allowance. Reduce the cap for constrained hosts. No entire
shard is deserialized; safe_open.get_tensor reads selected tensors only.

Outputs: results.json (metadata, E0, rows/errors), metrics.csv (E1 per matrix),
report.md (B/C comparison). Failures propagate nonzero exit and record partial
results as failed; no excluded or unmeasured rows are silently treated as success.

## Findings on 2026-11-03

E0 synthetic hidden=6144, intermediate=256 (reduced intermediate for a light CPU
check), 8 tokens, seed=1234 passed. FP32 C vs original: max abs 2.265e-6, mean abs
3.757e-7, NMSE 5.315e-13. BF16 write-back C vs original: max abs 0.01187, mean abs
0.001959, NMSE 1.427e-5. B BF16 NMSE was 5.503e-6. These establish transform
correctness for the synthetic input, not GLM accuracy. Full intermediate size can
be selected with --e0-intermediate. Detailed numbers are in results/synthetic-e0.
Synthetic sharded E1 smoke results exercise reporting only; do not infer GLM gains.
Real E1 and real-activation W4A4 remain unmeasured. No algorithm bug identified.

Fetch attempts failed due to local network/process restrictions; not synchronized.
Pinned initial LLMC 824b9c37030c8413c551b0d5bd34cc8cc6917dca;
reference LLMC de16f385705239c10bc54add79d4881242ddbb8c;
ModelSlim 1c6391745db71912bb145e7f139db1cff8cad9e1;
adjacent CT a66733fcb18d52861ad4ab0e395f621577d51f02 (not imported).
No branch switch, rebase or merge performed.
