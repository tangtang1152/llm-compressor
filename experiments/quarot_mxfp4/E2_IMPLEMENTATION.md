# E2 real-activation expert reconstruction

Implemented: data/tokenizer preflight, an experimental sequential capture adapter
using the historical distributed loader, full sample-block capture/shard merging, and CPU
A/B/C replay. No real checkpoint or NPU job was run here. Synthetic tests establish
engineering properties, not QuaRot gains.

## Historical evidence

The user confirmed `/mnt/share/datasets/qwen3_cot_w4a4.json` exists on the server.
The historical driver is available with:

```bash
git show 97eabf56:examples/glm52_precision/oneshot_profile.py
```

The workspace archive `experiments/evidence/production-20261008-112706/` (outside
this development checkout) contains the source and `derived/timeline.log`: 42
samples, 8 ranks, batch size 1, max length 2688, 91834 valid tokens, lengths
1244..2672. JSON list[str] becomes text; JSONL text/messages is also supported.
Messages-only records use apply_chat_template(tokenize=False,
add_generation_prompt=False). Tokenization uses truncation=True, max_length=2688,
add_special_tokens=False, return_token_type_ids=False. No shuffle; the original
get_rank_partition/ReadInstruction partition preserves global row IDs.
DataCollatorWithPadding uses batch size 1.

The new driver retains load_context(), local_files_only=True, BF16, eager
attention, auto_offload and max_memory/offload_folder support. NPU torchrun uses
cpu:gloo,npu:hccl. Native FP32 model exceptions are preserved. It applies no
modifiers, does not remount /dev/shm, save a checkpoint or generate tokens. Do not
execute the archived main unchanged: it compresses, remounts, saves and reloads.

## Capture chain and failure handling

Historical targets: input_layernorm / ExpertMLP / GlmMoeDsaMLP. Actual Tiny GLM
tracing confirmed that post_attention_layernorm, mlp.gate and selected experts
remain call_module nodes. The adapter reuses trace_subgraphs, Subgraph.forward,
IntermediatesCache and disable_offloading, with a single original-weight pass.

Graph checks require exactly one norm/router/expert call, norm and router in the
same partition, and no later decoder/LM head in the executable prefix. The last
graph is cut immediately after the final selected expert call; every local sample
executes that prefix before rank merging. No early hook exception drops remaining
samples. Layer 40 selected experts execute to validate their actual inputs; later
experts/layers and the LM head are not needed. Tracing builds graph metadata but
does not run a full model forward. Changed graph layouts fail explicitly.

A tiny instance of the installed GlmMoeDsaRMSNorm with unit gain computes z using
its actual forward. No division by gamma or mutation of the model norm. z*gamma
is checked against actual norm output (rtol=atol=.01); MSE/NMSE and epsilon
are recorded. Capture reductions use FP32 for NPU compatibility. Actual original
z_gamma is stored directly, including BF16 rounding. Zero/negative gamma works.

The router's actual outputs are observed, never rerun. Selected Expert pre-hooks
compare the SHA256 of actual BF16 input rows against original routing dispatch.
LinearExperts2D uses SLOT-major ordering, not token-major ordering. Padding is
included in dispatch verification but excluded from saved tokens. Missing,
duplicate or mismatched hooks fail. Onloaded expert weights are hashed once;
hashes must agree across ranks and match the CPU replay checkpoint.

Fixed selection: 3:[76,158,202], 6:[101,182,240], 16:[70,81,164], 40:[60,165,236].
All valid originally routed tokens from the calibration samples are retained,
without sampling, padding, lossy compression or random filling. Each sample/expert
is a separate safetensors block: BF16 z/z_gamma/gamma, original router weight dtype,
int64 global sample_id and token_position. The NPU driver's collector writes each
block immediately; full activations are never held across all samples in RAM.
Gamma is also represented losslessly as BF16-exact numeric values in the manifest,
including layers/experts with zero routed tokens. Source/dataset/model/weight
hashes and norm epsilon remain intact.

Archive format 2 is explicitly marked all_routed_valid_tokens. Capped format 1
archives are rejected as incomplete candidates for this full-archive workflow;
no old files are rewritten. Previously discarded candidates cannot be recovered
without recapture. Archives with fewer than 128 tokens remain valid raw archives,
but E2 Replay fails explicitly for those experts.

Merge validates every block's file hash, tensor dtype/shape/bytes, token identities,
gamma and counts; rank global sample-ID sets must be disjoint, and all selected
layers must cover every sample in each partition. Global coverage must equal the
preflight sample IDs. For every expert, retained == sum(rank available) == unique
token count == sum(block token counts). Loss/truncation/duplicates cannot produce
a complete manifest. Merge copies one block at a time and verifies the copied hash.

Only after the full archive is verified and marked complete are temporary rank
shards removed. Cleanup is restricted to the explicitly supplied sibling shard
directories with exactly the verified manifest/block file set; links, junctions
and unexpected files are rejected. A merge failure preserves every original shard
and marks the new destination incomplete. Cleanup failure is reported separately:
the verified final archive remains complete, cleanup_status=failed, remaining
shards are preserved and the driver reports failure. Final archives, unrelated
files, offload directories and historical results are never deleted.

The source retains config/modeling/software hashes, data/tokenized hashes,
epsilon, positions, original routing weights, selected weight identities and
block hashes. Abrupt worker termination can leave incomplete shard files; no
complete manifest is emitted before dispatch verification. Torchrun supervises
worker failures. Original route alignment checks and stop boundary are unchanged.

## Replay and metrics

A: original captured z_gamma and raw weights. B: z with fused gate/up and unchanged
down. C: BF16(zQ), fused/input-rotated gate/up and output-rotated down. Uses existing
transform(), HadamardRotation B32 seed1234, FP32 transforms and BF16 write-back.
C outputs pass through inverse_hidden before comparison.

All FP/W4A16/W16A4/W4A4 outputs share the **same original A FP reference** and
also report MSE/NMSE against their branch FP. B/C FP common-reference errors expose
transform/write-back rounding. Router weights are preserved but not applied to
primary Expert output metrics. Gate/up/down activation errors are computed only
when QDQ is actually applied (W16A4 and W4A4). Down input comes from that mode's
actual SiLU(gate)*up result. FP/W4A16 do not perform hypothetical activation QDQ.

Static CT weight QDQ and existing transforms are unchanged. Weight NMSE tables,
correlations, associations.csv, max/mean absolute error, RMSE, cosine and P90/P99
are removed. Activation rows contain MSE/NMSE and their SSE/element/energy audit
terms. Output rows retain both common and branch-FP MSE/NMSE plus per-token NMSE
P50/P95. All metric reductions use FP64 on CPU.

Definitions (never interchanged):

- Matrix MSE = SSE / elements; NMSE = SSE / reference energy.
- Pooled MSE = sum(SSE) / sum(elements).
- Pooled NMSE = sum(SSE) / sum(reference energy).
- Mean Expert NMSE = arithmetic mean of the expert NMSE values.

Zero reference energy gives null NMSE. Output token quantiles use defined token
NMSE values, with token_nmse_defined recording their count; all-undefined quantiles
are null. Mean Expert NMSE is null if any expert is undefined, rather than silently
omitting experts. NumPy linear quantiles are computed from exact per-token vectors
within the current expert, never by averaging batch percentiles.

### Replay sampling is independent of archival retention

--max-replay-tokens defaults to 512. It selects the lowest SHA256 priorities of
(seed1234,layer,expert,global_sample,token_position), with identity tie-breaking,
independent of file/rank ordering. Values 1..127 and negatives are rejected; 0 uses
all archived tokens. Fewer than 128 available tokens fails. A/B/C use the exact
same selected identities; there is no per-branch resampling.

results.json records available_tokens, replay_tokens, seed and a selection hash
per expert. Default selections also include positions; all-token runs store only
the hash/counts in results, with identities retained on the server. All-token
replay runs in batches of 512, with transformed/QDQ weights cached once per expert;
SSE/energy/elements aggregate exactly and output token vectors remain limited to
the current expert. Original activations are validated block-by-block before use.
Default arithmetic is BF16; --arithmetic float32 remains a separate diagnostic.

## Archive size and resource budget

For hidden H, each routed expert-token contributes 4*H bytes for z/z_gamma,
16 bytes for the two int64 identifiers, plus router_weight.element_size() bytes.
Gamma copies, safetensors headers and manifests are additional. With H=6144 and
FP32 router weights this is 24596 bytes/token. Total size depends on actual routing,
not the 512-token replay limit.

At the historical 91834 valid tokens, **if** top-k=8 among 256 experts and routing
were uniform, the 12 selected experts would receive about 34438 token assignments:
about 0.79 GiB plus gamma/headers. This is a planning assumption, not a measurement.
A conservative bound allowing every valid token to reach all three selected
experts in each of four layers is 1102008 assignments, about 25.24 GiB plus metadata.
Budget up to twice the archive size during validated copy/merge; temporary rank
shards are deleted only after success. Model offload space is entirely additional.

Manifest tensor_bytes measures numeric tensor payload; archive_file_bytes sums
actual safetensors file sizes including headers. The manifest JSON itself is not
included in those counters. Use the query below or du for the full directory size.
No real archive size is measured locally. CPU fixtures are small and their actual
sizes are reported separately from these server estimates.

Memory: streaming capture/merge holds one sample/expert block at a time plus identity
sets and existing sequential caches. Replay holds one 512-token batch, one mmap-backed
archive block and one expert's metadata/per-token vectors. Static BF16 raw/fused/
rotated and QDQ weights total about 432 MiB for 6144x2048 experts, with additional
FP32/FP64 temporaries. Allow several GiB for CPU replay. Original model loading,
shared memory and attention/DSA intermediates still require the historical large
server allocation; this change does not make the full loader small. Peak server
RAM/VRAM, disk throughput and runtime remain unmeasured.

## Local verification and limits

2026-11-05 validation: the full experiment-directory Pytest suite passed 34 tests;
Ruff passed. The full-capture CPU fixture has 6 experts x 640 retained tokens at
hidden 64, with 1,053,696 numeric bytes, 1,058,880 safetensors bytes, and1,065,645 total
archive file bytes including the manifest (~1.02 MiB). Its verified temporary rank
shards were removed. Replay used 512 tokens/expert by default and 640 with limit 0.
These are actual synthetic fixture measurements, not a real GLM archive estimate.

Tests include full capture above 512 tokens in one sample, multirank merge counts
and identity/tensor equality, deterministic 512-token replay, all-token replay
across batch boundaries, common and branch references, actual quantized down input,
QDQ mode switches, known MSE/NMSE/pooled definitions, corruption, missing/overlapping
partitions, duplicate tokens, and successful-only shard cleanup. The actual Tiny
GLM sequential-vs-direct capture test is retained, including padding and stopping
before the next decoder/LM head. Original E1/E1.2/E1.3 sources are unchanged.

Real checkpoint loading/NPU/HCCL, current server data/tokenizer identity and research
results remain unverified here. The previous full tiny-loader attempt was blocked
by a missing optional torchvision dependency during Transformers imports; no loader
bypass is introduced by this refactor. No real server experiment was started.

## Server commands (not executed here)

From the repository root in the validated NPU environment. Use fresh paths. This
first stage checks actual data/tokenizer without loading weights:

```bash
python -m pytest experiments/quarot_mxfp4/test_e2.py -q -p no:cacheprovider
python -m ruff check experiments/quarot_mxfp4
free -h
df -h /dev/shm /mnt/share
RUN_ID=$(date +%Y%m%d-%H%M%S)
BASE=/mnt/share/t50065065/results/quarot/e2-$RUN_ID
python -m experiments.quarot_mxfp4.e2_preflight \
  --checkpoint /mnt/weight/GLM-5.2 \
  --dataset /mnt/share/datasets/qwen3_cot_w4a4.json --samples 42 \
  --output "$BASE-preflight"
```

Preflight must match archived token counts. Matching lengths alone does not prove
identical token IDs; tokenized SHA256 is recorded and rechecked before loading.
Then **L4 single selected layer**, on eight local NPU ranks:

```bash
torchrun --standalone --nproc-per-node=8 -m experiments.quarot_mxfp4.e2_collect \
  --checkpoint /mnt/weight/GLM-5.2 \
  --dataset /mnt/share/datasets/qwen3_cot_w4a4.json --samples 42 \
  --preflight "$BASE-preflight/data_preflight.json" \
  --device npu --layers 3 --threads 4 \
  --offload-dir "$BASE-L4-offload" --output "$BASE-L4"
```

Reuse the historical placement policy. If that launch used explicit max_memory,
pass the same capacity-checked JSON with --max-memory; no capacity is invented here.
The sample offload path is shared storage: use the prior validated faster backing
filesystem if available. Offload and result directories must both be new.

L4 passes only if every rank status and the global capture are complete, all three
exact Layer3 experts retain every available token and have at least 128 for replay,
norm errors pass, actual-dispatch checks
pass and resource use is within server limits. Inspect rank logs/counts. **Only
then** run the four-layer local-expert capture and CPU replay:

```bash
torchrun --standalone --nproc-per-node=8 -m experiments.quarot_mxfp4.e2_collect \
  --checkpoint /mnt/weight/GLM-5.2 \
  --dataset /mnt/share/datasets/qwen3_cot_w4a4.json --samples 42 \
  --preflight "$BASE-preflight/data_preflight.json" \
  --device npu --layers 3 6 16 40 --threads 4 \
  --offload-dir "$BASE-offload" --output "$BASE-capture"
python -m experiments.quarot_mxfp4.e2 \
  --checkpoint /mnt/weight/GLM-5.2 \
  --capture "$BASE-capture/capture" \
  --arithmetic bfloat16 --threads 4 --max-replay-tokens 512 \
  --output "$BASE-replay"
```

Replay requires the complete exact twelve-expert selection, not the single-layer
gate. No full-model L5 is recommended. Synthetic captures are rejected by the CLI.
All directories must be new. No plots or large per-token CSVs. Only verified temporary
rank shards are cleaned after a successful full merge.

## Server archive query and files to share

The final raw archive is `$BASE-capture/capture/`. It contains the manifest and
sample blocks under captured_activations/. It is reusable and never deleted by
Replay. Temporary rankN/ directories disappear only after successful merge.

Inspect counts and actual file sizes without reading/printing activation values:

```bash
CAPTURE="$BASE-capture/capture"
python - "$CAPTURE" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / "capture_manifest.json").read_text())
print("status", m["status"], "cleanup", m.get("cleanup_status"))
for e in m["experts"]:
    print("layer", e["layer"], "expert", e["expert"],
          "available", e["available"], "retained", e["retained"],
          "parts", len(e["parts"]))
print("numeric payload bytes", m["tensor_bytes"])
print("safetensors file bytes", m["archive_file_bytes"])
print("all directory file bytes", sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))
PY
du -sh "$CAPTURE"
```

For future server-side exploration, use `read_manifest(root)` and
`read_part(root, entry["parts"][i])` from e2_capture; they validate the block hash,
dtype and identity counts. Each block returns z/z_gamma/gamma, original router
weights and sample/position tensors. Do not transmit these raw arrays to chat.

To expand replay without reloading the model, use a fresh result directory:

```bash
python -m experiments.quarot_mxfp4.e2 \
  --checkpoint /mnt/weight/GLM-5.2 --capture "$CAPTURE" \
  --max-replay-tokens 0 --threads 4 --output "$BASE-replay-all"
```

Replay outputs **only six files**:

- results.json: configuration/provenance, archive reference, available/replay counts,
  selection identities/hash and core results.
- activations.csv: Layer/Expert/Branch/Mode/Projection MSE/NMSE, W16A4/W4A4 only.
- outputs.csv: common-A and branch-FP MSE/NMSE plus token P50/P95.
- activation_summary.csv: Layer/Projection/Branch/Mode pooled and mean-expert metrics.
  Branch/mode remain separate so different controls are never pooled together.
- layer_summary.csv: Layer/Branch/Mode pooled and mean-expert output metrics for both
  references.
- summary.md: core A/B/C comparisons and interpretation limits.

A 12-expert run produces 144 output rows, 216 activation rows, 48 output-summary rows
and 72 activation-summary rows, regardless of replay sample limit. Share these six
small result files with ChatGPT; keep full activations, rank/offload files and model
weights on the server. No Weight NMSE table, associations, copied capture manifest,
large token table or figures are emitted into Replay results.

W4A4 C/B and C/A gains answer the two baseline comparisons. W16A4 vs W4A16 measures
activation vs weight sensitivity; transform rounding remains visible through the
branch-FP reference. These interacting errors are not additive, and synthetic
fixtures establish no real net-benefit, downstream accuracy or layer-skip conclusion.

## Safe Git sync

No amend or force-push. On the server inspect git status/log, then git fetch origin.
If clean and compatible, git merge --ff-only origin/quarot-experiments. If histories
diverge, preserve needed changes on a backup branch/patch. Only if tracked local
edits are explicitly disposable use git reset --hard origin/quarot-experiments.
Never git clean or remove result directories; do not blindly pull rewritten history.
