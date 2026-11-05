# E2 real-activation expert reconstruction

Implemented: data/tokenizer preflight, an experimental sequential capture adapter
using the historical distributed loader, bounded capture/shard merging, and CPU
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
is checked against actual norm output (rtol=atol=.01); max/mean/NMSE and epsilon
are recorded. Capture reductions use FP32 for NPU compatibility. Actual original
z_gamma is stored directly, including BF16 rounding. Zero/negative gamma works.

The router's actual outputs are observed, never rerun. Selected Expert pre-hooks
compare the SHA256 of actual BF16 input rows against original routing dispatch.
LinearExperts2D uses SLOT-major ordering, not token-major ordering. Padding is
included in dispatch verification but excluded from saved tokens. Missing,
duplicate or mismatched hooks fail. Onloaded expert weights are hashed once;
hashes must agree across ranks and match the CPU replay checkpoint.

Fixed selection: 3:[76,158,202], 6:[101,182,240], 16:[70,81,164], 40:[60,165,236].
Each expert retains the 512 lowest SHA256 priorities of
(seed1234,layer,expert,global_sample,token_position). Selection is independent of
errors, input order and rank partition. Rank-local counts may be below128; only
the merged global capture requires128. Shards merge one at a time with bounded
reservoirs. Insufficient counts produce an incomplete manifest with actual counts;
increase samples from the same source, rerun preflight and capture in new paths.
No expert substitution or random filling.

Provenance includes model config/modeling source, software/source hashes,
dataset/tokenized hashes, epsilon, sample positions, routing weight ranges,
selected weight identities and capture file hashes. Failures leave failed/running
rank status and no complete global capture. Torchrun supervises worker failures;
abrupt death can leave running status/offload files. No automatic cleanup.

## Replay and metrics

A: original captured z_gamma and raw weights. B: z with fused gate/up and unchanged
down. C: BF16(zQ), fused/input-rotated gate/up and output-rotated down. Uses existing
transform(), HadamardRotation B32 seed1234, FP32 transforms and BF16 write-back.
C outputs pass through inverse_hidden before comparison.

All FP/W4A16/W16A4/W4A4 outputs share the **same original A FP reference** and also
report errors against their own branch FP. B/C FP common-reference errors expose
transform/input/coordinate-write-back rounding. Primary outputs are unweighted;
router weights are retained, not reapplied or recomputed. Gate/up/down input QDQ
is enabled only by the activation switch. Down input always comes from that
mode's actual SiLU(gate)*up computation.

CT MXFP4 E2M1/E8M0 group32 axis=-1 is retained. Static weight QDQ still uses
MemorylessMinMaxObserver and is cached once per representation. Activation QDQ
uses CT compute_dynamic_scales_and_zp + fake_quantize, with scales shaped
[tokens,input_dim/32]. The previous experimental helper used a static observer
for activations, which could reduce token/group axes. Correcting that helper
restores native dynamic semantics; production quantization and E1 weight QDQ are
unchanged.

Default replay arithmetic is BF16. --arithmetic float32 is a separate diagnostic:
stored BF16 transformed weights are QDQ'd before promotion; intermediate inputs
and activation QDQ then use FP32. Do not combine these runs. CPU error reductions
use FP64. Activation records with applied=False describe hypothetical QDQ error,
not a perturbation to the actual computation.

## Resource budget

- Final z/z_gamma at hidden6144 and512 tokens: ~12 MiB/expert, ~144 MiB for12 experts.
  Routing/manifest metadata is small. Retained audit shards add up to8*144 MiB for
  eight ranks: **~1.27 GiB total activation payload**, including final capture.
  Replay does not duplicate tensor files. Rank files are not automatically removed.
- Each rank retains a bounded ~144 MiB reservoir plus temporary replacements.
  Norm/router tensors live only in the same subgraph/batch; pending Expert checks
  retain hashes rather than tensors across batches. Candidate transfers are batched.
- A2688x6144 BF16 tensor is31.5 MiB. Intermediate caches, residuals, attention masks,
  DSA temporaries and allocators are additional and can dominate. Full loader
  host/shared-memory/disk backing remains in the historical TB class: stopping at
  Layer40 does not avoid loading backing for the checkpoint. Small capture files
  do not imply a small model load.
- One6144x2048 expert's three BF16 weights total72 MiB; A/B/C plus a cached QDQ
  representation use288 MiB, plus FP32/FP64 temporaries. Allow several GiB CPU RAM
  for replay. Real peak RAM/VRAM/offload volume/runtime are **unmeasured**.
- Check free/cgroup memory, /dev/shm, free offload disk and placement limits before
  loading. Do not blindly reuse historical available-memory figures or change mounts.
  Historical versions: torch2.10.0, torch-npu2.10.0.post4, transformers5.14.1,
  compressed-tensors0.17.0, accelerate1.14.0, safetensors0.8.0. Record the current
  server environment and run its local-size tests before L4.

## Local validation and remaining limits

2026-11-05: `python -m pytest experiments/quarot_mxfp4 -q -p no:cacheprovider`
passed **25 tests in15.04s**; Ruff check passed. Capture/replay CLI --help passed.

Synthetic replay: two layers x three experts x128 tokens;72 output,54 weight and216
activation records. A FP reference error is zero. FP32 transform algebra errors
<1e-10; synthetic BF16 branch FP NMSE <0.003. These are fixture properties, not real
model bounds. Tests cover mode switches, dynamic group/token scale granularity,
inverse direction, actual quantized down input, deterministic selection, insufficient
counts, corruption, failure status, no overwrite, zero/negative gamma and absent
graph boundaries.

A separate actual Tiny GlmMoeDsaForCausalLM test uses original routing, two selected
MoE layers/four experts per layer/two samples including padding. Sequential captures
match direct original-model captures bit-for-bit; simulated rank shards merge to
identical selections/tensors; overlap fails; hooks prove the next decoder and LM
head are not executed. This exercises the real model class and sequential engine.

**Unverified:** real checkpoint load, NPU/HCCL multiprocess behavior, server data/
tokenizer identity, full-size graph/resource use and research results. A local
attempt to test load_context + auto_offload on a saved tiny checkpoint failed in
Transformers monkey_patching.apply_patches importing an unrelated Aria optional
module: missing torchvision. No loader success is claimed and no fake bypass was
added. Constructor/linearize/sequential tests are independent. Check/resolve the
server environment before L4.

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
exact Layer3 experts retain128..512 tokens, norm errors pass, actual-dispatch checks
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
  --arithmetic bfloat16 --threads 4 --output "$BASE-replay"
```

Replay requires the complete exact twelve-expert selection, not the single-layer
gate. No full-model L5 is recommended. Synthetic captures are rejected by the CLI.
All directories must be new. No plots, large per-token CSVs or automatic cleanup.

## Outputs and interpretation

Capture: per-rank status JSON and rank shards; capture/capture_manifest.json and
capture/captured_activations/*.safetensors. Only a complete global manifest denotes
success; incomplete/rank-only files cannot be replayed as real results.

Replay: results.json (configuration/provenance/scalars), a reference copy of the
capture manifest, weights.csv, activations.csv, outputs.csv, layer_summary.csv,
associations.csv, summary.md. Tensor paths remain relative to the original capture
directory recorded in results. Full runs contain144 output,108 weight and432
activation rows. Metrics include NMSE/max/mean/RMSE/per-token P50/P90/P95/P99 and
cosine (null for zero denominator). Mean-expert, token-weighted and energy-pooled
NMSE are distinct columns.

Use W4A4 C/B and C/A gains for both baselines. W16A4 versus W4A16 distinguishes
activation and weight sensitivity; FP and branch_fp errors expose transform
rounding. Associations compare projection weight/activation NMSE to common output
NMSE across experts. These are small, dependent, exploratory correlations. SiLU,
gate/up multiplication and down introduce interactions; errors cannot be added.
No real net-benefit/accuracy/selective-rotation conclusion follows from synthetic
tests. No layer-skip implementation is included.

## Safe Git sync

No amend or force-push. On the server inspect git status/log, then git fetch origin.
If clean and compatible, git merge --ff-only origin/quarot-experiments. If histories
diverge, preserve needed changes on a backup branch/patch. Only if tracked local
edits are explicitly disposable use git reset --hard origin/quarot-experiments.
Never git clean or remove result directories; do not blindly pull rewritten history.
