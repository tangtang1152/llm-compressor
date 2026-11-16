# No-Hidden QuaRot ablation — BF16 gate NOT passed

Work branch: quarot-experiments, starting at89d814b2 (clean and equal to fetched
origin/quarot-experiments). No reset/rebase/force-push. Deployment/GPQA is left to
user per explicit follow-up. No server/NPU experiment was executed.

## Implementation and mapping audit

QuaRotModifier.enabled_rotation_spaces defaults toNone (all four existing spaces).
No-Hidden selects [rot_b_proj, rot_uv, rot_kv_b_proj]. RotationPlan filters complete
spaces in canonical order; embedding is optional. Disabled spaces build no matrices
and execute no rotations. All original norm fusions remain, including Router,
Indexer, MLP and final norm/head. Tied embedding/head must still be untied for final
norm fusion. Fusion consumers remain in dtype/offload/alias validation even when
hidden rotations no longer target them. No inference hooks/runtime changes.

| Space | Producer / consumer compensation | No-Hidden |
|---|---|---|
| rot | embedding output; head input; q_a/kv_a/indexer wk and weights_proj inputs; o output; Router and all dense/shared/routed gate/up inputs and down outputs | all removed |
| rot_b_proj | q_a output; q_b and indexer wq_b input | retained, shifted B32 |
| rot_kv_b_proj | kv_a latent rows (offset0, stride kv_rank+rope); kv_b input | retained; rope rows untouched |
| rot_uv | kv_b value rows (offset nope, stride nope+value); o input blocks | retained |

Metadata records actual enabled spaces. Existing default Full weights are bitwise
identical to the pinned89d814b2 implementation in regression testing. New per-space
logs are summed host wall times, not synchronized NPU kernel timings. Norm fusion,
rotation and total progress logs remain; historical Profile captures complete
oneshot/load/save timing and memory snapshots.

## Local evidence and blocking issue

-10 targeted synthetic mapping/lifecycle/artifact/driver tests passed initially.
-Actual ModelSlim source differential, pin1c639174: No-Hidden retains original norm
 fusion and three original rotation pairs. Maximum weight difference1.0430813e-7
 (rtol2e-5, atol2e-7), passed. Oracle source remains outside Apache checkout; no
 ModelSlim algorithm code copied. GLM mapping source SHA256:
 fd2ef95b73c85a45255d2c82684ac6d7eb93858592da25905bf8a5956145deae.
-Tiny synthetic MLA FP32/BF16 tests cover individual spaces and combined local
 spaces, q/k, attention probabilities, indexer scores, Router logits and final
 output; fixed NMSE bounds1e-10/3e-4 respectively passed.
-Actual installed Transformers GlmMoeDsaForCausalLM (tiny initialized weights,
 dense+MoE, actual indexer/router) exposes a failing BF16 case, seed1234.
 Full experiment-directory run: **44 passed,1 failed**. The failure is retained,
 not xfailed, skipped or relaxed. No real-model accuracy conclusion is drawn.

| Actual Tiny GLM spaces | FP32 output NMSE | BF16 output NMSE |
|---|---:|---:|
| Query only |2.94985e-14|3.66750e-6|
| Value only |8.78520e-14|3.75691e-5|
| KV only |7.92663e-14|0.00141195|
| No-Hidden (all local) |9.16116e-14|0.00140416|
| Full (all four) |1.45381e-13|0.00142644|

No-Hidden actual attention output NMSE by layer:3.55530e-5 /4.50677e-5.
Indexer top-k sets unchanged. Router logit NMSE1.13881e-5; one Token changes selected
Expert. Discrete routing amplifies BF16 rounding. Full also fails the fixed3e-4
final-output gate, so this is not evidence of a newly missing compensation pair.
It is also not permission to relax the gate. **Full quantization is not cleared.**
Further gate criteria/real-layer validation must resolve routing sensitivity first;
do not modify forward/router or use online transforms to hide this difference.

## Historical pipeline (reused, not invented)

Pinned97eabf56:examples/glm52_precision/oneshot_profile.py and mixed_mxfp.yaml.
prepare exports those exact files into a fresh server run directory, injects only
recipe[0].enabled_rotation_spaces and adjusts the driver's git-provenance cwd.
Recipe YAML bytes otherwise unchanged. Historical driver forces QuaRot FP32 and
FlexSmooth max_tokens=None. Order is QuaRot -> FlexSmooth(norm-linear,ov) -> mixed
MXFP quantization. Routed experts remain MXFP4; original selected attention,
indexer wq_b, dense/shared experts remain MXFP8; other exclusions unchanged.

Uses original42 sample source, max2688, batch1, eight-rank auto_offload/HCCL,
sequential targets, source-rank compression/save/tokenizer and representative
MXFP4/8 save/reload equality checks. It does NOT add resume support where the
historical driver had none; failed transforms must restart from raw BF16.

Important inherited server action: historical driver remounts /dev/shm to1200G
AFTER loading. Requires the original privileges and physical/cgroup capacity;
this behavior is recorded in preparation.json, not silently replaced. Check the
historical initial shared-memory size and offload policy before execution.
The preparer checks source config, baseline metadata, new paths and free output
space (>=2x baseline checkpoint bytes). This is not a complete memory/offload
capacity proof; inspect offload disk, RAM, /dev/shm and cgroup limits separately.
Use original max_memory JSON if the baseline run specified one.

## Server commands — prepare only until BF16/L4 gates pass

```bash
source /mnt/share/t50065065/venvs/l5-llmc/bin/activate
cd /mnt/share/t50065065/llm-compressor
python -m pytest experiments/quarot_mxfp4/test_no_hidden.py -q -p no:cacheprovider
```

Current local version has the actual-GLM failure above. Do not proceed to all-model
quantization simply because synthetic MLA or source differential passed. No L5
recommendation is made before real-layer L4 validation.

To inspect/export the exact runnable driver without loading weights, supply the
actual existing Full checkpoint path (not guessed here):

```bash
export FULL_QUAROT_CHECKPOINT=/actual/existing/full-quarot-checkpoint
RUN=/mnt/share/t50065065/results/quarot/no-hidden-$(date +%Y%m%d-%H%M%S)
NEW=/mnt/share/weights/LLMC/GLM-5.2-MXFP4-MXFP8-NoHiddenRot
python -m experiments.quarot_mxfp4.no_hidden_run prepare \
  --checkpoint /mnt/weight/GLM-5.2 \
  --baseline-full "$FULL_QUAROT_CHECKPOINT" \
  --dataset /mnt/share/datasets/qwen3_cot_w4a4.json \
  --run-dir "$RUN" --output "$NEW" --offload-dir "$RUN-offload"
```

New paths must not already exist. Artifacts: oneshot_profile.py, mixed_mxfp.yaml,
preparation.json and quantize.sh. After resolved BF16 and server L4 gates only:
`bash "$RUN/quantize.sh"`. The generated script itself runs test_no_hidden first
with set -euo pipefail; a failing test aborts BEFORE model loading. It does not
ignore failures or overwrite checkpoints. Runtime artifacts: quantize.log,
profile/rank-*.json, new checkpoint/tokenizer/config, artifact_validation.json.

Artifact validation checks all shard headers and index key/file coverage, tensor
shapes/dtypes against the actual Full checkpoint, exact quantization/FlexSmooth
configuration, model config and QuaRot metadata. This catches layout/exclusion
changes but is not a numerical correctness proof; historical tile reload checks,
server initialization and the user's GPQA test remain necessary.

## Comparison status

Historical archive: workspace experiments/evidence/production-20261008-112706.
Its timeline records Full norm fusion1588.96s, rotation1376.84s, QuaRot total2976.03s.
Rank1 report records oneshot_total7280.19s, model_load2095.56s, save5186.02s, status
passed. These phases are not an independently measured end-to-end wall clock;
rank timings must not be summed across workers. This archive is not assumed to be
the user's latest GPQA baseline without checking identity and evaluation settings.

| Metric | Full QuaRot | No-Hidden | Delta |
|---|---|---|---|
| Hidden rotation |Yes|No (implemented)|configuration only|
| Query/KV/Value |Yes|Yes|unchanged|
| Routed MXFP4 / other policy |historical mixed|same exported recipe|unmeasured output|
| QuaRot preprocessing |2976.03s historical|not executed|unknown|
| oneshot_total |7280.19s historical rank1|not executed|unknown|
| Complete quantization wall time |not recovered|not executed|unknown|
| GPQA accuracy/correct count |not supplied|user handles|unknown|
| Checkpoint size |inspect actual Full baseline|not produced|unknown|

User-supplied E2 results (37041 routed tokens; C/A gains -17.94%,-37.43%,-69.44%,
-10.24%) motivate this ablation but do not prove a full-model GPQA improvement.
Background only: https://github.com/vllm-project/llm-compressor/issues/3387 and
https://arxiv.org/abs/2509.23202. No GPTQ/rotation redesign/runtime workaround added.
