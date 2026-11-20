#!/usr/bin/env bash
# Run one explicit stage per invocation; never run P1/QuaRot/GPQA.
set -euo pipefail
STAGE=${1:?Use plan, load, smoke, formal, offline-smoke, search, evaluate, or analyze}
: "${EXPECTED_COMMIT:?Set the delivered/integrated full Git SHA}"
: "${RUN_ROOT:?Use a new run under next/layer_extension/distributed}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select idle physical NPUs explicitly}"
NPROC=${NPROC:-8}
OLD=${OLD:-/mnt/share/t50065065/results/flexsmooth/full}
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet HEAD -- experiments/flexsmooth_proxy
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

case "$STAGE" in
  plan|load|smoke|formal)
    : "${MODEL:?Set original GLM model path}"
    : "${DATA:?Set the same local calibration JSON/JSONL as E1}"
    : "${SHARED_CPU_GIB:?Set one shared CPU backing budget}"
    : "${PRIVATE_CPU_GIB:?Set working/merge memory budget per rank}"
    : "${NPU_GIB:?Set memory budget per NPU}"
    : "${OUTPUT_RESERVE_GIB:?Set disk reserve for shards/cache/results}"
    test ! -e "$RUN_ROOT/$STAGE"
    mkdir -p "$RUN_ROOT/logs"
    mkdir "$RUN_ROOT/logs/$STAGE"
    extra=()
    if [[ "$STAGE" == smoke || "$STAGE" == formal ]]; then
      extra+=(--load-gate "$RUN_ROOT/load/complete.json")
    fi
    if [[ "$STAGE" == smoke ]]; then
      extra+=(--reference-cache "${REFERENCE_CACHE:-$OLD/capture/cache.pt}")
    elif [[ "$STAGE" == formal ]]; then
      extra+=(--smoke-gate "$RUN_ROOT/smoke/complete.json")
    fi
    torchrun --standalone --nproc-per-node="$NPROC" --max-restarts=0 \
      --log-dir "$RUN_ROOT/logs/$STAGE" --redirects 3 --tee 3 \
      -m experiments.flexsmooth_proxy.e2_collect --mode "$STAGE" \
      --model-path "$MODEL" --data "$DATA" --old-cache "$OLD/capture/cache.pt" \
      --output "$RUN_ROOT/$STAGE" --offload-dir "$RUN_ROOT/$STAGE-offload" \
      --shared-cpu-gib "$SHARED_CPU_GIB" --private-cpu-gib "$PRIVATE_CPU_GIB" \
      --npu-gib "$NPU_GIB" --output-reserve-gib "$OUTPUT_RESERVE_GIB" \
      --context "${CONTEXT:-2048}" --seed "${SEED:-42}" \
      --max-load-seconds "${MAX_LOAD_SECONDS:-600}" \
      --nmse-tol "${NMSE_TOL:-0.00001}" --max-abs-tol "${MAX_ABS_TOL:-0.125}" \
      "${extra[@]}"
    ;;
  offline-smoke|search|evaluate|analyze)
    E2="$RUN_ROOT/formal"
    test -f "$E2/complete.json"
    args=(--expected-commit "$EXPECTED_COMMIT" --device npu:0 --batch 128 --rows 256)
    case "$STAGE" in
      offline-smoke) sub=e2-smoke; args+=(--cache "$E2/capture/cache.pt" --output "$E2/smoke");;
      search) sub=e2-search; args+=(--cache "$E2/capture/cache.pt" --smoke "$E2/smoke" --output "$E2/search");;
      evaluate) sub=e2-evaluate; args+=(--cache "$E2/capture/cache.pt" --search "$E2/search/search.pt" --output "$E2/evaluation");;
      analyze) sub=e2-analyze; args+=(--old "$OLD" --new "$E2" --output "$E2/analysis");;
    esac
    mkdir -p "$RUN_ROOT/logs"
    test ! -e "$RUN_ROOT/logs/$STAGE.log"
    python -m experiments.flexsmooth_proxy.next_stage.run "$sub" "${args[@]}" \
      2>&1 | tee "$RUN_ROOT/logs/$STAGE.log"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2;;
esac
