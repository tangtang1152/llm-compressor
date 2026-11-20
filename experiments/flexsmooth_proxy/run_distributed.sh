#!/usr/bin/env bash
# Run one explicit stage per invocation; never run P1/QuaRot/GPQA.
set -euo pipefail
STAGE=${1:?Use all, capture, offline-smoke, search, evaluate, or analyze}
: "${EXPECTED_COMMIT:?Set the delivered/integrated full Git SHA}"
: "${RUN_ROOT:?Use a new run under next/layer_extension/distributed}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select idle physical NPUs explicitly}"
OLD=${OLD:-/mnt/share/t50065065/results/flexsmooth/full}
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet HEAD -- experiments/flexsmooth_proxy
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

if [[ "$STAGE" == all ]]; then
  for step in capture offline-smoke search evaluate analyze; do
    bash "$0" "$step"
  done
  exit 0
fi

case "$STAGE" in
  capture)
    : "${MODEL:?Set original GLM model path}"
    : "${DATA:?Set the same data as E1}"
    test ! -e "$RUN_ROOT/formal"
    mkdir -p "$RUN_ROOT/logs"
    mkdir "$RUN_ROOT/logs/capture"
    torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
      --log-dir "$RUN_ROOT/logs/capture" --redirects 3 --tee 3 \
      -m experiments.flexsmooth_proxy.e2_collect \
      --model-path "$MODEL" --data "$DATA" --old-cache "$OLD/capture/cache.pt" \
      --reference-cache "${REFERENCE_CACHE:-$OLD/capture/cache.pt}" \
      --output "$RUN_ROOT/formal" --offload-dir "$RUN_ROOT/offload" \
      --context "${CONTEXT:-2048}" --seed "${SEED:-42}"
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
