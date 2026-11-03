# E2: next experiment, not an implemented calibration pipeline

Hypothesis: Hidden B32 may trade weight error for activation error; down's input
axis remains unrotated and its input distribution is SiLU(gate)*up. Neither E1
matrix gain nor Gaussian synthetic replay can predict full Expert W4A4 NMSE.

Minimum experiment after E0/E1: one routed layer, 1-2 short calibration samples
(initially 128-256 tokens), then selected experts with nonzero routing counts.
Reuse src/llmcompressor/pipelines/sequential/pipeline.py, IntermediatesCache,
existing pre-hooks and CT auto_offload. Historical partition targets were
input_layernorm / ExpertMLP / GlmMoeDsaMLP, 8 ranks, propagate_error=True,
moe_calibrate_all_experts=False. Verify those class names on the server before
tracing. Start one layer L4; no L5 recommendation before L0-L4 evidence.

Capture once from original BF16 routing: selected expert input after weighted
post_attention RMSNorm, layer/expert ID, token/sample/position IDs, router top-k
membership, routing coefficients, total routing count and retained token count.
Also retain norm gain and epsilon, config, checkpoint identity and cache dtype.
Prefer recording unit-gain RMSNorm input alongside weighted norm input: dividing
by gain is invalid at zero gains. Capture residual/pre-norm input if the existing
hook only exposes weighted input. Do not re-route B and C independently.

Replay B and C from the identical unit-gain normalized tokens and selected weights.
C input is rotated; C output is inverse-rotated before comparison. Evaluate:
1. transformed unquantized B/C versus original FP32 expert reference;
2. W4A16 versus each branch's unquantized reference and the common original;
3. W4A4 with dynamic QDQ at gate/up inputs and at SiLU(gate)*up before down.
Do not quantize gate/up outputs or final down output independently. Report all
three max/mean/NMSE statistics. W4A4 minus W4A16 is a diagnostic difference, not an
additive causal decomposition (quantization errors interact). Keep routing weights
separate from standalone expert outputs; document any later MoE aggregation.

core.expert supplies only the small float/W4A16/W4A4 replay interface; core.qdq
accepts activation=True. Caller must specify arithmetic/storage dtype consistently:
first FP32 replay with original BF16-derived parameters, then a separate BF16
runtime-parity run. The helper QDQs weights in their stored dtype before promoting them to the input
arithmetic dtype, preserving E1 semantics for BF16 stored weights. No real-activation pipeline has been implemented in this iteration.

Resource plan: weight memory O(3*hidden*intermediate) for one expert, temporary QDQ
buffers, plus O(retained_tokens*hidden + retained_tokens*intermediate) activations.
At 6144x2048 BF16 expert weights are ~72 MiB; FP32 copies ~144 MiB. Budget at least
1-2 GiB for replay buffers and measure RSS/accelerator peak, separately from
sequential loading's shared CPU/disk backing. 256 BF16 input tokens need ~3 MiB
per expert before metadata; cap retained tokens per expert, stream cache to disk.
The full checkpoint still needs server CPU/disk backing; tiny token counts do not
remove that requirement. No mounts or checkpoint writes.

Risks: packed experts versus explicit modules; AST partition coverage and dispatch
hooks; zero-token experts; shared cache concurrency across 8 ranks; duplicate token
captures; backend CPU/NPU rounding parity; dtype differences; propagated upstream
quantization error. For rotation isolation, freeze the original activation cache
and routing, irrespective of historical propagate_error=True. Run propagation as
a later separate experiment, never mix it into B/C inputs.

Pass criterion: E0 tolerance holds on selected shapes; cache has matching token IDs
and counts; QDQ parity with server CT is verified; all outputs finite; every result
has common coordinates and explicit reference/dtype. Log selected tensor shapes,
scales, zero rates, routing counts, token sizes, memory peaks and source hashes.
Report per-expert errors, token-count weighted mean NMSE, and pooled SSE/energy
separately (they are different aggregations); zero-token experts are unmeasured.
Improvement is a research outcome, not a precondition for a valid experiment.
