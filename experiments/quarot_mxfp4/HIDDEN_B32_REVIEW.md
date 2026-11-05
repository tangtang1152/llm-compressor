# Read-only review: can selected routed experts skip global Hidden B32?

Conclusion: the current implementation has no equivalent local skip switch.
Removing gate/up/down operations for selected experts/layers while keeping the
global hidden rotation generally changes the network function. This review adds
no skip feature and changes no production code.

Call chain (repository-relative sources):

1. src/llmcompressor/modifiers/transform/quarot/base.py:
   on_initialize -> _validate_model -> build_glm_plan. The initialization creates
   one matrix per RotationSpace, including the shared hidden space named rot.
2. .../quarot/mappings.py:93 declares RotationSpace("rot", hidden). Lines 135-157
   schedule embedding WQ, lm_head WQ, attention hidden-input WQ and o_proj Q^T W.
3. .../quarot/mappings.py:167-200 collects router, routed and shared expert gate/up
   consumers; fuses post_attention_layernorm gain into all those consumers, then
   schedules input-axis rot on them and output-axis rot on each down_proj.
4. .../quarot/base.py:288-365 on_calibration_start fuses every planned norm, then
   iterates (plan.embedding, *plan.rotations) without a per-layer/expert skip.
   _rotate (line 271) looks up _matrices[operation.space], calls rotate_axis, and
   updates the stored weight (and output bias if present).
5. .../quarot/rotation.py: rotate_axis implements axis=1 as WQ, axis=0 as Q^T W.
   This is an offline weight transform; there are no local runtime boundary
   rotations added around an expert.

Mathematical reason (row-vector convention): after embedding and residual-path
transforms, hidden activations are x'=xQ. With norm gamma already fused, expert
input weights G'=GQ and U'=UQ give x'G'^T=xG^T, x'U'^T=xU^T. Its intermediate
SiLU(gate)*up is unchanged. D'=Q^T D yields output y'=yQ, consistent with the
rotated residual stream and other routed/shared outputs. Unit-gain RMSNorm commutes
with orthogonal Q up to arithmetic rounding; the original nonuniform gamma must
first be fused consistently into its consumers.

Skipping only gate/up leaves xQ G^T / xQ U^T instead of the original preactivations.
Skipping only down produces y instead of yQ and breaks coordinate compatibility
at expert aggregation/residual addition. Skipping all three is generally
F(xQ) != F(x)Q because SiLU and elementwise gating do not commute with an arbitrary
signed Hadamard. Skipping all routed experts in a layer still has the same boundary
problem; the router/shared expert and residual coordinates do not disappear.
Keeping router rotation preserves routing scores but cannot repair expert outputs.
Likewise, leaving selected norm gains unfused while resetting the shared norm to
ones would change those consumers' function.

An equivalent alternative would require explicit boundary compensation: convert
x' to x with Q^T before an otherwise fused/unrotated expert and rotate its output
back with Q, or redesign the surrounding layer/residual basis consistently across
all consumers and branches. These introduce runtime work or broader architectural
changes and are outside the current offline modifier. Orthogonal hidden rotation
may be chosen globally, but it cannot simply be omitted locally because E1 weight
NMSE benefit is small. E1.3's group-flat gamma is a diagnostic intervention and
does not establish a safe selective-skip optimization.
