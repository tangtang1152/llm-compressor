# IterSmooth

`IterSmoothModifier` performs calibration-based outlier suppression before weight
and activation quantization. It supports fused norm-linear, linear-linear, gated
up-down, and value-output attention mappings, including GQA/MQA.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model=model,
    processor=tokenizer,
    dataset=dataset,
    recipe=[
        IterSmoothModifier(alpha=0.9),
        QuantizationModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
    ],
    pipeline="independent",
)
```

Use independent calibration epochs so the quantizer observes smoothed inputs.
Unlike an iterative optimizer, the referenced msModelSlim implementation computes
one analytic scale per subgraph. Bias-free RMSNorm models require the default
`symmetric=True`; non-fused runtime hooks are not supported.

See [the migration guide](../../../../../docs/itersmooth_migration_zh.md) for the
source analysis, mapping schema, supported layouts, and intentional differences.
