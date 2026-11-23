import torch

from llmcompressor.modifiers.transform.flex_smooth.search import ov_scales

from ..benchmark_reuse import assess
from ..core import Objective


class RecipeObjective(Objective):
    """Full inherited 42-candidate search; only assess dispatch is optimized."""

    def __init__(self, *args, optimized=True, cache_mib=64, geometry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimized, self.cache_mib, self.geometry = optimized, cache_mib, geometry
        if geometry and self.w.shape[1] != geometry["heads"] * geometry["value_dim"]:
            raise ValueError("OV o_proj width differs from head/value geometry")

    def assess(self, scale, proxy=None, activation=False):
        if not self.optimized:
            return super().assess(scale, proxy, activation)
        return assess(self, scale, proxy, activation, cache_mib=self.cache_mib)

    def applied_scale(self, alpha, beta):
        if self.geometry:
            # Production GLM mapping is expanded MLA, not GQA repeated V heads.
            heads = self.geometry["heads"]
            return ov_scales(self.amax, self.wmax, alpha, beta, heads, heads)[0]
        return super().applied_scale(alpha, beta)


def select_group(name, group, dtypes):
    """Keep full compensation metadata; objective contains only homogeneous targets.

    Mixed target dtype is rejected, never silently concatenated/promoted. Current
    recipe's quantized consumers are homogeneous BF16; FP32 weights_proj excluded.
    """
    suffixes = (
        ("q_a_proj", "kv_a_proj_with_mqa")
        if name.endswith("input_layernorm")
        else (
            ("q_b_proj", "indexer.wq_b")
            if name.endswith("q_a_layernorm")
            else ("o_proj",)
        )
    )
    selected, weights, offset = [], [], 0
    for consumer, rows in zip(group["consumers"], group["consumer_rows"]):
        w = group["weight"][offset : offset + rows]
        offset += rows
        if any(consumer.endswith("." + s) for s in suffixes):
            dtype = getattr(torch, dtypes[consumer].removeprefix("torch."))
            native = w.to(dtype)
            if not torch.equal(native.to(w.dtype), w):
                raise ValueError(
                    "Cache weight cannot be losslessly restored to native dtype"
                )
            selected.append(consumer)
            weights.append(native)
    if not selected or len({w.dtype for w in weights}) != 1:
        raise ValueError(
            "Missing targets or mixed target dtype: separate objectives required"
        )
    dtype = weights[0].dtype
    result = dict(
        group,
        weight=torch.cat(weights),
        consumers=selected,
        consumer_rows=[len(w) for w in weights],
        compensation_consumers=group["consumers"],
        native_dtypes=dtypes,
    )
    for split in ("calibration", "validation"):
        x = group[split].to(dtype)
        if not torch.equal(x.to(group[split].dtype), group[split]):
            raise ValueError(
                "Cached activation is not losslessly recoverable in target dtype"
            )
        result[split] = x
    return result
