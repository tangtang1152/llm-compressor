# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent calibrated FlexSmooth modifier for GLM MLA subgraphs."""

import math
import weakref
from copy import deepcopy
from typing import Literal

import torch
from compressed_tensors.offload import (
    OffloadCache,
    align_module_device,
    get_execution_device,
)
from loguru import logger
from pydantic import Field, PrivateAttr, field_serializer, field_validator

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier

from .mappings import SmoothMapping, apply_scales, glm_mappings
from .search import NoFiniteCandidateError, ov_scales, search_alpha_beta, smooth_scale

__all__ = ["FlexSmoothModifier"]


class FlexSmoothModifier(Modifier):
    """Search independent alpha/beta and fuse norm-linear or MLA V/O scales.

    This modifier preserves floating-point behavior and performs no quantization.
    Search uses a symmetric INT8 reconstruction proxy regardless of the eventual
    quantization recipe. Place it after QuaRot and before weight quantization.

    :param alpha: fixed alpha, used only when beta is also specified. If either
        value is omitted both are searched on the reference's 0:0.05:1 grid.
    :param beta: fixed beta, used only together with alpha.
    :param subgraphs: GLM subgraph types to enable. OV executes before norm-linear.
    :param max_tokens: cache the first N tokens per subgraph on CPU. None retains
        all tokens, matching reference collection. A cap changes the search sample
        AND the applied activation statistics; diagnostics report seen/used counts.
    :param on_degenerate: error if every candidate is nonfinite, or explicitly
        record an identity fallback. Nonfinite inputs/weights always fail.

    Initial scope is unsharded GLM MLA with expanded KV-B heads, no MTP. Calibration
    hooks are released on modifier failures and finalize. Persisted config prevents
    reapplying a new modifier; an interrupted/failed transform requires a fresh model.
    """

    requires_calibration_data: Literal[True] = True
    alpha: float | None = Field(default=None, ge=0, le=1)
    beta: float | None = Field(default=None, ge=0, le=1)
    subgraphs: tuple[Literal["norm-linear", "ov"], ...] = ("norm-linear", "ov")
    max_tokens: int | None = Field(default=None, strict=True, gt=0)
    on_degenerate: Literal["error", "identity"] = "error"

    _plan: tuple[SmoothMapping, ...] = PrivateAttr(default=())
    _model_ref: weakref.ReferenceType | None = PrivateAttr(default=None)
    _cache: dict[str, list[torch.Tensor]] = PrivateAttr(default_factory=dict)
    _seen: dict[str, int] = PrivateAttr(default_factory=dict)
    _used: dict[str, int] = PrivateAttr(default_factory=dict)
    _processed: set[str] = PrivateAttr(default_factory=set)
    _diagnostics: dict[str, dict] = PrivateAttr(default_factory=dict)
    _collecting: bool = PrivateAttr(default=False)
    _complete: bool = PrivateAttr(default=False)

    @field_validator("subgraphs")
    @classmethod
    def validate_subgraphs(cls, value):
        if not value or len(set(value)) != len(value):
            raise ValueError("Select nonempty, distinct FlexSmooth subgraph types")
        return value

    @field_serializer("subgraphs")
    def serialize_subgraphs(self, value):
        return list(value)

    @property
    def diagnostics(self) -> dict[str, dict]:
        """Detached per-subgraph parameters, scales, losses and cache counts."""
        return deepcopy(self._diagnostics)

    def _check_model(self, model):
        if self._model_ref is None or self._model_ref() is not model:
            original = self._model_ref() if self._model_ref is not None else None
            if original is not None and self._collecting:
                self._fail(original)
            else:
                self._release()
            raise ValueError("FlexSmooth lifecycle model changed")
        if (getattr(model.config, "flex_smooth_config", None) or {}).get(
            "status"
        ) == "failed":
            raise ValueError("Failed FlexSmooth model requires a fresh model")

    def _preflight(self, model):
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError("Distributed FlexSmooth is not supported")
        with OffloadCache.disable_onloading():
            plan = glm_mappings(model, self.subgraphs)
            targets = {name for item in plan for name in item.targets}
            aliases = {}
            for name, parameter in model.named_parameters(remove_duplicate=False):
                key = (
                    (str(parameter.device), parameter.untyped_storage().data_ptr())
                    if not parameter.is_meta
                    and parameter.layout == torch.strided
                    and parameter.numel()
                    else id(parameter)
                )
                aliases.setdefault(key, []).append(name)
            for names in aliases.values():
                if len(names) > 1 and any(
                    name.rsplit(".", 1)[0] in targets for name in names
                ):
                    raise ValueError(
                        f"Shared FlexSmooth parameter storage is unsupported: {names}"
                    )
            for item in plan:
                dtypes = set()
                for name in item.targets:
                    module = model.get_submodule(name)
                    if getattr(module, "quantization_status", None) not in (
                        None,
                        "initialized",
                        "calibration",
                    ):
                        raise ValueError(
                            "FlexSmooth must run before weight quantization"
                        )
                    for parameter in module.parameters(recurse=False):
                        if parameter.layout != torch.strided:
                            raise ValueError(f"{name}: dense parameters required")
                        if parameter.dtype not in (
                            torch.float32,
                            torch.bfloat16,
                            torch.float16,
                            torch.float64,
                        ):
                            raise ValueError(f"{name}: floating parameters required")
                        if parameter.is_meta and not isinstance(
                            module._parameters, OffloadCache
                        ):
                            raise ValueError(f"{name}: unmaterialized meta parameter")
                    if str(get_execution_device(module)) in ("meta", "disk"):
                        raise ValueError(f"{name}: execution device required")
                    if name in item.consumers:
                        dtypes.add(module.weight.dtype)
                if len(dtypes) != 1:
                    raise ValueError(
                        "Mixed consumer dtypes are unsupported by the reference search"
                    )
        return plan

    def on_initialize(self, state: State, **kwargs) -> bool:
        if getattr(state.model.config, "flex_smooth_config", None) is not None:
            raise ValueError("Model already has FlexSmooth state; use a fresh model")
        if state.loss_masks is not None:
            raise ValueError("FlexSmooth loss masking is not supported")
        self._plan = self._preflight(state.model)
        self._model_ref = weakref.ref(state.model)
        return True

    def _release(self):
        self.remove_hooks()
        self._cache.clear()
        self._collecting = False

    def _fail(self, model):
        if not getattr(model.config, "flex_smooth_config", None):
            model.config.flex_smooth_config = {"version": 1}
        model.config.flex_smooth_config["status"] = "failed"
        self._release()

    def _capture(self, key, model):
        @torch.no_grad()
        def hook(module, args, kwargs):
            if not self._collecting or key in self._processed:
                return
            try:
                tensor = args[0] if args else kwargs.get("input")
                if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                    raise ValueError(
                        f"{key}: expected tensor input with a channel axis"
                    )
                values = tensor.detach().reshape(-1, tensor.shape[-1])
                count = values.shape[0]
                self._seen[key] = self._seen.get(key, 0) + count
                if not count:
                    return
                if not torch.isfinite(values).all():
                    raise ValueError(f"{key}: nonfinite calibration activations")
                used = self._used.get(key, 0)
                remaining = (
                    count if self.max_tokens is None else max(0, self.max_tokens - used)
                )
                values = values[:remaining]
                if not values.numel():
                    return
                self._cache.setdefault(key, []).append(values.to("cpu", copy=True))
                self._used[key] = used + values.shape[0]
            except Exception:
                self._fail(model)
                raise

        return hook

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model
        self._check_model(model)
        if self._collecting or self._complete:
            return
        if state.loss_masks is not None:
            raise ValueError("FlexSmooth loss masking is not supported")
        if self._preflight(model) != self._plan:
            raise ValueError("FlexSmooth topology changed after initialization")
        if (getattr(model.config, "quarot_config", None) or {}).get("status") in (
            "failed",
            "in_progress",
        ):
            raise ValueError("QuaRot must complete before FlexSmooth calibration")
        model.config.flex_smooth_config = {
            "version": 1,
            "status": "collecting",
            "proxy": "symmetric_int8",
            "max_tokens": self.max_tokens,
            "subgraphs": list(self.subgraphs),
            "results": {},
        }
        self._collecting = True
        try:
            for item in self._plan:
                self.register_hook(
                    model.get_submodule(item.consumers[0]),
                    self._capture(item.source, model),
                    "forward_pre",
                    with_kwargs=True,
                )
        except Exception:
            self._fail(model)
            raise

    @torch.no_grad()
    def _smooth(self, model, item):
        first = model.get_submodule(item.consumers[0])
        device = get_execution_device(first)
        weights = []
        for name in item.consumers:
            module = model.get_submodule(name)
            with align_module_device(module, device):
                weights.append(module.weight.detach().clone())
        weight = torch.cat(weights, dim=0)
        act = torch.cat(self._cache[item.source]).to(device=device, dtype=weight.dtype)
        if not torch.isfinite(act).all():
            raise ValueError(
                f"{item.source}: nonfinite activations after dtype conversion"
            )
        if not torch.isfinite(weight).all():
            raise ValueError(f"{item.source}: nonfinite weights")
        alpha, beta = self.alpha, self.beta
        result, fallback = None, None
        if alpha is None or beta is None:
            try:
                result = search_alpha_beta(act, weight)
                alpha, beta = result.alpha, result.beta
            except NoFiniteCandidateError:
                if self.on_degenerate != "identity":
                    raise
                fallback = "no_finite_candidate"
                alpha = beta = None
                logger.warning(f"FlexSmooth identity fallback: {item.source}")
        if fallback:
            scale = torch.ones(act.shape[1], device=device, dtype=weight.dtype)
        elif item.kind == "norm-linear":
            scale = smooth_scale(act.abs().amax(0), weight.abs().amax(0), alpha, beta)
        else:
            scale, _ = ov_scales(
                act.abs().amax(0),
                weight.abs().amax(0),
                alpha,
                beta,
                item.heads,
                item.heads,
            )
        apply_scales(model, item, scale)
        details = {
            "alpha": alpha,
            "beta": beta,
            "loss": result.loss if result else None,
            "fallback": fallback,
            "tokens_seen": self._seen[item.source],
            "tokens_used": self._used[item.source],
            "nonfinite_candidates": sum(
                not math.isfinite(value)
                for value in (*result.alpha_losses, *result.beta_losses)
            )
            if result
            else None,
        }
        model.config.flex_smooth_config["results"][item.source] = details
        self._diagnostics[item.source] = {**details, "scale": scale.detach().cpu()}
        self._processed.add(item.source)
        del self._cache[item.source]

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules=None, **kwargs
    ):
        model = state.model
        self._check_model(model)
        if not self._collecting:
            return
        members = set(model.modules() if modules is None else modules)
        try:
            for item in self._plan:
                if item.source in self._processed or item.source not in self._cache:
                    continue
                participants = {model.get_submodule(name) for name in item.targets}
                if not participants & members:
                    continue
                if not participants <= members:
                    raise ValueError(
                        "Sequential partition splits FlexSmooth subgraph: "
                        f"{item.source}"
                    )
                self._smooth(model, item)
        except Exception:
            self._fail(model)
            raise

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self._check_model(state.model)
        if self._complete:
            return
        try:
            self.on_sequential_epoch_end(state, event)
            missing = [
                item.source for item in self._plan if item.source not in self._processed
            ]
            if missing:
                raise ValueError(
                    f"No calibration inputs for FlexSmooth subgraphs: {missing}"
                )
            state.model.config.flex_smooth_config["status"] = "applied"
            self._complete = True
        except Exception:
            self._fail(state.model)
            raise
        finally:
            self._release()

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._release()
        if not self._complete:
            if getattr(state.model.config, "flex_smooth_config", None) is not None:
                state.model.config.flex_smooth_config["status"] = "failed"
            raise ValueError("FlexSmooth finalized before successful calibration")
        self._plan = ()
        self._model_ref = None
        return True
