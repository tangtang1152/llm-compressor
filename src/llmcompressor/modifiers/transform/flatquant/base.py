"""
FlatQuant: learnable Kronecker-factored activation transforms, as described
in https://arxiv.org/pdf/2502.09796 (ported from msmodelslim).
"""

import inspect
from dataclasses import dataclass, field

import torch
from compressed_tensors.offload.dist_utils import is_distributed
from compressed_tensors.quantization import DynamicType
from compressed_tensors.utils import (
    align_modules,
    get_execution_device,
    match_modules_set,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, field_validator
from torch.nn import Module, Parameter
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import Event, State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.transform.awq.base import (
    get_lowest_common_ancestor_with_avoid,
)
from llmcompressor.modifiers.transform.flatquant.components import (
    ActivationQuantizer,
    GeneralMatrixTrans,
    WeightQuantizer,
    get_decompose_dim,
    get_init_scale,
)
from llmcompressor.modifiers.transform.flatquant.mappings import (
    ATTN_LINEAR_LINEAR,
    MLP_LINEAR_LINEAR,
    NORM_LINEAR,
    FlatQuantMapping,
    ResolvedFlatQuantMapping,
    get_layer_mappings_from_model,
)
from llmcompressor.modifiers.transform.flatquant.patch import (
    FLATQUANT_STATE_ATTR,
    MODE_CALIB,
    MODE_EVAL,
    MODE_ORG,
    FlatQuantState,
    get_patched_modules,
    patch_flatquant_linear,
    patch_flatquant_norm,
    register_flatquant_trans,
    set_flatquant_mode,
    unpatch_flatquant_module,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import get_module_to_name_dict

__all__ = ["FlatQuantModifier"]

# parameter-name substrings of the transform / quantizer parameters, used to
# collect trainable parameters (mirrors msmodelslim's get_trainable_parameters)
BASE_LR_KEYS = ("linear_u", "linear_v", "trans_linear", "linear_diag", "diag_scale")
CLIP_LR_KEYS = ("clip_factor",)
TRAINABLE_KEYS = BASE_LR_KEYS + CLIP_LR_KEYS


def _is_ancestor_name(ancestor: str, descendant: str) -> bool:
    """
    Return True if the module name ``ancestor`` refers to an ancestor of (or
    the same module as) the module named ``descendant``.
    """
    return descendant == ancestor or descendant.startswith(ancestor + ".")


@dataclass
class TrainingUnit:
    """
    A group of structure-pair mappings which are trained together.

    Mappings are first grouped by the lowest common ancestor of all modules
    they involve (source + targets), so that the ancestor's forward replay
    covers every transform of the group. Groups whose ancestors are nested
    (e.g. ``self_attn`` inside a decoder layer) are then merged into the
    outermost ancestor, yielding one unit per transformer layer, which is
    the granularity at which the msmodelslim blueprint trains.

    :param parent: module whose forward is replayed to extract fp teacher
        outputs and to train the transforms
    :param parent_name: name of the parent module
    :param mappings: resolved mappings merged into this unit
    :param modules: all patched modules below parent, refreshed after
        patching (norms and linears carrying a FlatQuantState)
    """

    parent: Module
    parent_name: str
    mappings: list[ResolvedFlatQuantMapping] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)


class FlatQuantModifier(Modifier):
    """
    Implements the FlatQuant algorithm, as described in
    https://arxiv.org/pdf/2502.09796, ported from msmodelslim.

    For each supported structure pair (norm -> linears, v_proj -> o_proj,
    up_proj -> down_proj), a learnable transform ``A = kron(L, R) * diag(d)``
    is applied online to the activations feeding the pair's target linears,
    while ``A^-T`` is applied to the target weights. The transform parameters
    (orthogonal Kronecker factors, learnable diagonal, activation/weight clip
    factors) are optimized with a per-layer MSE loss against the fp outputs of
    the layer, using the cached calibration data. After training, the weight
    transforms are absorbed (in fp64) into the target weights, the diagonal is
    absorbed into the source module's weight, and the remaining online factors
    are frozen into buffers, leaving every module output mathematically
    unchanged while reshaping activation/weight distributions to be more
    quantization-friendly.

    Like AWQModifier, this is a transform-based modifier: it does not
    quantize anything by itself and must be applied before a modifier which
    inherits from QuantizationMixin, which will then observe the transformed
    weights. During training, fake-quantizers matching the intended
    downstream scheme (``w_bits``/``a_bits``/symmetry/lwc/lac) are used to
    shape the transforms; they are disabled once training completes, so the
    eval-mode forward is mathematically equivalent to the original.

    Example recipe:
    ```yaml
    - FlatQuantModifier:
        epochs: 15
        lr: 5.0e-3
    - QuantizationModifier:
        scheme: W4A4
        targets: Linear
    ```

    Lifecycle:

    - on_initialize
        - set unresolved mappings if not set by user, based on architecture
    - on_calibration_start
        - resolve mappings against the model, constructing transforms
        - patch the forward of all involved norms/linears (org mode)
        - freeze model parameters inside the training units
        - attach forward-arg cache hooks on each unit parent and input-max
            statistics hooks on each target linear
    - on_sequential_epoch_end
        - for each unit whose parent ran during this subgraph pass:
            - extract fp teacher outputs by replaying cached parent args
            - initialize learnable diagonals from weight/activation maxima
            - train all transforms of the unit jointly (one step per sample)
            - absorb the transforms into the involved weights and freeze
    - on_calibration_end
        - raise if any cached data was left unconsumed
        - remove hooks (falling back any never-executed units to original)
    - on_finalize
        - clear all runtime state

    Deviations from the msmodelslim blueprint, all noted for maintainers:

    - modules are forward-patched in place rather than replaced, so the
        quantization config applied by QuantizationMixin is unaffected
    - the fp teacher outputs are computed in the model's native dtype rather
        than casting each layer to fp32
    - the learnable diagonal is initialized from the elementwise max of the
        input-max statistics over all targets of the pair (the blueprint leaks
        a loop variable and uses the last target's statistics)
    - the diagonal absorption also scales the source bias (if any), which is
        required for exactness but omitted in the blueprint
    - distributed execution is not supported (parameters would diverge across
        ranks without gradient synchronization)

    :param mappings: list of structure pairs to transform. Each entry pairs a
        source module (norm or linear) with the target linears whose inputs it
        feeds, plus the pair kind. If regex is used, modules match with the
        largest overlap in module name. If unset, mappings are inferred from
        the model architecture
    :param epochs: number of epochs over the calibration data per unit
    :param lr: base learning rate for the transform parameters (AdamW). Clip
        factors use 10x this value
    :param diag_alpha: interpolation between weight (0.0) and activation
        (1.0) magnitudes used to initialize the learnable diagonal
    :param add_diag: include a learnable diagonal in the transforms of
        norm_linear and mlp_linear_linear pairs (the v_proj -> o_proj pair
        never has one)
    :param diag_relu: keep the learnable diagonal and SVD singular values
        positive via softplus
    :param tran_type: parameterization of the Kronecker factors, "svd"
        (orthogonal via Cayley) or "inv" (directly learnable invertible)
    :param warmup: prepend a short linear lr warmup to the cosine schedule
    :param deactive_amp: disable autocast during training (train in the
        model's native dtype instead)
    :param amp_dtype: dtype used by autocast during training
    :param seed: seed applied before transform construction, for
        reproducible orthogonal initializations
    :param w_bits: weight bit-width assumed by the fake weight quantizer
        used during training
    :param a_bits: activation bit-width assumed by the fake activation
        quantizer used during training
    :param w_sym: symmetric fake weight quantization during training
    :param a_sym: symmetric fake activation quantization during training
    :param lwc: learnable weight clipping during training
    :param lac: learnable activation clipping during training
    :param offload_device: device to which cached parent args are offloaded,
        which reduces memory requirements but requires more time to move
        data. Defaults to None (no offloading), or "cpu" for MoE models
    """

    requires_calibration_data: bool = True

    # Allow arbitrary types because ResolvedFlatQuantMapping has Module fields
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    mappings: list[FlatQuantMapping] | None = None
    epochs: int = 15
    lr: float = 5e-3
    diag_alpha: float = 0.3
    add_diag: bool = True
    diag_relu: bool = True
    tran_type: str = "svd"
    warmup: bool = False
    deactive_amp: bool = False
    amp_dtype: str = "bfloat16"
    seed: int = 0
    w_bits: int = 4
    a_bits: int = 4
    w_sym: bool = True
    a_sym: bool = True
    lwc: bool = True
    lac: bool = True
    offload_device: torch.device | None | Sentinel = Sentinel("not_provided")

    # Private vars set during calibration, cleared during finalization
    _resolved_mappings: list[ResolvedFlatQuantMapping] = PrivateAttr(
        default_factory=list
    )
    _training_units: list[TrainingUnit] = PrivateAttr(default_factory=list)
    # cache of forward args for each unit parent, one dict per batch
    _parent_args_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # per-target running max of |input|, used to initialize the diagonals
    _act_stats: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)

    @field_validator("amp_dtype")
    @classmethod
    def validate_amp_dtype(cls, value: str) -> str:
        if value not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be 'bfloat16' or 'float16', got {value}")
        return value

    @field_validator("tran_type")
    @classmethod
    def validate_tran_type(cls, value: str) -> str:
        if value not in ("svd", "inv"):
            raise ValueError(f"tran_type must be 'svd' or 'inv', got {value}")
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Start FlatQuant on the given state. This runs before quantization
        config has been applied.

        - infer unresolved mappings based on model architecture, if not set
        - set default offload device

        :param state: state to run FlatQuant on
        :return: True on a successful run, False otherwise
        """
        if self.mappings is None:
            logger.info("No FlatQuantModifier.mappings provided, inferring from model")
            self.mappings = get_layer_mappings_from_model(state.model)

        if self.offload_device == Sentinel("not_provided"):
            if is_moe_model(state.model):
                self.offload_device = torch.device("cpu")
                logger.info(
                    "MoE model detected: setting offload_device to 'cpu' by default "
                    "to reduce memory usage. You can override this by explicitly "
                    "setting offload_device in your recipe."
                )
            else:
                self.offload_device = None

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        """
        Start FlatQuant calibration. This runs after quantization config has
        been applied.

        - resolve mappings and construct transforms
        - validate downstream quantization schemes
        - build training units and patch all involved modules
        - attach parent-args cache hooks and input statistics hooks
        """
        if is_distributed():
            raise NotImplementedError(
                "FlatQuantModifier does not support distributed execution: the "
                "transform parameters would be trained independently on each "
                "rank without gradient synchronization"
            )

        torch.manual_seed(self.seed)

        self._resolve_mappings(state.model)
        self._validate_downstream_schemes()
        self._build_training_units()
        self._patch_units()
        self._setup_hooks()

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """
        Train and absorb the transforms of every unit whose parent module
        executed during this sequential pass. Units of other subgraphs are
        identified by their (still empty) args cache and skipped.
        """
        for unit in self._training_units:
            cache = self._parent_args_cache.get(unit.parent)
            if cache is not None and len(cache) > 0:
                self._train_unit(state.model, unit)

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by verifying all cached data was consumed, falling
        any never-executed unit back to its original forward, and removing
        hooks.
        """
        for cache in self._parent_args_cache.values():
            if len(cache) > 0:
                raise RuntimeError("Some cached activations were not used")

        for unit in self._training_units:
            if unit.parent in self._parent_args_cache:
                # parent never executed (e.g. calibration data did not reach
                # it): fall back to the original forward
                self._unpatch_unit(unit)

        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up all runtime state. The absorbed weights and frozen transform
        buffers stay attached to the model and are persisted with its
        state_dict.

        :param state: unused
        :return: True
        """
        self._parent_args_cache.clear()
        self._act_stats.clear()
        self._training_units.clear()
        self._resolved_mappings.clear()

        return True

    ###############################
    # Mapping and unit resolution #
    ###############################

    def _resolve_mappings(self, model: Module) -> None:
        """
        Resolve mappings against the model, constructing the transform of
        each pair. Mappings with incompatible shapes or untrainable
        dimensions are skipped with a warning (e.g. hidden sizes which admit
        no Kronecker decomposition).
        """
        module_to_name = get_module_to_name_dict(model)
        resolved: list[ResolvedFlatQuantMapping] = []
        num_skipped = 0

        for mapping in self.mappings:
            for source_modules, *nested_target_modules in match_modules_set(
                model, (mapping.source_layer, *mapping.target_layers)
            ):
                if len(source_modules) > 1:
                    raise ValueError(
                        "FlatQuant needs to match a single source module for each "
                        f"mapping but got {len(source_modules)} for mapping: "
                        f"{mapping}"
                    )
                source = source_modules[0]
                source_name = module_to_name[source]

                targets = tree_leaves(nested_target_modules)
                target_names = [module_to_name[target] for target in targets]

                if len(targets) == 0:
                    logger.warning(
                        f"Skipping {source_name} for mapping {mapping} because no "
                        "target layers were found."
                    )
                    continue

                if not any(
                    hasattr(module, "quantization_scheme")
                    for module in (source, *targets)
                ):
                    logger.warning(
                        f"Skipping {source_name} for mapping {mapping} because no "
                        "modules are targeted for quantization. FlatQuant must be "
                        "run with a subsequent quantization modifier."
                    )
                    continue

                trans = self._build_trans(model, mapping, source, targets)
                if trans is None:
                    num_skipped += 1
                    continue

                ancestor_name, ancestor = get_lowest_common_ancestor_with_avoid(
                    [source_name, *target_names], model, torch.nn.ModuleList
                )

                resolved.append(
                    ResolvedFlatQuantMapping(
                        pair_kind=mapping.pair_kind,
                        source_name=source_name,
                        source_module=source,
                        target_names=target_names,
                        target_modules=targets,
                        parent=ancestor,
                        parent_name=ancestor_name,
                        trans=trans,
                    )
                )

        if num_skipped > 0:
            logger.warning(
                f"{num_skipped} mappings were skipped due to incompatible shapes "
                "or unsupported dimensions. Set logging to DEBUG to view the "
                "full list."
            )
        if len(resolved) == 0:
            raise ValueError(
                "No FlatQuant mappings were resolved against the model. Check "
                f"the mappings argument: {self.mappings}"
            )

        self._resolved_mappings = resolved

    def _build_trans(
        self, model: Module, mapping: FlatQuantMapping, source: Module, targets: list
    ) -> GeneralMatrixTrans | None:
        """
        Construct the GeneralMatrixTrans for a resolved structure pair,
        validating shapes. Returns None (with a debug log) if the pair is not
        trainable, e.g. because a dimension admits no Kronecker decomposition.
        """
        for target in targets:
            if not isinstance(target, torch.nn.Linear):
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: target "
                    f"{type(target)} is not a Linear"
                )
                return None

        if mapping.pair_kind == NORM_LINEAR:
            if source.weight.dim() != 1:
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: source {type(source)} "
                    "does not have a 1D weight"
                )
                return None
            dims = get_decompose_dim(source.weight.shape[0])
            if dims is None:
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: hidden size "
                    f"{source.weight.shape[0]} admits no Kronecker decomposition "
                    "(== 2 mod 4)"
                )
                return None
            return GeneralMatrixTrans(
                *dims,
                add_diag=self.add_diag,
                diag_relu=self.diag_relu,
                tran_type=self.tran_type,
            )

        if mapping.pair_kind == ATTN_LINEAR_LINEAR:
            if not isinstance(source, torch.nn.Linear):
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: source "
                    f"{type(source)} is not a Linear"
                )
                return None
            head_dim = self._head_dim(model, source, targets[0])
            if head_dim is None:
                return None
            # the left factor acts across the heads of the target input, which
            # correctly handles GQA (where v_proj has fewer heads than o_proj)
            left_size = targets[0].in_features // head_dim
            return GeneralMatrixTrans(
                left_size,
                head_dim,
                add_diag=False,
                diag_relu=self.diag_relu,
                tran_type=self.tran_type,
            )

        # MLP_LINEAR_LINEAR
        if not isinstance(source, torch.nn.Linear):
            logger.debug(
                f"Skipping FlatQuant mapping {mapping}: source {type(source)} "
                "is not a Linear"
            )
            return None
        dims = get_decompose_dim(source.weight.shape[0])
        if dims is None:
            logger.debug(
                f"Skipping FlatQuant mapping {mapping}: source output size "
                f"{source.weight.shape[0]} admits no Kronecker decomposition "
                "(== 2 mod 4)"
            )
            return None
        return GeneralMatrixTrans(
            *dims,
            add_diag=self.add_diag,
            diag_relu=self.diag_relu,
            tran_type=self.tran_type,
        )

    def _head_dim(self, model: Module, source: Module, target: Module) -> int | None:
        """
        Resolve the attention head dimension for a v_proj -> o_proj pair,
        following the msmodelslim blueprint: the model config's head_dim if
        present, else hidden_size // num_attention_heads.
        """
        config = getattr(model, "config", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_attention_heads", None)
            if hidden_size is not None and num_heads:
                head_dim = hidden_size // num_heads

        if head_dim is None:
            logger.debug(
                "Skipping FlatQuant v_proj -> o_proj mapping: could not resolve "
                "head_dim from the model config"
            )
            return None
        if source.out_features % head_dim or target.in_features % head_dim:
            logger.debug(
                "Skipping FlatQuant v_proj -> o_proj mapping: head_dim "
                f"{head_dim} does not divide v_proj out_features "
                f"{source.out_features} / o_proj in_features {target.in_features}"
            )
            return None
        return head_dim

    def _validate_downstream_schemes(self) -> None:
        """
        The fake activation quantizers used while training are per-token
        dynamic, and the online transforms reshape activations after
        calibration, so static (or locally-static) input activation schemes
        downstream would observe the wrong distributions. Warn in that case.
        """
        static_targets = []
        for mapping in self._resolved_mappings:
            for name, target in zip(mapping.target_names, mapping.target_modules):
                scheme = getattr(target, "quantization_scheme", None)
                activations = getattr(scheme, "input_activations", None)
                if activations is not None and activations.dynamic in (
                    False,
                    DynamicType.LOCAL,
                ):
                    static_targets.append(name)

        if static_targets:
            logger.warning(
                f"Found static input activation quantization schemes on "
                f"{len(static_targets)} FlatQuant targets (e.g. {static_targets[:3]}). "
                "FlatQuant assumes dynamic (per-token) activation quantization "
                "downstream; static activation scales observed after the "
                "transform may be inaccurate."
            )

    def _build_training_units(self) -> None:
        """
        Group resolved mappings into training units: first by the lowest
        common ancestor of all involved modules, then merging groups whose
        ancestors are nested into the outermost ancestor (e.g. the
        self_attn group of a v_proj -> o_proj pair merges into the decoder
        layer group which also holds the norm -> q/k/v pair), yielding one
        unit per transformer layer as in the msmodelslim blueprint.
        """
        units: dict[Module, TrainingUnit] = {}
        for mapping in self._resolved_mappings:
            unit = units.setdefault(
                mapping.parent, TrainingUnit(mapping.parent, mapping.parent_name)
            )
            unit.mappings.append(mapping)

        # merge units whose parents are ancestors of one another, repeating
        # until no parent is nested inside another unit's parent
        merged = True
        while merged:
            merged = False
            for inner_parent, inner_unit in units.items():
                for outer_parent, outer_unit in units.items():
                    if outer_parent is inner_parent:
                        continue
                    if _is_ancestor_name(
                        outer_unit.parent_name, inner_unit.parent_name
                    ):
                        outer_unit.mappings.extend(inner_unit.mappings)
                        del units[inner_parent]
                        merged = True
                        break
                if merged:
                    break

        self._training_units = list(units.values())
        logger.info(
            f"FlatQuant resolved {len(self._resolved_mappings)} mappings into "
            f"{len(self._training_units)} training units"
        )

    ################
    # Patching and #
    # hook set-up  #
    ################

    def _patch_units(self) -> None:
        """
        Patch the forward of every source norm and target linear, register
        per-target fake quantizers (sharing one lac clip factor per pair) and
        the transform homes, then freeze all parameters inside the units
        (trainable parameters are re-enabled at training time).
        """
        for unit in self._training_units:
            for mapping in unit.mappings:
                trans = mapping.trans
                if mapping.pair_kind == NORM_LINEAR:
                    patch_flatquant_norm(mapping.source_module, trans)
                    for target in mapping.target_modules:
                        self._patch_linear(target, weight_in_trans=trans)
                elif mapping.pair_kind == ATTN_LINEAR_LINEAR:
                    # the right factor is applied to the v_proj output (and
                    # absorbed into its weight), the left factor stays online
                    # at the o_proj input, and the full transform is absorbed
                    # into the o_proj weight
                    self._patch_linear(
                        mapping.source_module, weight_out_trans=trans.right_trans
                    )
                    for target in mapping.target_modules:
                        self._patch_linear(
                            target,
                            act_in_trans=trans.left_trans,
                            weight_in_trans=trans,
                        )
                        register_flatquant_trans(target, trans)
                else:  # MLP_LINEAR_LINEAR
                    for target in mapping.target_modules:
                        self._patch_linear(
                            target, act_in_trans=trans, weight_in_trans=trans
                        )
                        register_flatquant_trans(target, trans)

                if self.lac:
                    # one shared clip factor per pair (the blueprint
                    # overrides the quantizer's default 4.0 init with 1.0)
                    clip_factor = Parameter(torch.ones(1) * 1.0, requires_grad=True)
                    for target in mapping.target_modules:
                        state = getattr(target, FLATQUANT_STATE_ATTR)
                        state.act_quantizer.clip_factor = clip_factor

            # freeze all parameters inside the unit; the transforms are
            # re-enabled by name at training time
            for param in unit.parent.parameters():
                param.requires_grad = False

            unit.modules = get_patched_modules(unit.parent)

    def _patch_linear(self, module: Module, **transforms) -> FlatQuantState:
        """
        Patch a linear module, or extend the FlatQuantState of an already
        patched one (e.g. v_proj is patched as a norm-pair target first and
        gains a weight output transform from the v_proj -> o_proj pair).
        """
        state = getattr(module, FLATQUANT_STATE_ATTR, None)
        if state is None:
            state = patch_flatquant_linear(
                module,
                act_quantizer=ActivationQuantizer(
                    self.a_bits, sym=self.a_sym, lac=self.lac
                ),
                weight_quantizer=WeightQuantizer(
                    in_size=module.weight.shape[1],
                    out_size=module.weight.shape[0],
                    bits=self.w_bits,
                    sym=self.w_sym,
                    lwc=self.lwc,
                ),
                **transforms,
            )
        else:
            for attr, trans in transforms.items():
                if trans is not None:
                    setattr(state, attr, trans)
        return state

    def _setup_hooks(self) -> None:
        """
        Attach a forward_pre hook on each unit parent caching the kwargs
        needed to replay its forward, and a forward hook on each target linear
        accumulating per-input-channel |input| maxima (pre-initialized to
        1e-5, as in the blueprint).
        """

        def cache_parent_kwargs_hook(
            module: Module,
            args: tuple,
            kwargs: dict,
        ):
            # bind() nests **kwargs contents under a 'kwargs' key; flatten
            # them so the result can be replayed as module(**cached)
            arguments = (
                inspect.signature(module.forward).bind(*args, **kwargs).arguments
            )
            arguments |= arguments.pop("kwargs", {})

            from compressed_tensors.modeling.kvcache import QuantizedKVCache

            for key, value in arguments.items():
                if isinstance(value, QuantizedKVCache):
                    arguments[key] = None

            # setdefault for robustness: the cache entry is deleted after a
            # unit is trained (to free memory), and no forward of its parent
            # is expected afterwards; if one happens anyway, it re-creates
            # the entry and on_calibration_end raises a descriptive error
            # instead of a KeyError mid-forward
            self._parent_args_cache.setdefault(
                module, IntermediatesCache(None, self.offload_device)
            ).append(arguments)

        def make_stat_hook(name: str):
            def stat_input_hook(_module: Module, args: tuple, _output):
                stats = self._act_stats.get(name)
                if stats is None:
                    # unit was already trained and its stats consumed
                    return
                x = args[0] if isinstance(args, tuple) and args else None
                if not isinstance(x, torch.Tensor):
                    return
                x_max = x.reshape(-1, x.shape[-1]).abs().amax(dim=0)
                torch.maximum(stats, x_max.to(stats.dtype).cpu(), out=stats)

            return stat_input_hook

        for unit in self._training_units:
            if unit.parent not in self._parent_args_cache:
                self._parent_args_cache[unit.parent] = IntermediatesCache(
                    None, self.offload_device
                )
                self.register_hook(
                    unit.parent,
                    cache_parent_kwargs_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

            for mapping in unit.mappings:
                for name, target in zip(mapping.target_names, mapping.target_modules):
                    self._act_stats[name] = torch.full(
                        [target.weight.shape[1]], 1e-5, dtype=target.weight.dtype
                    )
                    self.register_hook(target, make_stat_hook(name), "forward")

    ##################
    # Training units #
    ##################

    def _train_unit(self, model: Module, unit: TrainingUnit) -> None:
        """
        Train all transforms of one unit against fp teacher outputs, then
        absorb them into the model weights.

        1. replay the cached parent args in org mode to extract fp teacher
            outputs (falling back to the original forward if the data is
            empty or non-finite)
        2. initialize the learnable diagonals from weight and activation maxima
        3. train the transforms jointly: one optimizer over all transform and
            clip-factor parameters of the unit, one step per calibration
            sample, MSE loss normalized by its (detached) value
        4. absorb the transforms (fp64 weight absorption, fp32 diagonal
            absorption into the source weight), freeze them into buffers,
            disable the fake quantizers, and switch to eval mode
        """
        parent = unit.parent
        cache = self._parent_args_cache[parent]

        # pin memory for faster onloading during replay
        for batch_index in range(len(cache)):
            cache.pin_memory(batch_index)

        involved = [parent]
        for mapping in unit.mappings:
            involved.append(mapping.source_module)
            involved.extend(mapping.target_modules)

        with (
            align_modules(involved),
            HooksMixin.disable_hooks(),
        ):
            # 1. fp teacher outputs
            set_flatquant_mode(unit.modules, MODE_ORG)
            with calibration_forward_context(model):
                fp_outs = self._run_samples(parent)
            if len(fp_outs) == 0 or all(f.numel() == 0 for f in fp_outs):
                logger.info(
                    f"Skipping FlatQuant unit {unit.parent_name}: no activations "
                    "found to transform. This can occasionally occur in MoE models "
                    "when no calibration data reaches the unit."
                )
                self._unpatch_unit(unit)
                return
            if not all(torch.isfinite(f).all() for f in fp_outs):
                logger.warning(
                    f"Skipping FlatQuant unit {unit.parent_name}: NaN or inf "
                    "outputs found in the fp teacher forward pass. The model may "
                    "be generating NaN outputs with the provided calibration data, "
                    "or the mappings are set incorrectly."
                )
                self._unpatch_unit(unit)
                return
            fp_outs = [f.cpu() for f in fp_outs]

            # 2. diagonal initialization
            self._init_diag_scales(unit)

            # 3. training
            set_flatquant_mode(unit.modules, MODE_CALIB)
            self._train_loop(unit, fp_outs)

            # 4. absorption
            with torch.no_grad():
                self._absorb_unit(unit)
            set_flatquant_mode(unit.modules, MODE_EVAL)

        # remove caches and stats belonging to this unit
        del self._parent_args_cache[parent]
        for mapping in unit.mappings:
            for name in mapping.target_names:
                self._act_stats.pop(name, None)

    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        """
        Replay the cached forward args of a module, one forward pass per
        batch, returning the first output of each pass.
        """
        cache = self._parent_args_cache[module]
        use_prefetch = active_session().state.sequential_prefetch
        batch_iter = cache.iter_prefetch() if use_prefetch else cache
        outputs = [module(**batch) for batch in batch_iter]
        return [
            # if tuple, the first output is the hidden state
            output[0] if isinstance(output, tuple) else output
            for output in outputs
        ]

    def _init_diag_scales(self, unit: TrainingUnit) -> None:
        """
        Initialize the learnable diagonal of each pair transform from the
        per-input-channel maxima of the target weights and of the target
        input activations. Unlike the blueprint (which leaks a loop variable
        and uses only the last target's statistics), the activation maxima of
        all targets of the pair are combined elementwise.
        """
        for mapping in unit.mappings:
            trans = mapping.trans
            if trans.diag_trans is None:
                continue

            weight_max = (
                torch.cat(
                    [target.weight.data.float() for target in mapping.target_modules],
                    dim=0,
                )
                .abs()
                .max(dim=0)[0]
            )
            input_max = torch.stack(
                [self._act_stats[name].float() for name in mapping.target_names],
                dim=0,
            ).max(dim=0)[0]

            diag_scale = trans.diag_trans.diag_scale
            diag_scale.data = get_init_scale(
                weight_max.to(diag_scale), input_max.to(diag_scale), self.diag_alpha
            )

    def _train_loop(self, unit: TrainingUnit, fp_outs: list[torch.Tensor]) -> None:
        """
        Jointly train all transforms of a unit: an AdamW optimizer over the
        transform parameters (base lr) and clip factors (10x lr) of the unit,
        with a cosine annealing schedule over epochs * num_samples steps (and
        an optional linear warmup), one step per calibration sample, under
        autocast. The MSE loss against the fp teacher output is normalized by
        its detached value, so the gradient direction is scale-invariant.
        """
        param_groups = self._get_trainable_parameters(unit.parent, self.lr)
        if not param_groups:
            logger.debug(
                f"FlatQuant unit {unit.parent_name} has no trainable parameters, "
                "skipping training"
            )
            return

        optimizer = torch.optim.AdamW(param_groups)
        num_samples = len(fp_outs)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * num_samples, eta_min=self.lr * 1e-3
        )
        if self.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=16
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [scheduler_warmup, scheduler_main]
            )
        else:
            scheduler = scheduler_main

        loss_fn = torch.nn.MSELoss()
        device = get_execution_device(unit.parent)
        cache = self._parent_args_cache[unit.parent]
        use_prefetch = active_session().state.sequential_prefetch
        amp_dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16

        pbar = tqdm(
            total=self.epochs * num_samples, desc=f"FlatQuant {unit.parent_name}"
        )
        # the sequential pipeline invokes this hook inside a no_grad context
        with torch.enable_grad():
            for _epoch in range(self.epochs):
                epoch_mse = 0.0
                batch_iter = cache.iter_prefetch() if use_prefetch else cache
                for sample_idx, batch in enumerate(batch_iter):
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=not self.deactive_amp,
                    ):
                        output = unit.parent(**batch)
                        quant_output = (
                            output[0] if isinstance(output, tuple) else output
                        )
                        loss = loss_fn(
                            quant_output, fp_outs[sample_idx].to(quant_output.device)
                        )

                    epoch_mse += loss.detach().float().item()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)

                pbar.set_postfix({"mse": f"{epoch_mse / max(num_samples, 1):.3e}"})
                logger.debug(
                    f"FlatQuant unit {unit.parent_name} epoch {_epoch}: "
                    f"MSE loss {epoch_mse:.6e}, lr {scheduler.get_last_lr()[0]:.8f}"
                )
        pbar.close()

    def _get_trainable_parameters(
        self, parent: Module, base_lr: float
    ) -> list[dict]:
        """
        Collect the transform and clip-factor parameters within a unit,
        grouped by name for the optimizer (clip factors train at 10x lr),
        mirroring msmodelslim's get_trainable_parameters. Parameters are
        deduplicated: the lac clip factor is shared across the targets of a
        pair (registered under each target's act quantizer), and passing the
        same parameter to the optimizer multiple times would step it once per
        occurrence.
        """
        params: dict[str, list[Parameter]] = {key: [] for key in TRAINABLE_KEYS}
        seen: set[int] = set()
        for name, param in parent.named_parameters():
            for key in TRAINABLE_KEYS:
                if key in name:
                    param.requires_grad = True
                    if id(param) not in seen:
                        seen.add(id(param))
                        params[key].append(param)
                    break

        param_groups = [
            {"params": params[key], "lr": base_lr}
            for key in BASE_LR_KEYS
            if params[key]
        ]
        if params["clip_factor"]:
            param_groups.append({"params": params["clip_factor"], "lr": base_lr * 10})

        return param_groups

    ##############
    # Absorption #
    ##############

    def _absorb_unit(self, unit: TrainingUnit) -> None:
        """
        Absorb the trained transforms of a unit into the model weights, in
        the order used by the blueprint:

        1. per target linear: apply the weight input transform (inverse
            transpose, in fp64), the learned weight clipping, and the weight
            output transform (including the bias) to the weight, then
            reparameterize the fake quantizers and drop the transform
            references
        2. per pair: absorb the learnable diagonal (in fp32) into the source
            module's weight (and bias, if any)
        3. freeze the Kronecker factors into buffers
        4. disable the fake quantizers, so eval mode is mathematically
            equivalent to the original module and the actual quantization is
            left to the subsequent quantization modifier
        """
        # 1. weight absorption, per patched linear
        for module in unit.modules:
            if isinstance(module, torch.nn.Linear):
                self._absorb_linear(module)

        # 2. diagonal absorption into the source module
        for mapping in unit.mappings:
            trans = mapping.trans
            if trans is None:
                continue

            if trans.diag_trans is not None:
                source = mapping.source_module
                diag = trans.diag_trans.diag_scale.data.to(torch.float32)

                weight = source.weight.data.to(torch.float32)
                if weight.dim() == 1:
                    weight = weight * diag
                elif weight.dim() == 2:
                    weight = weight * diag.unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported source weight dim: {weight.dim()}")
                update_offload_parameter(
                    source, "weight", weight.to(source.weight.dtype)
                )

                bias = getattr(source, "bias", None)
                if bias is not None:
                    update_offload_parameter(
                        source,
                        "bias",
                        (bias.data.to(torch.float32) * diag).to(bias.dtype),
                    )

            # 3. freeze the kron factors (and null the diagonal) into
            #    buffers, for every pair: v_proj -> o_proj pairs carry no
            #    diagonal but their factors must still be frozen
            trans.to_eval_mode()

        # 4. disable fake quantizers: eval forward = online transform +
        #    absorbed weights == original math
        for module in unit.modules:
            state = getattr(module, FLATQUANT_STATE_ATTR)
            if state.act_quantizer is not None:
                state.act_quantizer.enable = False
            if state.weight_quantizer is not None:
                state.weight_quantizer.enable = False
                state.weight_quantizer.enable_find = False

    def _absorb_linear(self, module: Module) -> None:
        """
        Absorb the transforms attached to a patched linear into its weight
        (and bias), in fp64 as in the blueprint, then reparameterize its fake
        quantizers. Idempotent: modules with no weight transforms left (or
        none to begin with) only reparameterize.
        """
        state = getattr(module, FLATQUANT_STATE_ATTR)
        weight_in_trans = state.weight_in_trans
        weight_out_trans = state.weight_out_trans

        if weight_in_trans is not None or weight_out_trans is not None:
            weight = module.weight.data
            ori_dtype = weight.dtype
            weight = weight.to(torch.float64)

            if weight_in_trans is not None:
                weight = weight_in_trans(weight, inv_t=True)
            if state.weight_quantizer.lwc:
                weight = state.weight_quantizer.apply_wclip(weight)
            if weight_out_trans is not None:
                weight = weight_out_trans(weight.T).T
                if module.bias is not None:
                    # the blueprint absorbs the bias in its original dtype
                    update_offload_parameter(
                        module, "bias", weight_out_trans(module.bias.data)
                    )

            update_offload_parameter(module, "weight", weight.to(ori_dtype))
            state.weight_in_trans = None
            state.weight_out_trans = None

        state.act_quantizer.reparameterize()
        state.weight_quantizer.reparameterize()

    def _unpatch_unit(self, unit: TrainingUnit) -> None:
        """
        Restore the original forward of all modules of a unit which was
        skipped (no data, or non-finite teacher outputs), removing the
        transforms and fake quantizers so the modules keep their original
        semantics and state_dict.
        """
        for module in get_patched_modules(unit.parent):
            unpatch_flatquant_module(module)
        self._parent_args_cache.pop(unit.parent, None)
        for mapping in unit.mappings:
            for name in mapping.target_names:
                self._act_stats.pop(name, None)
        logger.info(f"Fell FlatQuant unit {unit.parent_name} back to original forward")
