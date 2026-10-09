from dataclasses import dataclass, field

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    AWQ_DYNAMIC_MAPPING_REGISTRY,
)
from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
)
from llmcompressor.modifiers.transform.flatquant.components import GeneralMatrixTrans

__all__ = [
    "ATTN_LINEAR_LINEAR",
    "FLATQUANT_MAPPING_REGISTRY",
    "MLP_LINEAR_LINEAR",
    "NORM_LINEAR",
    "PAIR_KINDS",
    "FlatQuantMapping",
    "ResolvedFlatQuantMapping",
    "default_mappings",
    "get_layer_mappings_from_model",
]

# pair kinds, mirroring the structure pairs supported by msmodelslim's
# FlatQuant implementation:
#   - norm -> linears (e.g. input_layernorm -> q/k/v_proj): the transform is
#     applied online to the norm output and absorbed (inverse-transpose) into
#     each target weight; the learnable diagonal is absorbed into the norm
#     weight
#   - linear -> linear through attention (v_proj -> o_proj): the within-head
#     (right) factor is absorbed into the v_proj weight and bias, the
#     across-head (left) factor stays online at the o_proj input, and the
#     full transform is absorbed into the o_proj weight
#   - linear -> linear through the MLP nonlinearity (up_proj -> down_proj):
#     the full transform stays online at the down_proj input and is absorbed
#     into the down_proj weight; the learnable diagonal is absorbed into the
#     up_proj weight
NORM_LINEAR = "norm_linear"
ATTN_LINEAR_LINEAR = "attn_linear_linear"
MLP_LINEAR_LINEAR = "mlp_linear_linear"
PAIR_KINDS = (NORM_LINEAR, ATTN_LINEAR_LINEAR, MLP_LINEAR_LINEAR)


@dataclass
class FlatQuantMapping:
    """
    Dataclass storing the config of a single FlatQuant structure pair.

    The output of ``source_layer`` feeds into each of the ``target_layers``.
    A learnable Kronecker-factored transform is optimized for the pair
    during calibration and reparameterized (absorbed into the involved
    weights) afterwards, leaving the module outputs mathematically
    unchanged while reshaping activation/weight distributions to be more
    quantization-friendly.

    ``FlatQuantMapping``s are resolved into ``ResolvedFlatQuantMapping``s,
    which retain pointers to the actual ``torch.nn.Module``s, the
    constructed transform, and additional metadata at runtime.

    :param source_layer: regex or name of the source module. A
        normalization layer for ``norm_linear`` pairs, or a linear layer
        for the linear-linear pair kinds. If regex is used, modules match
        with the largest overlap in module name
    :param target_layers: list of regex or names of the linear layers
        whose inputs are transformed by the pair
    :param pair_kind: one of ``norm_linear``, ``attn_linear_linear``
        (v_proj -> o_proj style: the right factor is absorbed into the
        source weight/bias and the left factor stays online at the target
        input) or ``mlp_linear_linear`` (up_proj -> down_proj style: the
        full transform stays online at the target input)
    """

    source_layer: str
    target_layers: list[str]
    pair_kind: str

    def __post_init__(self):
        if self.pair_kind not in PAIR_KINDS:
            raise ValueError(
                f"Invalid flatquant pair_kind: {self.pair_kind}. "
                f"Must be one of {PAIR_KINDS}"
            )


default_mappings = [
    FlatQuantMapping(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        pair_kind=NORM_LINEAR,
    ),
    FlatQuantMapping("re:.*v_proj$", ["re:.*o_proj$"], pair_kind=ATTN_LINEAR_LINEAR),
    FlatQuantMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
        pair_kind=NORM_LINEAR,
    ),
    FlatQuantMapping(
        "re:.*up_proj$", ["re:.*down_proj$"], pair_kind=MLP_LINEAR_LINEAR
    ),
]


def _is_supported_target(target: str) -> bool:
    """
    Router/gate targets (``mlp.gate$``, ``mlp.router$``) are structurally
    valid norm-fed linears, but routers are kept in high precision and are
    outside the scope of the FlatQuant blueprint, so they are dropped
    """
    return not (target.endswith("mlp.gate$") or target.endswith("mlp.router$"))


def _from_awq_mapping(mapping: AWQMapping) -> FlatQuantMapping | None:
    """
    Convert an AWQMapping into a FlatQuantMapping, inferring the pair kind
    from the source layer pattern. Returns None for AWQ mappings which have
    no FlatQuant analog:

    - activation sources (e.g. ``gelu_impl``): FlatQuant linear-linear pairs
      require a Linear source, since the learnable diagonal is absorbed into
      the source weight
    - fused sources (``qkv_proj``, ``gate_up_proj``): absorbing into or out
      of a fused projection would require slice-aware handling of the
      merged weight, which is not implemented
    - MLA layernorms (``q_a_layernorm``, ``kv_a_layernorm``):
      ``kv_a_layernorm`` only normalizes a slice of its target's input
      (structurally invalid as a norm_linear pair) and both are beyond the
      structure pairs supported by the msmodelslim blueprint
    """
    source = mapping.smooth_layer
    if "norm" in source:
        if "q_a" in source or "kv_a" in source:
            return None
        pair_kind = NORM_LINEAR
    elif source.endswith("v_proj$") and "qkv" not in source:
        pair_kind = ATTN_LINEAR_LINEAR
    elif source.endswith("up_proj$") and "gate_up" not in source:
        pair_kind = MLP_LINEAR_LINEAR
    else:
        return None

    targets = [t for t in mapping.balance_layers if _is_supported_target(t)]
    if not targets:
        return None

    return FlatQuantMapping(source, targets, pair_kind)


def _convert_awq_mappings(
    awq_mappings: list[AWQMapping] | None,
) -> list[FlatQuantMapping]:
    if awq_mappings is None:
        return []
    return [
        converted
        for converted in (_from_awq_mapping(mapping) for mapping in awq_mappings)
        if converted is not None
    ]


# FlatQuant supports a strict subset of the AWQ structure pairs, so rather
# than duplicating the (large, architecture-specific) mapping tables, the
# registry is derived from the AWQ registry by kind inference. Entries
# which do not map onto a supported structure pair are dropped.
FLATQUANT_MAPPING_REGISTRY: dict[str, list[FlatQuantMapping]] = {
    arch: converted
    for arch, awq_mappings in AWQ_MAPPING_REGISTRY.items()
    if (converted := _convert_awq_mappings(awq_mappings))
}


def get_layer_mappings_from_model(model: Module) -> list[FlatQuantMapping]:
    """
    Infer FlatQuant mappings from a model. Checks the AWQ dynamic mapping
    registry first (for models needing runtime-generated mappings), then
    the converted static registry, then falls back to default mappings.

    :param model: the model to infer mappings for
    :return: list of FlatQuantMapping for the model
    """
    model_name = model.__class__.__name__

    if model_name in AWQ_DYNAMIC_MAPPING_REGISTRY:
        awq_mappings = AWQ_DYNAMIC_MAPPING_REGISTRY[model_name](model)
        converted = _convert_awq_mappings(awq_mappings)
        if converted:
            return converted

    if model_name in FLATQUANT_MAPPING_REGISTRY:
        return FLATQUANT_MAPPING_REGISTRY[model_name]

    logger.info(
        f"Architecture {model_name} not found in flatquant mappings. "
        f"Using default mappings: {default_mappings}"
    )
    return default_mappings


@dataclass
class ResolvedFlatQuantMapping:
    """
    Dataclass for storing a resolved mapping between a source module and
    the target linear modules whose inputs it feeds, along with the
    constructed transform for the pair.

    :param pair_kind: one of the ``PAIR_KINDS``
    :param source_name: name of the source module
    :param source_module: PyTorch module of the source (norm or linear)
    :param target_names: list of names of the target modules
    :param target_modules: list of PyTorch modules of the targets
    :param parent: lowest common ancestor of source and targets, avoiding
        ``torch.nn.ModuleList`` parents (whose forward is not directly
        called in MoE models)
    :param parent_name: name of the parent module
    :param trans: the GeneralMatrixTrans constructed for this pair
    """

    pair_kind: str
    source_name: str
    source_module: Module
    target_names: list[str]
    target_modules: list[Module]
    parent: Module
    parent_name: str
    trans: GeneralMatrixTrans | None = field(default=None)
