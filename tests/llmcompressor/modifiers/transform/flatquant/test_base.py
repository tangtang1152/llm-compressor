from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import apply_quantization_config
from pydantic import ValidationError
from torch.nn import Linear
from torch.testing import assert_close

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from llmcompressor.modifiers.transform.flatquant import (
    ATTN_LINEAR_LINEAR,
    MLP_LINEAR_LINEAR,
    NORM_LINEAR,
    FlatQuantMapping,
    FlatQuantModifier,
    default_mappings,
    get_layer_mappings_from_model,
)
from llmcompressor.modifiers.transform.flatquant.components import get_decompose_dim
from llmcompressor.modifiers.transform.flatquant.mappings import _from_awq_mapping
from llmcompressor.modifiers.transform.flatquant.patch import (
    FLATQUANT_STATE_ATTR,
    MODE_CALIB,
    MODE_EVAL,
    MODE_ORG,
    get_patched_modules,
    set_flatquant_mode,
)

HIDDEN = 16
HEAD_DIM = 8
NUM_HEADS = 2  # q_proj/o_proj have 2 heads, k_proj/v_proj have 1 (GQA)
INTERMEDIATE = 32


def _toy_mappings() -> list[FlatQuantMapping]:
    return [
        FlatQuantMapping(
            "re:.*input_layernorm$",
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            pair_kind=NORM_LINEAR,
        ),
        FlatQuantMapping(
            "re:.*v_proj$", ["re:.*o_proj$"], pair_kind=ATTN_LINEAR_LINEAR
        ),
        FlatQuantMapping(
            "re:.*post_attention_layernorm$",
            ["re:.*gate_proj$", "re:.*up_proj$"],
            pair_kind=NORM_LINEAR,
        ),
        FlatQuantMapping(
            "re:.*up_proj$", ["re:.*down_proj$"], pair_kind=MLP_LINEAR_LINEAR
        ),
    ]


def _build_toy_model(with_config: bool = True) -> torch.nn.Module:
    """
    Build a toy single-layer transformer-ish model. Attention is GQA-shaped:
    q_proj/o_proj operate on 2 heads of 8, k_proj/v_proj on a single kv head.
    """
    decoder = torch.nn.ModuleDict(
        {
            "input_layernorm": torch.nn.LayerNorm(HIDDEN),
            "self_attn": torch.nn.ModuleDict(
                {
                    "q_proj": Linear(HIDDEN, HIDDEN),
                    "k_proj": Linear(HIDDEN, HEAD_DIM),
                    "v_proj": Linear(HIDDEN, HEAD_DIM),
                    "o_proj": Linear(HIDDEN, HIDDEN),
                }
            ),
            "post_attention_layernorm": torch.nn.LayerNorm(HIDDEN),
            "mlp": torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(HIDDEN, INTERMEDIATE),
                    "up_proj": Linear(HIDDEN, INTERMEDIATE),
                    "down_proj": Linear(INTERMEDIATE, HIDDEN),
                }
            ),
        }
    )
    model = torch.nn.ModuleDict({"decoder": decoder})
    if with_config:
        # v_proj -> o_proj pairs resolve head_dim from the model config,
        # as in the msmodelslim blueprint
        model.config = SimpleNamespace(
            hidden_size=HIDDEN, num_attention_heads=NUM_HEADS
        )
    return model


def _apply_quantization_config(model: torch.nn.Module) -> None:
    apply_quantization_config(
        model,
        config=QuantizationModifier(
            scheme="W4A16_ASYM"
        ).resolve_quantization_config(),
    )


def _toy_decoder_forward(decoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Minimal GQA decoder layer forward: norm, per-head attention over the
    (possibly transformed) v_proj output, o_proj, then a norm + gated MLP.
    The residual stream itself is never transformed by FlatQuant.
    """
    batch, seq, _ = x.shape
    hidden = decoder["input_layernorm"](x)
    q = decoder["self_attn"]["q_proj"](hidden).view(batch, seq, NUM_HEADS, HEAD_DIM)
    k = decoder["self_attn"]["k_proj"](hidden).view(batch, seq, 1, HEAD_DIM)
    v = decoder["self_attn"]["v_proj"](hidden).view(batch, seq, 1, HEAD_DIM)
    k = k.expand(batch, seq, NUM_HEADS, HEAD_DIM)
    v = v.expand(batch, seq, NUM_HEADS, HEAD_DIM)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (HEAD_DIM**0.5)
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, HIDDEN)
    x = x + decoder["self_attn"]["o_proj"](attn_out)

    hidden = decoder["post_attention_layernorm"](x)
    gate = torch.nn.functional.gelu(decoder["mlp"]["gate_proj"](hidden))
    up = decoder["mlp"]["up_proj"](hidden)
    return x + decoder["mlp"]["down_proj"](gate * up)


def _setup_modifier(model: torch.nn.Module, **kwargs) -> FlatQuantModifier:
    modifier = FlatQuantModifier(mappings=_toy_mappings(), **kwargs)
    modifier._resolve_mappings(model)
    modifier._build_training_units()
    modifier._patch_units()
    return modifier


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_flatquant_is_registered():
    """Ensure FlatQuantModifier is registered in ModifierFactory"""
    modifier = ModifierFactory.create(
        type_="FlatQuantModifier",
        allow_experimental=False,
        allow_registered=True,
    )

    assert isinstance(modifier, FlatQuantModifier), "FlatQuantModifier not registered"


@pytest.mark.unit
def test_validate():
    FlatQuantModifier(amp_dtype="bfloat16", tran_type="svd")
    with pytest.raises(ValidationError):
        FlatQuantModifier(amp_dtype="float64")
    with pytest.raises(ValidationError):
        FlatQuantModifier(tran_type="foo")
    with pytest.raises(ValueError):
        FlatQuantMapping("re:.*norm$", ["re:.*proj$"], pair_kind="unsupported")


@pytest.mark.unit
@pytest.mark.parametrize(
    "size, expected",
    [
        (1, (1, 1)),
        (4, (2, 2)),
        (8, (2, 4)),
        (16, (4, 4)),
        (12, (2, 6)),
        (6, None),  # == 2 mod 4 admits no factorization
        (10, None),
    ],
)
def test_get_decompose_dim(size, expected):
    dims = get_decompose_dim(size)
    if expected is None:
        assert dims is None
    else:
        left, right = dims
        assert (left, right) == expected
        assert left * right == size


@pytest.mark.unit
def test_default_mappings_fallback():
    """Unrecognized architectures fall back to the generic default mappings"""
    model = _build_toy_model()
    assert get_layer_mappings_from_model(model) == default_mappings


@pytest.mark.unit
def test_from_awq_mapping_excludes_unsupported_pairs():
    """Fused projections and MLA layernorms have no FlatQuant analog and router
    targets (kept in high precision) are dropped from otherwise valid pairs"""
    # fused qkv / gate_up sources cannot absorb the transform
    assert _from_awq_mapping(AWQMapping("re:.*qkv_proj$", ["re:.*o_proj$"])) is None
    assert (
        _from_awq_mapping(AWQMapping("re:.*gate_up_proj$", ["re:.*down_proj$"])) is None
    )
    # activation sources (non-Linear) cannot host the learnable diagonal
    assert _from_awq_mapping(AWQMapping("re:.*gelu_impl$", ["re:.*down_proj$"])) is None
    # MLA layernorms only normalize a slice of the target input
    assert (
        _from_awq_mapping(
            AWQMapping("re:.*q_a_layernorm$", ["re:.*q_b_proj$"])
        )
        is None
    )
    # router targets are dropped, but the remaining targets still resolve
    converted = _from_awq_mapping(
        AWQMapping(
            "re:.*post_attention_layernorm$",
            ["re:.*mlp.router$", "re:.*up_proj$"],
        )
    )
    assert converted is not None
    assert converted.target_layers == ["re:.*up_proj$"]
    assert converted.pair_kind == NORM_LINEAR

    # supported conversions
    assert (
        _from_awq_mapping(
            AWQMapping("re:.*input_layernorm$", ["re:.*q_proj$", "re:.*v_proj$"])
        ).pair_kind
        == NORM_LINEAR
    )
    assert (
        _from_awq_mapping(AWQMapping("re:.*v_proj$", ["re:.*o_proj$"])).pair_kind
        == ATTN_LINEAR_LINEAR
    )
    assert (
        _from_awq_mapping(AWQMapping("re:.*up_proj$", ["re:.*down_proj$"])).pair_kind
        == MLP_LINEAR_LINEAR
    )


@pytest.mark.unit
def test_resolve_mappings():
    torch.manual_seed(0)
    model = _build_toy_model()
    _apply_quantization_config(model)
    modifier = FlatQuantModifier(mappings=_toy_mappings())
    modifier._resolve_mappings(model)

    assert len(modifier._resolved_mappings) == 4
    by_source = {m.source_name: m for m in modifier._resolved_mappings}

    norm_pair = by_source["decoder.input_layernorm"]
    assert norm_pair.pair_kind == NORM_LINEAR
    assert set(norm_pair.target_names) == {
        "decoder.self_attn.q_proj",
        "decoder.self_attn.k_proj",
        "decoder.self_attn.v_proj",
    }
    # unlike AWQ, the source is included in the LCA so that the replay of the
    # training unit covers the norm's online transform as well
    assert norm_pair.parent_name == "decoder"
    assert (
        norm_pair.trans.left_trans.size * norm_pair.trans.right_trans.size == HIDDEN
    )
    assert norm_pair.trans.diag_trans is not None

    assert "decoder.post_attention_layernorm" in by_source

    attn_pair = by_source["decoder.self_attn.v_proj"]
    assert attn_pair.pair_kind == ATTN_LINEAR_LINEAR
    assert attn_pair.target_names == ["decoder.self_attn.o_proj"]
    # GQA: the left factor acts across the o_proj heads (2), not across the
    # single v_proj kv head
    assert attn_pair.trans.left_trans.size == NUM_HEADS
    assert attn_pair.trans.right_trans.size == HEAD_DIM
    assert attn_pair.trans.diag_trans is None

    mlp_pair = by_source["decoder.mlp.up_proj"]
    assert mlp_pair.pair_kind == MLP_LINEAR_LINEAR
    assert mlp_pair.target_names == ["decoder.mlp.down_proj"]
    assert (
        mlp_pair.trans.left_trans.size * mlp_pair.trans.right_trans.size == INTERMEDIATE
    )
    assert mlp_pair.trans.diag_trans is not None


@pytest.mark.unit
def test_v_to_o_pair_requires_head_dim_from_config():
    """Without a model config, v_proj -> o_proj pairs cannot resolve head_dim
    and are skipped, while the norm and mlp pairs still resolve"""
    torch.manual_seed(0)
    model = _build_toy_model(with_config=False)
    _apply_quantization_config(model)
    modifier = FlatQuantModifier(mappings=_toy_mappings())
    modifier._resolve_mappings(model)

    assert [m.source_name for m in modifier._resolved_mappings] == [
        "decoder.input_layernorm",
        "decoder.post_attention_layernorm",
        "decoder.mlp.up_proj",
    ]


@pytest.mark.unit
def test_no_mappings_resolved_raises():
    """Sizes == 2 mod 4 admit no Kronecker decomposition, so all pairs are
    skipped and resolution raises"""
    torch.manual_seed(0)
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "input_layernorm": torch.nn.LayerNorm(6),
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": Linear(6, 6),
                            "k_proj": Linear(6, 6),
                            "v_proj": Linear(6, 6),
                            "o_proj": Linear(6, 6),
                        }
                    ),
                    "post_attention_layernorm": torch.nn.LayerNorm(6),
                    "mlp": torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(6, 6),
                            "up_proj": Linear(6, 6),
                            "down_proj": Linear(6, 6),
                        }
                    ),
                }
            )
        }
    )
    _apply_quantization_config(model)
    modifier = FlatQuantModifier(mappings=_toy_mappings())
    with pytest.raises(ValueError, match="No FlatQuant mappings"):
        modifier._resolve_mappings(model)


@pytest.mark.unit
def test_training_units_merge_nested_ancestors():
    """The self_attn (v->o) and mlp (up->down) groups merge into the decoder
    layer group which holds the norm pairs, yielding one unit per layer as in
    the msmodelslim blueprint"""
    torch.manual_seed(0)
    model = _build_toy_model()
    _apply_quantization_config(model)
    modifier = FlatQuantModifier(mappings=_toy_mappings())
    modifier._resolve_mappings(model)
    modifier._build_training_units()

    assert len(modifier._training_units) == 1
    unit = modifier._training_units[0]
    assert unit.parent is model["decoder"]
    assert len(unit.mappings) == 4
    # 2 norms and 7 linears are patched inside the unit
    assert len(unit.modules) == 9


@pytest.mark.unit
@pytest.mark.parametrize("tran_type", ["svd", "inv"])
def test_calib_and_eval_forwards_match_original(tran_type):
    """
    The central invariant of FlatQuant: with fake quantization disabled, both
    the online (calib) transform and the absorbed (eval) weights must
    reproduce the original outputs exactly, for every structure pair at once
    (norm pairs including the learnable diagonal and bias absorption, the GQA
    v->o pair, and the up->down pair).
    """
    torch.manual_seed(0)
    model = _build_toy_model()
    _apply_quantization_config(model)

    # disable learnable clipping so the calib forward is mathematically exact;
    # _absorb_unit disables the fake quantizers regardless
    modifier = _setup_modifier(model, lwc=False, lac=False, tran_type=tran_type)

    assert len(modifier._training_units) == 1
    unit = modifier._training_units[0]
    decoder = model["decoder"]
    patched = get_patched_modules(decoder)
    assert len(patched) == 9

    # move the diagonals away from their ones initialization so the diagonal
    # absorption paths are actually exercised
    for mapping in unit.mappings:
        if mapping.trans.diag_trans is not None:
            size = mapping.trans.diag_trans.diag_scale.numel()
            mapping.trans.diag_trans.diag_scale.data = torch.rand(size) + 0.5

    x = torch.randn(2, 5, HIDDEN)

    # 1. original semantics
    set_flatquant_mode(patched, MODE_ORG)
    y_orig = _toy_decoder_forward(decoder, x)

    # 2. online transforms, fake quantization disabled
    for module in patched:
        state = getattr(module, FLATQUANT_STATE_ATTR)
        if state.act_quantizer is not None:
            state.act_quantizer.enable = False
        if state.weight_quantizer is not None:
            state.weight_quantizer.enable = False
    set_flatquant_mode(patched, MODE_CALIB)
    y_calib = _toy_decoder_forward(decoder, x)
    assert_close(y_calib, y_orig, rtol=1e-4, atol=1e-5)

    # 3. absorb the transforms into the weights (fp64) and freeze the online
    #    factors into buffers; the eval forward must still reproduce the
    #    original outputs
    with torch.no_grad():
        modifier._absorb_unit(unit)
    set_flatquant_mode(patched, MODE_EVAL)
    y_eval = _toy_decoder_forward(decoder, x)
    assert_close(y_eval, y_orig, rtol=1e-4, atol=1e-5)

    # the frozen transforms persist as buffers in the state dict
    assert any("flatquant_trans" in name for name in decoder.state_dict())


@pytest.mark.unit
def test_unpatch_restores_original_forward():
    """Units which are never trained fall back to their original forward,
    leaving no trace in the module tree or the state dict"""
    torch.manual_seed(0)
    model = _build_toy_model()
    _apply_quantization_config(model)
    modifier = _setup_modifier(model)
    unit = modifier._training_units[0]
    decoder = model["decoder"]

    x = torch.randn(2, 5, HIDDEN)
    set_flatquant_mode(get_patched_modules(decoder), MODE_ORG)
    y_before = _toy_decoder_forward(decoder, x)

    modifier._unpatch_unit(unit)

    y_after = _toy_decoder_forward(decoder, x)
    assert_close(y_after, y_before)
    assert get_patched_modules(decoder) == []
    assert not any("flatquant" in name for name, _ in decoder.named_modules())
    assert not any("flatquant" in name for name in decoder.state_dict())
