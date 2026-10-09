"""
Forward-patching machinery for FlatQuant.

Rather than replacing modules (which would break the quantization config
applied by QuantizationMixin modifiers), FlatQuant patches the ``forward``
attribute of the target modules in place. Because these modules are leaf
modules of the sequential subgraph tracer, the patched forwards are
executed directly at runtime and their Python-level mode switches
(org / calib / eval) remain fully effective.

Patched modules carry a ``FlatQuantState`` object (a plain Python object,
registered under a private attribute so it is invisible to the module tree)
which holds references to the shared transforms and fake quantizers, plus
the saved original forward.
"""

import types
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

from llmcompressor.modifiers.transform.flatquant.components import (
    ActivationQuantizer,
    GeneralMatrixTrans,
    WeightQuantizer,
)

__all__ = [
    "FLATQUANT_STATE_ATTR",
    "FLATQUANT_TRANS_NAME",
    "FLATQUANT_ACT_QUANTIZER_NAME",
    "FLATQUANT_WEIGHT_QUANTIZER_NAME",
    "MODE_CALIB",
    "MODE_EVAL",
    "MODE_ORG",
    "FlatQuantState",
    "get_patched_modules",
    "patch_flatquant_linear",
    "patch_flatquant_norm",
    "register_flatquant_trans",
    "set_flatquant_mode",
    "unpatch_flatquant_module",
]

# plain (non-module) attribute holding the FlatQuantState of a patched module
FLATQUANT_STATE_ATTR = "_flatquant_state"
# names under which transforms/quantizers are registered as submodules so
# that their parameters are tracked by the optimizer and saved in the
# state_dict
FLATQUANT_TRANS_NAME = "flatquant_trans"
FLATQUANT_ACT_QUANTIZER_NAME = "flatquant_act_quantizer"
FLATQUANT_WEIGHT_QUANTIZER_NAME = "flatquant_weight_quantizer"

MODE_ORG = "org"
MODE_CALIB = "calib"
MODE_EVAL = "eval"


@dataclass
class FlatQuantState:
    """
    Runtime state of a single flatquant-patched module.

    :param mode: forward mode, one of "org" (original semantics),
        "calib" (dynamic transforms + fake quantization, used during
        training) or "eval" (absorbed weights + online transform + fake
        quantization with frozen scales)
    :param orig_forward: the forward callable of the module before patching
    :param norm_trans: for norm modules, the GeneralMatrixTrans applied to
        the norm output
    :param act_in_trans: online transformation applied to the input
        activations of a linear module
    :param weight_in_trans: transformation whose inverse-transpose is
        (dynamically, in calib mode) applied to the weight
    :param weight_out_trans: transformation applied to the transposed weight
        output (v_proj-style output transforms)
    :param act_quantizer: fake quantizer for input activations
    :param weight_quantizer: fake quantizer for weights
    """

    mode: str = MODE_ORG
    orig_forward: Optional[Callable] = None
    norm_trans: Optional[GeneralMatrixTrans] = None
    act_in_trans: Optional[GeneralMatrixTrans] = None
    weight_in_trans: Optional[GeneralMatrixTrans] = None
    weight_out_trans: Optional[GeneralMatrixTrans] = None
    act_quantizer: Optional[ActivationQuantizer] = None
    weight_quantizer: Optional[WeightQuantizer] = None
    extra: dict = field(default_factory=dict)


def _flat_linear_forward(module: Module, hidden_states):
    state: FlatQuantState = getattr(module, FLATQUANT_STATE_ATTR)
    if state.mode == MODE_ORG:
        return state.orig_forward(hidden_states)
    if state.mode == MODE_CALIB:
        return _calib_linear_forward(module, state, hidden_states)
    return _eval_linear_forward(module, state, hidden_states)


def _calib_linear_forward(module: Module, state: FlatQuantState, hidden_states):
    """
    Calibration-mode forward: apply the online input transform (if any),
    apply the (learnable) transforms to the weight, then fake-quantize both
    the activations and the transformed weight.
    """
    if state.act_in_trans is not None:
        hidden_states = state.act_in_trans(hidden_states)

    weight = module.weight.detach()
    if state.weight_in_trans is not None:
        weight = state.weight_in_trans(weight, inv_t=True)
    if state.weight_quantizer.lwc:
        weight = state.weight_quantizer.apply_wclip(weight)

    bias = module.bias
    if state.weight_out_trans is not None:
        weight = state.weight_out_trans(weight.T).T
        if bias is not None:
            bias = state.weight_out_trans(bias)

    hidden_states = state.act_quantizer(hidden_states)
    weight = state.weight_quantizer(weight)
    return F.linear(hidden_states, weight, bias)


def _eval_linear_forward(module: Module, state: FlatQuantState, hidden_states):
    """
    Eval-mode forward: the weight transforms have been absorbed into the
    module weights, so only the online activation transform and fake
    quantization (with frozen scales) are applied.
    """
    if state.act_in_trans is not None:
        hidden_states = state.act_in_trans(hidden_states)

    weight = module.weight
    bias = module.bias

    if state.act_quantizer is not None and state.act_quantizer.enable:
        hidden_states = state.act_quantizer(hidden_states)
    if state.weight_quantizer is not None and state.weight_quantizer.enable:
        # enable_find is disabled after training, so this quantizes with
        # the frozen scales rather than recomputing them each forward
        weight = state.weight_quantizer(weight)

    return F.linear(hidden_states, weight, bias)


def _flat_norm_forward(module: Module, hidden_states):
    state: FlatQuantState = getattr(module, FLATQUANT_STATE_ATTR)
    output = state.orig_forward(hidden_states)
    if state.mode != MODE_ORG and state.norm_trans is not None:
        output = state.norm_trans(output)
    return output


def patch_flatquant_linear(
    module: Module,
    act_in_trans: GeneralMatrixTrans | None = None,
    weight_in_trans: GeneralMatrixTrans | None = None,
    weight_out_trans: GeneralMatrixTrans | None = None,
    act_quantizer: ActivationQuantizer | None = None,
    weight_quantizer: WeightQuantizer | None = None,
) -> FlatQuantState:
    """
    Patch the forward of a linear module with the flatquant three-mode
    forward. The quantizers are registered as submodules of the module so
    their learnable clip factors are visible to the optimizer.

    :param module: the linear module to patch
    :param act_in_trans: online input-activation transform
    :param weight_in_trans: full transform absorbed into the weight
        (inverse-transpose applied in calib mode)
    :param weight_out_trans: output-space transform absorbed into the
        weight (and bias) of this module, e.g. the right factor for v_proj
    :param act_quantizer: fake quantizer for input activations
    :param weight_quantizer: fake quantizer for weights
    :return: the FlatQuantState associated with the module
    """
    if hasattr(module, FLATQUANT_STATE_ATTR):
        raise ValueError(f"Module {module} is already patched with flatquant")

    state = FlatQuantState(mode=MODE_ORG)
    state.orig_forward = module.forward
    state.act_in_trans = act_in_trans
    state.weight_in_trans = weight_in_trans
    state.weight_out_trans = weight_out_trans
    state.act_quantizer = act_quantizer
    state.weight_quantizer = weight_quantizer

    setattr(module, FLATQUANT_STATE_ATTR, state)
    if act_quantizer is not None:
        module.add_module(FLATQUANT_ACT_QUANTIZER_NAME, act_quantizer)
    if weight_quantizer is not None:
        module.add_module(FLATQUANT_WEIGHT_QUANTIZER_NAME, weight_quantizer)
    module.forward = types.MethodType(_flat_linear_forward, module)

    return state


def patch_flatquant_norm(module: Module, trans: GeneralMatrixTrans) -> FlatQuantState:
    """
    Patch the forward of a normalization module so that its output is
    multiplied by the general (Kronecker + diagonal) transform. The
    transform is registered as a submodule of the norm so that its
    parameters are visible to the optimizer and are saved in the
    state_dict (this is the canonical home of a norm->linear transform).

    :param module: the norm module to patch
    :param trans: the GeneralMatrixTrans to apply to the norm output
    :return: the FlatQuantState associated with the module
    """
    if hasattr(module, FLATQUANT_STATE_ATTR):
        raise ValueError(f"Module {module} is already patched with flatquant")

    state = FlatQuantState(mode=MODE_ORG)
    state.orig_forward = module.forward
    state.norm_trans = trans

    setattr(module, FLATQUANT_STATE_ATTR, state)
    module.add_module(FLATQUANT_TRANS_NAME, trans)
    module.forward = types.MethodType(_flat_norm_forward, module)

    return state


def register_flatquant_trans(module: Module, trans: GeneralMatrixTrans):
    """
    Register a transform as a submodule of a linear module (the canonical
    home of the transform for linear->linear structure pairs, e.g.
    v_proj->o_proj or up_proj->down_proj).
    """
    module.add_module(FLATQUANT_TRANS_NAME, trans)


def set_flatquant_mode(modules: list[Module], mode: str):
    """
    Set the forward mode of several patched modules at once.
    """
    for module in modules:
        state: FlatQuantState = getattr(module, FLATQUANT_STATE_ATTR)
        state.mode = mode


def get_patched_modules(module: Module) -> list[Module]:
    """
    Return all (recursively) patched modules below (and including) ``module``.
    """
    return [
        m
        for m in module.modules()
        if getattr(m, FLATQUANT_STATE_ATTR, None) is not None
    ]


def unpatch_flatquant_module(module: Module):
    """
    Restore the original forward of a patched module and remove the
    flatquant state, submodules and buffers. Used for mappings which are
    skipped (e.g. no calibration data reached them) so that the module
    falls back to its original semantics.
    """
    state: Optional[FlatQuantState] = getattr(module, FLATQUANT_STATE_ATTR, None)
    if state is None:
        return

    # forward was patched as an instance attribute; deleting it reveals the
    # original class-level forward
    try:
        del module.forward
    except AttributeError:
        pass

    delattr(module, FLATQUANT_STATE_ATTR)
    for name in (
        FLATQUANT_TRANS_NAME,
        FLATQUANT_ACT_QUANTIZER_NAME,
        FLATQUANT_WEIGHT_QUANTIZER_NAME,
    ):
        if hasattr(module, name):
            delattr(module, name)
