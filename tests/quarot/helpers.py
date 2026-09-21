# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-only CPU plan executor. Does not implement the Modifier lifecycle."""

from collections.abc import Iterable
from pathlib import Path

import torch
from quarot_under_test.rotation import make_hadamard_rotation, rotate_axis

from .source_loader import source_definitions


def llmc_fuse(precision):
    import compressed_tensors as ct

    # Execute the existing LLMC function with real CT device/offload helpers,
    # avoiding LLMC's eager entrypoint imports. Nothing in its body is changed.
    module, _ = source_definitions(
        Path(__file__).resolve().parents[2] / "src/llmcompressor/modeling/fuse.py",
        ["fuse_norm_linears"],
        {
            "torch": torch,
            "Iterable": Iterable,
            "PRECISION": precision,
            "align_module_device": ct.align_module_device,
            "get_execution_device": ct.get_execution_device,
            "update_offload_parameter": ct.update_offload_parameter,
        },
    )
    return module.fuse_norm_linears


def matrices_for(plan, block_size=32, dtype=torch.float32):
    return {
        space.name: make_hadamard_rotation(
            space.size, block_size=block_size, shifted=space.shifted, dtype=dtype
        )
        for space in plan.spaces
    }


@torch.no_grad()
def fuse_plan(model, plan, precision):
    fuse = llmc_fuse(precision)
    for item in plan.fusions:
        fuse(
            model.get_submodule(item.norm),
            [model.get_submodule(name) for name in item.consumers],
        )


@torch.no_grad()
def execute(model, operations, matrices, precision):
    # Test-only on-device tensors; production will use offload parameter updates.
    for op in operations:
        module = model.get_submodule(op.target)
        kwargs = dict(
            axis=op.axis, stride=op.stride, offset=op.offset, precision=precision
        )
        module.weight.copy_(rotate_axis(module.weight, matrices[op.space], **kwargs))
        if op.axis == 0 and getattr(module, "bias", None) is not None:
            module.bias.copy_(rotate_axis(module.bias, matrices[op.space], **kwargs))
