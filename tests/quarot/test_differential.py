# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from quarot_under_test.mappings import build_glm_plan
from quarot_under_test.rotation import make_hadamard_rotation

from .helpers import execute, fuse_plan, matrices_for
from .tiny_glm import TinyConfig, TinyGLM


@pytest.fixture(scope="module")
def oracle(request):
    from .modelslim_oracle import ModelSlimOracle

    root = Path(
        os.environ.get(
            "MODELSLIM_SOURCE",
            Path(__file__).resolve().parents[4] / "reference/msmodelslim",
        )
    )
    if not (root / "msmodelslim/model/glm_5/quarot.py").is_file():
        if request.config.getoption("--require-quarot-reference", default=False):
            pytest.fail(f"Missing ModelSlim reference source: {root}")
        pytest.skip(
            "L2 requires a local ModelSlim source checkout; set MODELSLIM_SOURCE"
        )
    return ModelSlimOracle(root)


@pytest.mark.parametrize(
    "size,block,shifted",
    [(64, 32, False), (64, -1, False), (64, 32, True), (32, 32, True)],
)
@pytest.mark.parametrize("seed", [0, 1234, 1701])
def test_rotation_matrices_match_reference(oracle, size, block, shifted, seed):
    from .modelslim_oracle import preserve_rng

    mode = (
        oracle.utils.QuaRotMode.BLOCK_HADAMARD_SHIFTED
        if shifted
        else oracle.utils.QuaRotMode.HADAMARD
    )
    with preserve_rng():
        reference = oracle.utils.create_rot(mode, size, block, seed=seed)
    ours = make_hadamard_rotation(
        size, block_size=None if block == -1 else block, shifted=shifted, seed=seed
    )
    torch.testing.assert_close(ours, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "indexers", [("full", "shared"), ("full", "full"), ("shared", "shared")]
)
def test_topology_matches_actual_glm52_adapter(oracle, indexers):
    model = TinyGLM(TinyConfig(indexer_types=indexers))
    plan = build_glm_plan(model)
    fusions, pre, stages, _ = oracle.plan(model, 32)
    assert {entry.norm: set(entry.consumers) for entry in plan.fusions} == {
        key: set(value) for key, value in fusions.items()
    }
    assert set(pre.right_rot) == {plan.embedding.target}
    for space, pair in stages.items():
        for axis, targets in ((0, pair.left_rot), (1, pair.right_rot)):
            assert {
                op.target
                for op in plan.rotations
                if op.space == space and op.axis == axis
            } == set(targets)


@pytest.mark.parametrize("q_rank", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [32, None])
@torch.no_grad()
def test_all_stages_match_modelslim(oracle, q_rank, dtype, block_size):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM(TinyConfig(q_lora_rank=q_rank)).to(dtype)
    reference = deepcopy(model)
    plan = build_glm_plan(model)
    matrices = matrices_for(plan, block_size=block_size)
    fusions, pre, stages, expected_matrices = oracle.plan(
        reference, -1 if block_size is None else block_size
    )
    for name in matrices:
        torch.testing.assert_close(
            matrices[name], expected_matrices[name], atol=0, rtol=0
        )
    report = {
        "fixture_seed": 42,
        "rotation_seed": 1234,
        "config": asdict(model.config),
        "dtype": str(dtype),
        "compute_dtype": "torch.float32",
        "block_size": block_size,
        "source_sha256": oracle.hashes,
        "stages": {},
    }
    # FP32 per-operation parity; BF16 can differ by one storage rounding unit when
    # a segmented matmul changes accumulation order relative to a dense zero block.
    atol, rtol = (2e-7, 2e-5) if dtype == torch.float32 else (2e-7, 0.008)

    def compare(stage):
        ours, theirs = model.state_dict(), reference.state_dict()
        assert ours.keys() == theirs.keys()
        errors = {}
        for name in ours:
            torch.testing.assert_close(
                ours[name], theirs[name], atol=atol, rtol=rtol, msg=f"{stage}: {name}"
            )
            errors[name] = (
                (ours[name].float() - theirs[name].float()).abs().max().item()
            )
        report["stages"][stage] = errors

    compare("initial")
    for norm, consumers in fusions.items():
        oracle.utils.fuse_ln_linear(
            [reference.get_submodule(norm)],
            [reference.get_submodule(name) for name in consumers],
        )
    fuse_plan(model, plan, torch.float32)
    compare("norm_fusion")
    oracle.rotate(reference, pre)
    execute(model, [plan.embedding], matrices, torch.float32)
    compare("embedding")
    for name, pair in stages.items():
        oracle.rotate(reference, pair)
        execute(
            model,
            [op for op in plan.rotations if op.space == name],
            matrices,
            torch.float32,
        )
        compare(name)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 9, 3, 7, 1]])
    actual_logits, actual_trace = model(tokens)
    reference_logits, reference_trace = reference(tokens)
    torch.testing.assert_close(
        actual_logits,
        reference_logits,
        atol=2e-5 if dtype == torch.float32 else 0.03,
        rtol=0.0001 if dtype == torch.float32 else 0.03,
    )
    for ours, theirs in zip(actual_trace, reference_trace, strict=True):
        for key in ours:
            if ours[key] is not None:
                torch.testing.assert_close(
                    ours[key],
                    theirs[key],
                    atol=3e-5 if dtype == torch.float32 else 0.08,
                    rtol=0.0001 if dtype == torch.float32 else 0.03,
                    msg=key,
                )
    report["logits_max_abs_error"] = (
        (actual_logits.float() - reference_logits.float()).abs().max().item()
    )
    if directory := os.environ.get("QUAROT_REPORT_DIR"):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (
            output
            / f"quarot-{str(dtype).removeprefix('torch.')}-q{q_rank}-b{block_size}.json"
        ).write_text(json.dumps(report, indent=2), encoding="utf-8")
