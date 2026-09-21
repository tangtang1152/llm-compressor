# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-light L0–L2 harness; deliberately not a Modifier lifecycle test.

Import the pure-torch package in a private namespace, avoiding LLMC's eager
entrypoint imports. No installed model/runtime package or checkpoint is needed.
"""

import importlib.util
import sys
from pathlib import Path

import torch

SOURCE = (
    Path(__file__).resolve().parents[2] / "src/llmcompressor/modifiers/transform/quarot"
)
spec = importlib.util.spec_from_file_location(
    "quarot_under_test",
    SOURCE / "__init__.py",
    submodule_search_locations=[str(SOURCE)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)


def pytest_addoption(parser):
    parser.addoption(
        "--require-quarot-reference",
        action="store_true",
        help="Fail instead of skipping L2 if the ModelSlim source checkout is missing",
    )


def pytest_sessionstart(session):
    # Tiny BLAS work is faster and more reproducible without a large thread pool.
    session._quarot_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def pytest_sessionfinish(session, exitstatus):
    torch.set_num_threads(session._quarot_threads)
