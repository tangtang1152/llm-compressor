# QuaRot local correctness harness

Run from the repository root:

```sh
python -m pytest tests/quarot/test_rotation.py -q -p no:cacheprovider
```

Only PyTorch and pytest are needed for tensor and topology-only tests. Tests load the pure-torch
implementation under a private namespace to avoid LLMC's eager optional runtime
imports. This is not a substitute for the later normal-import Modifier/recipe L3 gate.
No model downloads, `from_pretrained`, NPU, GPU, or datasets are used.

Tiny norm-fusion/forward tests also require compressed-tensors; they execute the
existing LLMC fuse function with real CT offload helpers in a private namespace.
The tests do not mock those helpers, but CPU-resident use is not an offload test.

For L2 set `MODELSLIM_SOURCE` to a local, unchanged ModelSlim source checkout
(or use the workspace's sibling `reference/msmodelslim`). NumPy and packaging are
needed for the original reference seed routine.
The test adapter executes selected original function bodies directly from that
checkout and records source SHA256 hashes. It does not vendor reference algorithms.
Missing reference source is an explicit skip for ordinary pytest runs. The gate
below fails for missing source, missing dependencies, or any skipped tests:

```sh
python tools/run_quarot_checks.py --modelslim-source /path/to/msmodelslim --report-dir /path/to/reports/run-001
```

The gate writes JUnit, environment/commit metadata and per-parameter error reports
for initial, norm fusion, embedding and all four rotation stages. Cases cover
block_size=32 and reference default (-1), q_lora_rank=32/64, FP32/BF16 storage,
and three matrix seeds. FP64 forward algebra uses a separately constructed FP64 Q.

Oracle boundary: selected original functions and GLM52 adapter methods are executed,
not the entire ModelSlim processor/import graph. NPU discovery and app initialization
are not run. CPU seed behavior is selected; logging uses Python logging; exception
numeric codes are irrelevant to numerical comparisons. Non-power-of-two asset lookup
is unsupported. Only named MTP-only targets are removed from the decoder reference
mapping. Every other target must resolve. No reference implementation is vendored.

This harness is an early correctness gate, not upstream-ready GLM-5.2 support.
MTP, full non-power-of-two bases, fused/distributed experts, MLA decode caches,
Modifier lifecycle, recipe serialization and mixed MXFP composition remain separate
tests to build. No accuracy or production deployment claims follow from this gate.
