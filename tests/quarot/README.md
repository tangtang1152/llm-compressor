# QuaRot local correctness harness

Run from the repository root:

```sh
python -m pytest tests/quarot -q
```

Only PyTorch and pytest are needed for L0/L1. These tests load the pure-torch
implementation under a private namespace to avoid LLMC's eager optional runtime
imports. This is not a substitute for the later normal-import Modifier/recipe L3 gate.
No model downloads, `from_pretrained`, NPU, GPU, or datasets are used.

For L2 set `MODELSLIM_SOURCE` to a local, unchanged ModelSlim source checkout.
The test adapter executes selected original function bodies directly from that
checkout and records source SHA256 hashes. It does not vendor reference algorithms.
Missing reference source is an explicit skip; the release gate must run with it present.
