# Qwen 24-layer compilation on 188Server

These records cover a full 24-layer Qwen2.5-0.5B two-token prefill, final
RMSNorm, complete 151936-token LM head, and 96 K/V outputs. Weights are
reproducible synthetic fixtures, not a pretrained checkpoint.

The compiler uses the `depth-dp` bootstrap strategy adapted from Fhelipe.
`summary.json` records the original native compilation, `gpu-plans/summary.json`
records the first single-GPU and four-GPU plans with Release/reuse, and
`unit-plaintext/summary.json` records the current plans after plaintext scale
absorption and Encode CSE. Older records describe their original compiler
versions; their source and binary provenance is historical.

The current compilation results are:

| Plan | Compile time | Peak compiler RSS | Per-GPU RNS object peak |
| --- | --- | --- | --- |
| One GPU | 16m 17s | 66.44 GiB | 8.08 TiB |
| Four GPUs | 20m 53s | 91.43 GiB | 2.65, 2.52, 2.40, 2.24 TiB |

Ciphertext-only peaks are 192.87 GiB for one GPU and 8.69, 16.70, 11.37,
9.42 GiB for four GPUs. All weight plaintexts are still encoded and uploaded
in initialization. Release does not implement on-demand weight loading, so the
current plans do not fit the server's four 32 GiB V100 GPUs.

The reports assume sequential completion of each instruction and exclude keys,
parameter tables, operator workspaces, transfer staging, encoding caches,
allocator pools/alignment, and object metadata. They are not measured GPU
memory peaks or bounds for parallel execution. Complete-model CKKS accuracy
and GPU execution have not been tested.

Compiler tests passed 7/7. The real CPU CKKS API tests passed 7/7, including
plaintext scale absorption and preservation of positive/negative prefix masks.
The memory-pass regression compared 16 cases with the previous implementation
and exercised a 20000-operation chain. Detailed logs are in `unit-plaintext/`.

Large artifacts remain outside Git at:

```
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-native
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-gpu-memory
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-unit-plaintext/gpu-plans
```

Lossless source bytecode is 14.82 GB and the deduplicated plaintext bundle is
14.70 GB. Diagnostic MLIR elides constants and cannot be recompiled. These
records keep scripts, profiles, summaries, and logs; generated IR, RuntimePlan
JSON, bundles, binary payloads, and Python caches are excluded from Git.

`run_fixture.py`, `run_frontend.py`, and `run_validation.py` are the original
server supervisors, including resource limits and timing/RSS reporting. Other
stage supervisors are under `gpu-plans/` and `unit-plaintext/`. These are
experiment records with fixed paths to the server workspace and `/tmp` helpers;
review their paths and output files before rerunning them.

`server-python/hecate/` preserves the private NumPy-only frontend used on the
server. It removes the unused Torch import and Tensor conversion from the
compiler frontend; native NumPy/C API calls, bytecode export, and source-frame
lookup are unchanged. This adapter is an experiment snapshot. The compiler's
normal frontend still supports Torch. Its native frontend library is supplied
by the server toolchain and is not committed here.
