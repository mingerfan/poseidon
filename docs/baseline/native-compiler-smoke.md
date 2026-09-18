# Native compiler diagnostic: real artifacts, adapter blocker

Date: 2026-09-05. Parent commit `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`,
Dacapo commit `4616402710f39df3e5f5bd7930a6c036025aaac3`; submodule unmodified.

## Confirmed results

LLVM/MLIR/Clang 18.1.2 and SEAL 4.0.0 installed successfully. Dacapo's
`hecate-opt`, `HecateFrontend` and `SEAL_HEVM` built successfully in
`/home/lhy/poseidon-work/build-dacapo/hecate-18.1.2-nix`.
Dialect registration and actual CDLL loading/exported-symbol checks passed.
The CMake LLVM consumer probe linked and ran with `LLVM=18.1.2 glibc=2.39`.

Before Python wheel installation, `native_compiler_smoke.py` calls the same
`libHecateFrontend.so` C API used by `hecate/expr.py`: create context/function,
add encrypted input, construct operations, set return, save Earth and CST.
It then invokes the existing `hecate-opt --eva` pipeline with `--verify-each`,
debug Earth/CKKS snapshots and the **unchanged** upstream `config.json`.
This is C-API/compiler isolation testing, **not** Hecate Python/Torch tracing,
not a new CPU execution backend and not an Agent-generated program.

| Case | Graph | CST bytes | HEVM bytes | Real HEVM operations |
|---|---|---:|---:|---|
| add | `x + x` | 8 | 120 | ModswitchC, AddCC |
| mul_plain | `x * [1.5, -2, 0.25, 3]` | 48 | 136 | Alloc, ModswitchC, Encode, MulCP |

Both compiler commands exited 0. Input Earth, optimized Earth, CKKS, final IR,
CST, HEVM and SHA-256 manifests are saved under
`/home/lhy/poseidon-work/results/native-compiler-k73nx6wz`.
The CST payloads contain zero vectors for add and one exact fixed-weight vector
for mul_plain. No ciphertext was created or decrypted; no bootstrap was called.

Latest validation: **35/35 tests passed**, including checks of actual emitted
artifact hashes, exact CST weights and the current adapter's fail-closed report.
The unsupported-opcode test characterizes today's blocker, not a permanent
desired behavior. Log: `validation-tests.log` in the result directory.
The refreshed `environment-gate.json` reports successful C++ tool/SEAL/Python
interpreter checks, with NumPy and Torch missing; overall exit 1 is expected.

The profile has polynomialDegree=131072, bootstrapLevelUpperBound=16,
levelUpperBound=29 and rescalingFactor=51. Compiler-default waterline=20 and
output-val=10 were retained. These are recorded compiler inputs, **not** an
approved runnable GPU security/precision configuration or performance result.

## Confirmed failure: adapter opcode contract

Poseidon's existing CPU-only dump tool loaded the actual HEVM/CST artifacts,
reported their opcode distributions and rejected both with exit 1:

`dacapo_adapter ... @104: unsupported HEVM opcode 4`

The structured report names `ModswitchC`, count 1, supported=false. Schedule
construction, GPU preflight and communication checks did **not** run after
translation failed. Thus this is an **adapter compatibility blocker**, not a
GPU numerical error, network error, scheduler failure or successful preflight.

Source explanation:

- `lib/Dialect/Earth/Transforms/WaterlineRescaling.cpp` invokes
  `refineReturnValues` after scale propagation.
- `lib/Dialect/Earth/Transforms/Common.cpp:44` deliberately reduces return
  levels and records `init_level` using the profile's upper bound (16 here).
- `EarlyModswitch.cpp` moves these level drops earlier where legal.
- Actual CKKS snapshots show an input at level 16 and `downFactor=15`, leaving
  level 1 before AddCC/MulCP. No level/scale/opcode was guessed or rewritten.

Evidence-based inference: existing `GpuEvaluator::drop_modulus` may be a backend
primitive for a future verified ModswitchC mapping. This is **unconfirmed as a
compatible mapping**: HEVM level indexing, target parms_id, Q/P ordering, NTT
form and scale preservation still require tests. Do not map it to rescale,
remove the opcode, lower security parameters or disable readiness checks.

The separately identified short-CST repetition and MulCC/relinearization
differences remain open. They were not repaired or validated by this test.

## Reproduction

From the approved WSL source checkout (no downloads; existing Nix deps required):

```bash
timeout -k 10s 6m env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/native_compiler_smoke.py
```

The script prints a newly created result directory. For each case, invoke the
existing adapter; the following command intentionally returns 1 at the current
unsupported-opcode gate:

```bash
timeout -k 3s 60s /home/lhy/poseidon-work/build-poseidon/agent-dsl-adapter/bin/poseidon_mgpu_dacapo_hevm_dump \
  --hevm /home/lhy/poseidon-work/results/native-compiler-k73nx6wz/add/lowered._hecate_add.hevm \
  --constants /home/lhy/poseidon-work/results/native-compiler-k73nx6wz/add/_hecate_add.cst \
  --devices 1 --opcode-summary --communication-plan \
  --communication-execution-preflight --poseidon-gpu-preflight --require-ready \
  --write-summary-json /home/lhy/poseidon-work/results/native-compiler-k73nx6wz/add/poseidon-preflight.json \
  --no-schedule
```

Repeat for `mul_plain`, replacing directory/function names accordingly. No
GPU keys or communication availability are asserted. Validate captured artifacts:

```bash
timeout -k 3s 5m env PYTHONDONTWRITEBYTECODE=1 \
  POSEIDON_NATIVE_COMPILER_RESULTS=/home/lhy/poseidon-work/results/native-compiler-k73nx6wz \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -v
```

## Historical next approval gate: Python frontend/reference dependencies

This approval was subsequently granted and completed. See
[real Python trace/compile results](python-compiler-smoke.md). The following
records the earlier request, not the current installed state.

Use the already pinned Python 3.10.14 in an isolated venv at
`/home/lhy/poseidon-work/venvs/hecate-2.0.1-cpu`; wheel cache belongs under
`/home/lhy/poseidon-work/cache`, never system site-packages. Source remains here.
No pip/venv/wheel installation was performed in this step.

Version evidence is Dacapo's `requirements.txt`: NumPy 1.25.2, Torch 2.0.1,
filelock 3.12.2, Jinja2 3.1.2, MarkupSafe 2.1.3, mpmath 1.3.0, networkx 3.1,
sympy 1.12 and typing_extensions 4.7.1. Select the corresponding **CPU** Torch
wheel; do not install the repository's CUDA/NVIDIA/triton benchmark bundle.
Wheel hashes/transitive requirements must be locked and checked before install.

Read-only metadata verified on 2026-09-05:

- [Official Torch CPU index](https://download.pytorch.org/whl/cpu/torch/):
  `torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl`, HEAD size 195,422,835 bytes;
  index SHA-256 `fec257249ba014c68629a1994b0c6e7356e20e1afc77a87b9941a40e5095285d`.
- [NumPy release metadata](https://pypi.org/pypi/numpy/1.25.2/json): cp310
  manylinux x86_64 wheel, 18,219,190 bytes; SHA-256
  `f08f2e037bba04e707eebf4bc934f1972a315c883a9e0ebfa8a7756eabf9e357`.

The two primary wheels total approximately 203.75 MiB; estimate 220-250 MiB
including transitive/bootstrap wheels. Request an upper bound of **300 MiB**
and a **2 GiB** venv/cache reservation; stop if the resolved plan exceeds it.
These historic pins reproduce the research stack, not a current production
security recommendation. Do not load untrusted Torch pickle checkpoints.

After approval: verify hashes/dependencies, install in the venv, handle the
out-of-source Hecate library path without changing the gitlink, trace the real
Hecate Python examples, and confirm whether they reproduce this same opcode
blocker. GPU/CUDA installation and encrypted execution remain separate gates.
