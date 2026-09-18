# Hecate -> existing SEAL CPU: first real encrypted golden

## Follow-up: MLP and primitive semantics (2026-09-05)

Current default runner executes eight cases (`--suite all`); `--suite base`
retains the original three, and `--suite extended` selects the five additions.
Latest evidence is `/home/lhy/poseidon-work/results/seal-cpu-golden-l7iu78ak`.
All eight pass (32 encrypt/evaluate/decrypt input runs, 112 output values).
The independent two-layer polynomial MLP's max absolute error is
5.468147179499283e-8. Rotation probes, square and quartic also pass unchanged
tolerances. See [complete semantic scope](hecate-dsl-semantics-v0.md).

The key-helper build now also builds a read-only `seal_golden_metadata` shared
library against the same pinned SEAL ABI. It reads modulus count, scale and
ciphertext polynomial count from stock runtime objects, and cannot decrypt or
evaluate. Worker reports record these input/output observations and require
agreement with emitted HEVM metadata. The cryptographic runtime is unchanged.

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/seal_cpu_golden.py --suite all
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- timeout -k 3s 5m env PYTHONDONTWRITEBYTECODE=1 POSEIDON_SEAL_GOLDEN_RESULTS=/home/lhy/poseidon-work/results/seal-cpu-golden-l7iu78ak POSEIDON_SEAL_SEMANTICS_RESULTS=/home/lhy/poseidon-work/results/seal-cpu-golden-l7iu78ak python3 -m unittest discover -s scripts/baseline -p test_seal_cpu_golden.py -v
```

The remainder documents the first three-case result, retained as historical
evidence. Its missing MLP/rotation/rescale coverage is addressed above, not by
changing the old result files. Additional model families/Agent generation and
Poseidon GPU integration are still unverified.

## Outcome and scope (2026-09-05)

The user approved using Dacapo's existing SEAL CPU runtime for first-stage
DSL/Agent correctness. GPU/RNS/CUDA integration is deferred, not repaired
as a prerequisite to this milestone. No dependency installation, system
configuration change, download or Dacapo submodule edit was needed here.

Confirmed chain:

`manual Hecate Python -> Earth -> CKKS -> HEVM/CST -> stock SEAL_HEVM
-> encrypt/evaluate/decrypt/decode -> independent PyTorch float64 comparison`

Three manual programs pass. These are **not Agent-generated programs**.
Finite differential tests are evidence, not a proof for every input.
This backend change does not establish Poseidon adapter/GPU correctness or
permit SEAL performance numbers to stand in for Poseidon GPU performance.

## Recorded real run

Result directory: `/home/lhy/poseidon-work/results/seal-cpu-golden-7999_hvg`.
Main branch remains `feat/agent-dsl-correctness`, base
`4995e7cadedf2bfb9104658b5638662ecf6a1d0a`. Dacapo remains clean at
`4616402710f39df3e5f5bd7930a6c036025aaac3`.

| Manual program | Inputs / output values | MAE | Maximum absolute error | Max relative error, nonzero reference |
|---|---:|---:|---:|---:|
| x+x | 4 / 16 | 6.5730031e-9 | 1.9138589e-8 | 3.4103073e-8 |
| x*public weight | 4 / 16 | 3.8213495e-9 | 1.4125930e-8 | 7.0629649e-9 |
| Linear(4,2) | 4 / 8 | 5.7722434e-9 | 1.8393016e-8 | 1.2197547e-7 |

Every output satisfies `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
No tolerances, inputs, weights or plaintext reference were relaxed.
Cosine similarities across each case's concatenated output are approximately
1; these do not replace elementwise acceptance. Zero-reference entries have
null relative error and are still checked against the absolute tolerance.
Keys/encryption are randomized, so another genuine run need not reproduce
the exact errors, only the test configuration and acceptance rule.

Inputs are zero, fixed signed values, fixed seed-42 values, and range
boundaries. The signed Linear input `[0.5,-1,0.25,-0.75]` has independent
PyTorch reference `[0.5,0.75]`; observed decrypted output was
`[0.49999999421807656,0.7500000028799988]`.

Each case directory preserves the independent reference in `arrays.npz`
(`allow_pickle=False`), Torch FX graph, all three IR stages, HEVM/CST,
compiler command/log, `decrypted.npy`, execution diagnostics, and comparison
data in the parent `report.json`. Source, library, profile and input artifact
hashes are recorded. Nothing is downloaded during reproduction.

## Layer and parameter contract

The handwritten frontend is `scripts/baseline/golden_cases/trace_golden.py`.
Its source of public weights is the fixed array fixture. It consumes one
encrypted four-element vector. The existing compiler emits HEVM/CST; the
new driver calls the C ABI of the already-built `libSEAL_HEVM.so`. It does
not implement ciphertext arithmetic or an HEVM interpreter.

- Installed LLVM/MLIR 18.1.2, SEAL 4.0.0, Python 3.10.14, Torch 2.0.1+cpu,
  NumPy 1.25.2 are reused inside the pinned, offline Nix environment.
- Compiler profile is unchanged `third_party/dacapo/profiled_SEAL_CPU.json`,
  not the HEAAN-oriented `config.json` from previous adapter diagnostics.
- EVA compilation explicitly sets `--waterline=40` before these experiments.
  The CLI default is 20; this is an explicit precision budget selection, not
  an accuracy-threshold change or a claimed optimized setting.
- Runtime parameters match upstream `SEAL_HEVM::create_context`: CKKS,
  degree 32768, 16384 slots, fourteen 60-bit primes. The SEAL context passes
  explicit tc128 validation. This means SEAL's parameter security check
  passed, not a new cryptographic proof or a production deployment audit.
- The SEAL key context contains 14 primes; first data context has 13.
  HEVM level here is the data-prime count: level 13 maps to SEAL chain index
  12. The source uses `L-1-level` modulus drops. Do not import a Poseidon
  level convention into this runtime.
- All three artifacts input at level 13 / log2(scale)=40. Add outputs level
  1 / scale 40; multiply and Linear output level 2 / scale 80. These are
  observed compiler outputs, even though the profile's lower-bound field is 2.
- No rescale, ciphertext multiply or bootstrap is needed for these three
  graphs. Their correctness is not evidence for those opcodes; square-MLP
  and separate semantic tests must cover them next.

Constants and the four input elements repeat across all 16384 slots via
the stock runtime's `src[i % src.size()]` rule. Each Linear row computes
`p=x*w[row]; pairs=p+rotate(p,1); total=pairs+rotate(pairs,2); total+bias`.
Slot zero of each of the two result ciphertexts is selected. This exercises
real cross-element reduction. A full-cycle sum alone does **not** distinguish
left/right rotation direction; a separate asymmetric rotation oracle is
still required for a general DSL semantic specification.

## Safety boundary

`seal_artifact_gate.py` bounds HEVM/CST lengths/counts and checks native
operand indices, initialized registers, levels, required rotation keys,
result bindings and constant finiteness. It rejects opcodes 5 (upscale),
10 (bootstrap) and all unknown opcodes before loading the native library.
Alloc 65535 is an upstream no-op with unspecified other fields. Reuse of a
plaintext destination is conservatively rejected because stock preprocess
encodes all constants eagerly. This is a deliberately small golden subset,
not an assertion that the whole compiler supports this restricted gate.

This gate matters: stock release SEAL_HEVM can decrypt/re-encrypt in its
bootstrap branch after assertions are compiled out. No such opcode is
allowed or executed. Validated HEVM/CST bytes are supplied to the native
loader via sealed Linux memfd snapshots, avoiding a check/load file mismatch.

`seal_keys/main.cpp` is only parameter validation and key provisioning using
SEAL. It preserves the full modulus chain and generates public/secret/relin
keys plus actual Galois keys for rotations 1 and 2, instead of the stock
all-rotations key set. The VM itself is unmodified. The helper refuses to
overwrite existing keys. A fresh private directory and umask 077 keep key
material under native WSL results, never in the repository. **Do not publish
the private-keys directory.** The three observed worker peaks were below
931 MiB RSS each; no GPU memory was used.

Workers have hard wall/CPU/address-space limits, disabled core dumps and
process isolation because upstream exposes no VM destroy API. These bounds,
the gate and Nix **are not a sandbox for hostile generated Python**. AST,
capability isolation and immutable reference/parameter ownership remain
requirements before Agent execution.

## Reproduce and test

From PowerShell, with the already-approved installed environment:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/seal_cpu_golden.py
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- timeout -k 3s 5m env PYTHONDONTWRITEBYTECODE=1 POSEIDON_SEAL_GOLDEN_RESULTS=/home/lhy/poseidon-work/results/seal-cpu-golden-7999_hvg python3 -m unittest discover -s scripts/baseline -p test_seal_cpu_golden.py -v
```

The runner creates a fresh result directory, compiles only a tiny SEAL key
helper (`-j2`) out of source, and never rewrites old results/keys. Fresh runs
have a 900-second outer limit, 60-second trace/compile limits, 120-second
key setup limit and 150-second worker limit. Native worker CPU limit is 120
seconds and virtual address space 4 GiB.

All 13 new tests passed when pointed at the real run: ten parser/negative
gate tests plus three encrypted-evidence/parameter/metric tests. The full
baseline suite passed 58/58 with the previous native/Python compiler result
directories selected as well. Existing Poseidon C++ files were not changed
in this step; its earlier 53-pass/3-skip results are historical, not rerun here.

## Next gate

Construct and execute `Linear(4,4) -> square -> Linear(4,2)` with independent
fixed PyTorch reference and the same tolerance. Add isolated rotation,
rescale and ciphertext-multiply semantics tests, then freeze the initial
frontend operator/shape/layout contract. Only after these golden/spec gates
should the structured Agent generator and same-input deterministic translator
be connected. External LLM service/model/credentials/cost still need approval.
Expand to the planned 48 small-model cases with held-out graph structures;
do not claim that these initial three programs already meet that coverage.
