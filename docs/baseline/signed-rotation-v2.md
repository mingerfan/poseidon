# Signed rotation v2: bounded CPU correctness evidence

## Outcome

All six manually authored rotation programs (-3, -2, -1, +1, +2, +3) passed
Hecate tracing, Dacapo compilation, artifact inspection, actual Galois-key file
preflight, real SEAL HEVM CPU execution and independent numerical comparison.
An intentionally reversed-direction program reached decryption and failed the
comparison, as required. A separate actual missing-key probe failed before VM
execution. These are manual golden and deterministic baseline results, not new
LLM-generated results and not Poseidon GPU results.

Source remains on branch `feat/agent-dsl-correctness`, main HEAD
`4995e7cadedf2bfb9104658b5638662ecf6a1d0a`, Dacapo gitlink
`4616402710f39df3e5f5bd7930a6c036025aaac3`. The previously documented local
plaintext-subtraction frontend fix remains in the submodule working tree.

## Semantic contract

`hecate-function-v0` and `v1` retain their original allowed rotations (+1,+2).
`hecate-function-v2` adds signed literal steps ±1, ±2 and ±3 while retaining
the v1 arithmetic rules. A graph with rotation selects v2 automatically.

For the currently verified period-four layout:

```text
x = [a, b, c, d]
x.rotate(+1) = [b, c, d, a]
x.rotate(-1) = [d, a, b, c]
```

The independent reference uses list indexing; the trusted PyTorch model uses
`torch.roll(x, -step, 0)`; the DSL emits `x.rotate(step)`. Negative integer
literals are parsed structurally, never evaluated as arbitrary Python.

The physical ciphertext has 16384 slots containing repeated four-element
blocks. Thus +3 and -1 have the same logical output on this layout, despite
different physical rotations and key requirements. Six steps are NOT six
independent model families, six new operator kinds, or coverage of larger
layouts. Non-four-element layouts, rotating scalar-neuron outputs, unsupported
steps and expression-valued steps remain rejected.

## Key and artifact chain

1. Versioned task rules bind the allowed rotation policy.
2. `seal_keys/main.cpp` generates explicit selected Galois keys. Legacy callers
   still default to +1,+2; v2 callers request the six signed steps.
3. `seal_artifact_gate.py` decodes the signed HEVM rotation operand and checks
   it against the selected policy.
4. `seal_keys/metadata.cpp::verify_galois_file` loads actual parameter and
   Galois-key files, checks SEAL tc128 and required Galois elements, and returns
   failure for missing keys. It does not merely trust JSON metadata.
5. `seal_cpu_golden.py` requires that check before initializing the VM and
   records the result with each encrypted execution.

N=32768, fourteen 60-bit key moduli and thirteen initial data moduli are
unchanged. Input log2(scale)=40. No bootstrap, decrypt/re-encrypt replacement,
reduced security parameters, or weakened comparison threshold is used.

## Durable results

Results root: `/home/lhy/poseidon-work/results`.

- `rotation-batch-i9qaiev4/report.json`: six correct golden programs, four input
  sets each, 96 compared scalar outputs. All pass the frozen criterion
  `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
- The same report separately records one wrong-direction counterexample:
  four encrypted input sets, 16 outputs, expected numerical failure. Zero and
  symmetric boundary inputs alone do not detect this bug; asymmetric signed
  and random inputs do. Maximum absolute error is about 0.51698.
- `rotation-batch-i9qaiev4/missing-key.json`: checking required -1 against an
  actual legacy +1,+2 key file returns code 2 (missing), as expected. This is
  key preflight evidence, not encrypted execution evidence.
- `fx-batch-hln_htdq/report.json`: independent deterministic FX rotate(+3)
  baseline passes all four inputs, maximum absolute error 1.662562672e-6.

| Step | MAE | Maximum absolute error | Maximum nonzero-reference relative error |
|---|---:|---:|---:|
| -3 | 8.624788048e-7 | 1.684016873e-6 | 7.026280977e-6 |
| -2 | 4.220197799e-7 | 9.882624247e-7 | 2.503603115e-6 |
| -1 | 9.726138598e-7 | 3.495238255e-6 | 8.794895020e-6 |
| +1 | 1.521732982e-6 | 3.491819964e-6 | 2.856455193e-5 |
| +2 | 9.551916020e-7 | 2.704088620e-6 | 1.077054423e-5 |
| +3 | 1.307487326e-6 | 4.886002323e-6 | 1.228857280e-5 |

These fixture errors do not prove a bound for all accepted inputs or compositions.
Do not interpret display rounding to 0.00 as zero CKKS error.

## Tests and reproduction

`test_rotation_contract.py` covers signed AST dispatch, version rejection,
hand-calculated reference direction, artifact policy and invalid steps, plus
opt-in validation of real golden and missing-key reports.
`test_model_graph.py` compares all six reference/Torch/FX/golden cases.

Offline regression with historical evidence and the new rotation evidence:
226 tests, 211 passed and 15 skipped. Separately, all 28 tests in the pinned
Torch environment passed. No real API requests were made for these checks.

From the WSL source directory, using already-built isolated dependencies:

```bash
python3 scripts/baseline/run_model_batch.py --unit-tests
python3 scripts/baseline/run_rotation_goldens.py
python3 scripts/baseline/run_model_batch.py \
  --case scripts/baseline/cases/rotate-plus3.json
```

The last two commands generate fresh FHE keys and fresh evidence without
calling a provider. Do not share whole result directories: they contain private
FHE keys. API credentials remain exclusively configured in project-root `.env`
(with explicit process-environment overrides); no provider-specific files are
required. Compiler/runtime subprocesses do not receive API credentials.

## Remaining gates

This increment does not complete multi-input, larger shape/broadcast/repacking,
general-width Linear/MLP, small Conv/Pool, polynomial approximation error
separation, doubled model-family coverage, or Poseidon GPU execution. The
semantic inventory explicitly retains these missing rows. Newly enabled v2
syntax has golden evidence but still needs live Agent-generation experiments.
CUDA installation requires separate approval; installation alone will not
resolve GPU modulus/scale, opcode and relinearization compatibility.
