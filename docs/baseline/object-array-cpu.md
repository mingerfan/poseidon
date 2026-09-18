# Object arrays / upstream SumSlots: real CPU evidence

## Outcome and scope

The new `--object-arrays` request-v18 / AST-v17 path is wired through request
rules, semantic guidance, AST validation, sandbox dependencies, normalization,
real Hecate tracing, Dacapo compilation, HEVM emission, SEAL CPU execution and
decrypted differential checks. Old contract rules remain unchanged. Switching
the grammar requires a fresh batch rather than being labeled a same-prompt retry.

Six **manual** goldens first matched all expected outcomes: four positives
passed and two deliberate semantic mistakes failed numerical comparison AFTER
real encrypted execution. Two independent-key repeats also passed. These are
not measured online Agent generation rates and do not validate Poseidon GPU.

All fixture helper ASTs match actual upstream fint/roll/SumSlots. The Linear
fixture uses SumSlots for real cross-element weighted reduction; it is not an
elementwise approximation, nor a claim that all of MPCB.HE_Linear is validated.

## Numerical results

Fixed tolerance: `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
Four fixed input groups per run: zero, signed fixed input, seeded random input,
and declared-range boundary. Original reference/weights/inputs are immutable.

| Fixture | Expected | MAE | Maximum absolute error | Result |
|---|---|---:|---:|---|
| sumslots4 | Sum four adjacent slots | 2.7371761e-6 | 9.7756590e-6 | Passed |
| sumslots3 | Sum three adjacent slots | 1.3732521e-6 | 2.3039270e-6 | Passed |
| linear | Two independent weighted reductions | 2.1723153e-9 | 3.7803497e-9 | Passed |
| empty_views | 1.5*x + .375 using Empty, alias and copy | 3.0007486e-9 | 8.0149192e-9 | Passed |
| wrong_stride | Wrong rotation direction | .14828382 | .51698272 | Numerically rejected |
| wrong_empty | Wrong sign after Empty subtraction | .51763045 | 1.000000002 | Numerically rejected |
| sumslots4 repeat 1 | Same program/parameters, fresh keys | 1.3418092e-6 | 3.1482738e-6 | Passed |
| sumslots4 repeat 2 | Same program/parameters, fresh keys | 1.1964473e-6 | 2.6978365e-6 | Passed |

Initial six: 24 input groups and 88 compared scalar outputs. Including repeats:
32 input groups and 120 scalar outputs. Reports retain per-element values/errors,
nonzero-reference relative errors, cosine similarity and pass decisions.

### Precision caveat

The first sumslots4 run is close to the absolute-error gate for zero reference.
Do not discard that run or summarize it as ample precision margin. Two fresh-key
repeats pass, but three finite runs do not establish a statistical failure bound
or all-input correctness. No threshold was relaxed and no candidate/parameters
were changed between runs.

Confirmed IR fact: this pure reduction's CKKS lowering has input scale 40,
init_level 13, a modswitch downFactor 12, then two rotations and two additions
at level 1. This identifies the numerical execution path, not the proven cause
of the larger error. Attributing the error specifically to modswitch, key switch
or encoding remains unconfirmed; diagnose those layers if further tests fail.

## Parameters, evidence and retention

SEAL 4.0.0, polynomial degree 32768, Q profile 14 x 60-bit moduli, tc128;
LLVM/MLIR 18.1.2 and NumPy 1.25.2. No bootstrap, decrypt/re-encrypt substitution,
weakened parameters, new backend, GPU execution or paid model calls.

- Main report: `/home/lhy/poseidon-work/results/object-array-goldens-esh4cq7d/report.json`
  SHA256 `d559814f9aa82deb7247b371c0288480959f351f0afbb9f41966bd65b80f8e13`.
- Repeat report: `/home/lhy/poseidon-work/results/object-array-stability-ud_wvata/report.json`
  SHA256 `87a195a3ebdbe91c2503002e3aa1c6c48ea329013ce3da1ac99df4ea4b069013`.
- Focused construction report:
  `/home/lhy/poseidon-work/results/object-array-checks-ndzndigf/report.json`.

All eight runs have complete key-cleanup receipts and no surviving private-keys
directory. Reclaimed **5,431,840,080 bytes (about 5.06 GiB)**. Original random keys
cannot be recovered; new test keys can be regenerated. Reports, fixed arrays,
original/normalized DSL, Earth/CKKS IR, HEVM/CST and decrypted outputs remain.

Verification: **229/229** fixed-environment semantic and artifact-audit tests
pass, zero skips. Full system-Python regression: **649 total, 500 passed,
149 conditional skips**, no failures. NumPy-dependent tests and real artifact
checks are separately executed in the pinned environment, not counted as passed
because of the system environment's skips.

## Reproduction

From the WSL source root, no paid API call:

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2000s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_object_array_goldens.py
```

One fixture through the same complete execution path:

```bash
timeout -k 3s 300s python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/construction-linear.json \
  --golden-file scripts/baseline/golden_cases/object_arrays/linear.py \
  --object-arrays --max-repairs 0
```

`test_object_arrays.ObjectEvidenceTests` accepts `POSEIDON_OBJECT_REPORT` and
`POSEIDON_OBJECT_STABILITY_REPORT` in the pinned environment. It checks actual
producer hashes, frozen inputs, original and normalized source, construction
metadata, artifact hashes, HEVM opcode gate, independent reference formula,
decrypted comparison, unchanged security parameters and key cleanup.

## Remaining full-goal requirements

Online Agent generation/repair validation remains distinct from manual goldens.
Vectorized object arithmetic, more array operations, broader packing/shape,
complete high-level helpers, other public-language semantics, IR calls, real
bootstrap/upscale and Poseidon GPU end-to-end validation remain incomplete.
The full DSL goal is active; this milestone does not redefine completion.
