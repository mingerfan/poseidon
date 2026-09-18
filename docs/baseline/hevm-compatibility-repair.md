# HEVM constant repair and remaining runtime contract (2026-09-05)

## Outcome

**Confirmed:** the HEVM-specific plaintext encoder now repeats CST vectors across
all slots, matching pinned Dacapo. The general Poseidon vector encoder still
zero-pads. The HEVM entry also rejects empty/non-finite constants and a level
that resolves to an auxiliary QP key context instead of a Q-only data context.

**Confirmed:** all 53 executable CPU adapter CTests passed (3 external-artifact
tests skipped), and the Python baseline suite passed 45/45 without skips. The
real Linear CST's four vectors encode/decode with maximum absolute error
`3.9968e-15` under the unchanged CPU tc128 defaults. This is **not** evaluation,
decryption or the artifact's intended parameter profile.

**Still blocked:** real add/mul_plain/Linear artifacts compile but all fail at
unsupported `ModswitchC`. No schedule is constructed. GPU execution and the
frozen plaintext-versus-decrypted comparison remain unrun.

## Location, inputs, outputs and semantics

`prepare_hevm_static_execution_plan` consumes loaded artifacts, constructs an
I/O binding plan and calls `encode_hevm_plain_inputs`. That function consumes a
Poseidon context, HEVM constant indices/scale exponents/levels and CST vectors;
it returns materialized CPU `Plaintext` objects for later GPU upload. It does
not choose a runtime profile or execute encrypted arithmetic.

Fixed Dacapo commit `4616402710f39df3e5f5bd7930a6c036025aaac3`:

- `lib/Runtime/SEAL_HEVM.cpp:255` (`encode_internal`) fills each slot with
  `src[i % src.size()]` before encoding.
- `lib/Runtime/HEAAN_HEVM.cpp:265` (`to_msg`) and `:283` (`encode_internal`)
  use the same repeated real-valued pattern.

The repair is isolated to `src/poseidon/frontends/dacapo/hevm_plaintext_encoding.cpp`.
No general encoder, GPU kernel, scheduler, Dacapo source or gitlink was changed.
For `S` runtime slots and a nonempty CST vector of length `m`, the encoded slot
`i` contains `constant[i % m]`, for `0 <= i < S`. Thus:

- `m=1`: scalar broadcast;
- `1<m<S`: repetition, including a partial final period when `m` does not divide S;
- `m=S`: unchanged vector;
- `m>S`: first S elements, matching the upstream loop rather than inventing a reshape.

Empty vectors have no defined repeated pattern, and non-finite payloads have
no accepted CKKS meaning here: both fail closed. Any diagnostic clears earlier
successful plaintexts; no partial upload set escapes. Scale and parameter IDs
are retained. The extra cost is one S-element double buffer per current encode,
released before the next iteration; this is correctness work, not a speed claim.

### Q-only level guard

`CrtContext` populates `parms_id_map` with Q-only levels at `|Q|-1` **and** the
key QP context at `|Q|+|P|-1` (`src/poseidon/crt_context.cpp:45-87`). Therefore a
successful integer lookup is insufficient to validate a HEVM data level.
The HEVM encoder now checks that P is empty and `|Q|-1 == level` after lookup.

The existing CPU tc128 default at N=16384 has eight Q primes and one P prime.
Data level 7 is valid; map index 8 is the QP key context and is now rejected
at this HEVM boundary. The separate direct-encoder regression retains the
previous index-8 test as a general encoder characterization. No context's
moduli, security mode or algorithm were changed.

## Red-to-green experiments

All files below are in `/home/lhy/poseidon-work/results`.

1. `hevm-constant-red-short-test.log`: the new repeated-scalar assertion failed
   at constant 0, slot 1. Direct encoder zero-padding had first passed with
   max error `4.97526e-13`. This is the minimal semantic reproduction.
2. `hevm-constant-red-contract.log`: the combined contract test also exposed the
   pre-repair rejection of an oversized vector, whereas upstream slices it.
3. `hevm-level-red-test.log`: the new Q-only guard regression failed because
   the unmodified entry accepted a key QP context as a data level.
4. `adapter-20260905T084329-475.log`: repaired C++ code rebuilt with `-j2`;
   CTest 53 passed, 3 skipped, 0 failed, including the requested real CST check.
5. `hevm-constant-green-real-cst.log`: direct zero padding preserved;
   repeated/sliced HEVM payload error `5.06484e-13`; real Linear CST error
   `3.9968e-15` across all 8192 slots. Scale exponent 48, Q-only data level 7,
   N=16384, existing tc128 context. No encryption or decryption.
6. `python-compiler-r2ekrtjt/report.json`: new real Hecate trace/compile run,
   all three cases compiled, constant check passed, adapter-blocked. Its
   `validation-tests.log` records 45/45 Python baseline checks passing.

The first targeted rebuild failed at `ld: cannot find -lgmp` because it omitted
the already installed isolated GMP prefix from `LIBRARY_PATH`. Restoring the
same command-scoped include/link paths used by `run_adapter_checks.sh` fixed
the link; no package was downloaded or installed. That failed build and its
follow-on missing-executable error are retained in `hevm-constant-red-build.log`
and `hevm-constant-red-test.log`; they are environment failures, not test results.

## ModswitchC: what is known, what is not

| Layer | Confirmed contract | Remaining boundary |
|---|---|---|
| Dacapo CKKS IR | Same component count, output level = input level minus downFactor (or sentinel level 0) in `CKKSDialect.cpp:122-139` | No concrete Q/P prime list in this type |
| Dacapo SEAL_HEVM | Positive downFactor invokes that many `mod_switch_to_next` calls | Zero/negative factors do not initialize a distinct destination; do not invent identity semantics |
| Dacapo HEAAN_HEVM | Invokes `levelDownOne` repeatedly and adjusts tracked scale with the library's current scale factor | Closed HEAAN implementation and physical/logical prime grouping are not inferred from names |
| Poseidon GPU `drop_modulus` | Copies retained Q-prefix limbs, updates parms_id/Q count, retains scale and other metadata; rejects P limbs and non-full shards | Actual GPU correctness/aliasing and agreed HEVM-to-physical-level mapping not yet tested |
| Poseidon GPU `rescale` | Removes one physical Q prime and divides scale by that prime | It is not a substitute for ModswitchC |

Code evidence: `gpu_evaluator.cpp:2016-2102`,
`gpu_modswitch_handler.cpp:985-1042`, and `gpu_evaluator.cpp:1591-1695`.
The separate `rescale_dynamic` entry uses a scale-based planner; the ordinary
`rescale` entry above removes one physical prime. Neither should be conflated
with modulus dropping.

**Evidence-based inference:** `drop_modulus` is a plausible primitive for a
Q-only, physical-level ModswitchC contract. It is not a verified adapter mapping.
Consistent with the repository's gate, opcode 4 remains unsupported; no fake
schedule/preflight success, rescale substitution or bootstrap fallback was added.

## Independent parameter blocker, before CUDA installation

The profile probe printed:

```text
q_count=8 p_count=1 q_over_uint32=8 p_over_uint32=1 hevm_input_level16_available=0
```

These values characterize the **existing CPU default**, not every possible
Poseidon parameter set. Its Q/P primes have 48/50 bits
(`parameters_literal.cpp:432`). GPU residues are `std::uint32_t`
(`gpu_memory.h:27`); the parameter uploader rejects values exceeding that range
(`gpu_parameter.cpp:81-102`, used when building Q/P tables). Therefore this CPU
profile cannot be reused unchanged as a GPU profile. It also lacks input level
16 required by the current compiler outputs.

The stock compiler config declares degree 131072, rescalingFactor 51 and a
HEAAN runtime. **That does not supply an exact Q/P list or a verified 32-bit
physical/logical level map.** Do not equate the rescalingFactor with a proven
prime list or assume installing CUDA fixes this boundary.

The inspected small GPU test profiles in `dacapo_gpu_runtime_smoke_test.cpp`
and the parameter factory in `test_gpu_bootstrap_modraise.cpp` use
`sec_level_type::none`. They remain test fixtures, not approved safe profiles.
No production-security conclusion can be drawn from them, and they were not run.

Before real ModswitchC/GPU validation, choose an explicit small-model execution
profile with **tc128 validation enabled**, exact Q/P values, degree, slot count,
scale, level mapping and required keys. Keep the numerical threshold fixed.
This needs approval under the user's parameter-change boundary; it is not
permission to weaken security or run the full ResNet profile. A compiler logical
level grouping implementation would be a separate, larger design choice.

Read-only GPU build inventory: bundled RMM 24.12.01 requires CMake >=3.26.4;
bundled CCCL is 2.5.0; optional GPU CMake requires CUDAToolkit but does not pin
its version and defaults architecture to 75. No Toolkit version was guessed,
no architecture changed, and no driver/Toolkit installation occurred this turn.

## Reproduce locally

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- timeout -k 3s 900s bash scripts/baseline/run_adapter_checks.sh
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/python_compiler_smoke.py
```

The first runs CPU-only checks; the second includes real Linear CST encode/decode
and still returns nonzero at the ModswitchC gate. Each creates its own result
log/directory. The constant diagnostic's CPU parameters are explicitly separate
from the compiled artifact's parameter requirements. Decrypted metrics remain
null/not_run; the frozen `1e-5 + 1e-4*abs(reference)` gate is unchanged.

## Local changes and handoff

Source remains `D:\Code Space\Poseidon`, branch `feat/agent-dsl-correctness`,
HEAD `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`. Dacapo and `.gitmodules` are
unchanged. Existing user edits, sparse checkout and the skip-worktree bit remain.

This turn changes the HEVM plaintext encoder/header, extends the existing
encoding test, adds the real-CST diagnostic to the Python compiler driver/test,
and updates baseline docs. No GPU/runtime opcode mapping was enabled. Suggested
future local commit: `fix(dacapo): preserve HEVM constant layout and reject QP data levels`;
keep prior environment/build work in a separate commit. No commit, push or PR
was performed because none was authorized.
