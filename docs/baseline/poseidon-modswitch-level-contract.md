# Poseidon Modswitch primitive and explicit HEVM level conventions

Date: 2026-09-07. This is CPU evidence preparing the GPU bridge, not GPU or
HEVM end-to-end execution. Evidence:
`/home/lhy/poseidon-work/results/poseidon-modswitch-semantics-l12c16zn`.

## Confirmed source semantics

At pinned Dacapo commit 4616402710f39df3e5f5bd7930a6c036025aaac3:

- `lib/Runtime/SEAL_HEVM.cpp::modswitch` invokes SEAL mod-switch-to-next
  `downFactor` times for positive factors. It does not divide the CKKS scale.
  A zero factor does not implement a general destination copy, so the current
  artifact gate correctly rejects no-op opcode4 instead of assuming copy behavior.
- That runtime fixes L=14, giving 13 initial data Q primes, and its encoder
  drops from L-1 down to the supplied level. Its level is the active Q **count**.
- Poseidon `Ciphertext::level()` is Q count minus one. The context
  `parms_id_map` also contains a QP key context, so checking for a map hit alone
  is insufficient.
- `gpu/gpu_evaluator.cpp::drop_modulus` already exists. It validates the
  single-full-Q-shard case, allocates the target level, preserves scale and NTT
  metadata, and calls `GpuModSwitchHandler::drop_modulus_ciphertext`, which
  copies the retained Q-limb prefix.
- The top-level `GpuEvaluator::rescale_dynamic` is implemented using
  `plan_gpu_dynamic_rescale` and `rescale_many`. A different lower-level
  `GpuModSwitchHandler::rescale_dynamic_ciphertext` remains a throwing stub.
  It is inaccurate to describe the whole top-level dynamic-rescale API as absent.

## Actual Poseidon CPU oracle

`examples/ckks/test_ckks_modswitch_semantics.cpp` uses the existing software
evaluator, not a new CPU HEVM backend. Parameters are the existing secure default
CKKS N=16384 / tc128 profile: eight 48-bit Q primes and one 50-bit P prime.
The primitive test explicitly uses scale 2^40 and fixed signed inputs, including
zero, across all 8,192 slots. Exact Q/P values are in the result.

After one real encryption, it tests every retained Q prefix, counts 8 through 1:

- direct, in-place and sequential drops agree exactly at the retained limb level;
- scale and NTT form do not change;
- input ciphertext contents and metadata remain unchanged by the direct call;
- each target is a Q-only context, never QP;
- decrypted values pass atol=1e-5, rtol=1e-4 at every level.

The final recorded run has MAE `3.089835214447446e-8` and maximum absolute error
`2.288100516036416e-7` at every retained prefix, identical to its roundtrip
baseline. This is eight checks of one encrypted input, not eight independent
encryptions. Randomized encryption makes rerun errors vary.

A rescale contrast drops one Q prime but changes scale from 2^40 to approximately
0.00390625; therefore it cannot replace scale-preserving modswitch. This contrast
is a metadata-semantics check, not a claim that this low post-rescale scale is a
suitable numerical configuration. Empty ciphertext input is rejected.
The CPU no-op prefix test does not authorize a zero-factor HEVM instruction.

## Explicit plaintext level convention, preserving the old API

The frontend now offers an overload:

```cpp
encode_hevm_plain_inputs(context, plan, constants,
                        HevmLevelConvention::ActiveQCount);
```

The original three-argument function retains its symbol/signature and delegates
to `HevmLevelConvention::PoseidonChainIndex`. Existing callers are unchanged.
No automatic source-profile detection or fallback is performed.

| Meaning | Active Q primes | Poseidon map index |
|---|---:|---:|
| ActiveQCount level 8 | 8 | 7 |
| PoseidonChainIndex level 7 | 8 | 7 |
| ActiveQCount level 1 | 1 | 0 |
| ActiveQCount level 0 | invalid | rejected |

The explicit mode subtracts one before resolving the map and verifies the
resulting context's actual Q count and absence of P. It rejects zero counts,
unknown enum values and QP key-context matches, with no partial plaintext list
on failure. `HevmEncodedPlaintext.level` retains the original input metadata;
callers must retain the explicitly chosen interpretation.

C++ tests use secure defaults to encode/decode repeated constants at the first
and last active-Q levels, and exercise invalid values and legacy behavior. The
same existing native test binary also retains earlier constant-layout tests.
Some older unit fixtures use explicitly small sec_level_type::none parameters;
those are encoding-only unit tests, not security evidence. The new primitive
and explicit-convention checks use tc128 defaults.

## What remains for GPU

This overload is a one-to-one mapping of **physical** Q counts. It does not
convert one 60-bit compiler level into multiple low-bit GPU primes. GPU storage
uses uint32_t GpuWord and rejects moduli that cannot fit. Neither the 60-bit
SEAL profile nor this 48-bit Poseidon CPU profile is thereby made GPU-compatible.

The default static execution-plan caller still uses the legacy overload. The
new explicit overloads below let a caller carry its selected constant-encoding
convention through plan preparation. A future compiler-profile-aware bridge must
select it from compiler provenance and align
all ciphertext/constant levels, concrete moduli, scales and required keys.
The HEVM header has no backend/profile tag, so provenance must be provided
outside it. ModswitchC opcode4 remains rejected by the adapter until the GPU
mapping and actual execution are verified. No GPU preflight was bypassed.

Needed next: approved CUDA toolchain, safe GPU parameter profile, actual
drop_modulus CPU/GPU parity tests (including alias/layout/device failures), then
the explicit opcode/level mapping and small HEVM execution. Multiply/relinearize
and rescale compatibility remain separate checks.

## Reproduce without installing dependencies

Run in WSL from `/mnt/d/Code Space/Poseidon`. Reuse the configured builds:

```bash
cmake -S . -B /home/lhy/poseidon-work/build-poseidon/agent-dsl-cpu \
  -DPOSEIDON_BUILD_DEPS=OFF

env CPLUS_INCLUDE_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/include \
    LIBRARY_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/lib \
  cmake --build /home/lhy/poseidon-work/build-poseidon/agent-dsl-cpu \
    --target test_ckks_modswitch_semantics -j2

env CPLUS_INCLUDE_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/include \
    LIBRARY_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/lib \
  cmake --build /home/lhy/poseidon-work/build-poseidon/agent-dsl-adapter \
    --target poseidon_mgpu_hevm_plaintext_encoding_tests -j2

env CPLUS_INCLUDE_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/include \
    LIBRARY_PATH=/home/lhy/poseidon-work/deps/gmp-6.3.0/lib \
  cmake --build /home/lhy/poseidon-work/build-poseidon/agent-dsl-adapter \
    --target poseidon_mgpu_hevm_static_execution_plan_tests -j2

python3 scripts/baseline/run_modswitch_semantics.py
```

The first attempted build had no target because the existing Makefile had not
been regenerated. After configuration, compilation initially missed gmpxx.h:
the isolated GMP include directory was not in that command's environment.
Explicit per-command include/library paths resolved this; no package, driver or
system setting was installed or changed. Configuration's printed /usr/local
install prefix was not used: no install command ran.

The result runner records stdout/stderr hashes, binary hashes and source
snapshots. The earlier development run ending `7jxdpaqv` is preserved; the
final `l12c16zn` run includes the P-modulus record and stronger source metadata
invariance checks.

## Explicit convention at the static-plan entry

The existing entry signatures and their legacy default behavior are preserved.
Callers may now explicitly use:

```cpp
auto prepared = prepare_hevm_static_execution_plan_from_files(
    context, paths, pipeline_options, HevmLevelConvention::ActiveQCount);
// Equivalent for an already loaded DacapoHevmArtifactResult:
auto in_memory = prepare_hevm_static_execution_plan(
    context, artifacts, HevmLevelConvention::ActiveQCount);
```

Successful plans record `plaintext_level_convention`; the name deliberately
limits it to constant encoding. Schedule attributes, ciphertext input/output
metadata, constant values and debug dumps retain their original meaning and
values. No implicit profile detection, I/O conversion or GPU-readiness claim is
made. Adding a trailing plan field preserves existing source aggregate callers,
but consumers must rebuild; this is not a binary-layout compatibility promise.

The new native tests use mock HEVM/CST files, through both file and in-memory
entry points, with the secure N16384/tc128 default Poseidon context. At active-Q
counts 1, 2 and 8, actual encoded constants decode correctly across every slot
at scale 2^40. The same numerical level 2 selects Q2 in explicit count mode and
Q3 in legacy index mode, demonstrating why auto-detection is unsafe. A legacy
level 8 resolves to the QP key context and must fail, not fall back to count mode.

Counts 0, 9 (QP) and 63 (absent), and an unknown convention are rejected with
stage-specific diagnostics and an empty plan. Both conventions still reject
ModswitchC and UpscaleC at artifact translation, before a plan exists. Existing
missing-artifact, I/O and encoding-error tests remain intact.

`run_modswitch_semantics.py` now runs a third check, `static_plan_convention`,
and snapshots its source and runner. This is **constant encode/decode plus
CPU-side plan preparation** on mock artifacts; only the separate `primitive`
check encrypts data. Neither check executes HEVM or GPU operations.

The integrated run is
`/home/lhy/poseidon-work/results/poseidon-modswitch-semantics-2mwjd04c`.
All three native checks returned zero. The separately encrypted primitive rerun
has MAE `3.118419914984334e-8` and maximum absolute error
`2.0148281228908209e-7`, identical across its eight retained Q prefixes.
The plan test itself does not encrypt data. The older `l12c16zn` primitive
report above is preserved rather than overwritten.

Validation after this integration: five rebuilt adjacent CTest targets passed
(static execution plan, plaintext encoding, I/O binding, artifact readiness,
Dacapo adapter); full ordinary-WSL Python regression found 277 tests, with
254 passing and 23 skipped. The skips are not new execution evidence. The
intentional `--max-repairs 4` argparse diagnostic belongs to a negative test.
An initial combined CTest regex lost quoting across PowerShell/WSL and failed
before tests ran; invoking each exact target separately passed. No system
configuration change was needed.

Validate the integrated evidence with:

```bash
POSEIDON_STATIC_PLAN_CONVENTION_RESULTS=/home/lhy/poseidon-work/results/poseidon-modswitch-semantics-2mwjd04c \
  python3 -m unittest discover -s scripts/baseline -p test_static_plan_conventions.py -v
```
