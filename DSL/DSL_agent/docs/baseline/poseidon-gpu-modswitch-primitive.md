# Poseidon GPU CKKS Add / drop-modulus: real primitive evidence

Subsequent increment: [explicit physical-Q static schedule execution](poseidon-gpu-drop-schedule.md)
now passes through the real GPU backend. HEVM/compiler-profile mapping remains open.

## Outcome and scope

**Confirmed:** the native Poseidon GPU runtime builds with the isolated CUDA
12.5.1 components. A real CKKS encrypt -> GPU Add -> GPU modulus-prefix drop ->
download -> decrypt/decode test passes on device 0, compute capability 8.9.
The test requests `tc128`; a separate over-budget parameter set is explicitly
rejected by the library's security validation.

**Not completed:** Hecate/Earth/CKKS/HEVM execution on Poseidon GPU. This test
does not consume a compiler artifact. `ModswitchC` opcode 4 and `UpscaleC` opcode
5 remain rejected by the adapter. No opcode or existing GPU evaluator API was
changed by this increment. No new Agent calls were made.

## Reproducible profile and independent comparison

- N=16384, 8192 CKKS slots, input scale=2^40.
- Q: twelve 30-bit primes; P: two 30-bit primes (420 nominal total bits).
- Secret-key hamming-weight parameter 0; public-key encryption, software
  encode/encrypt/decrypt/decode, existing `GpuUploader` and `GpuEvaluator`.
- Every real input slot is `((i % 17) - 8) / 32`, including positive, negative,
  zero and the declared +/-0.25 endpoints. Reference is independently `2*x`.
- The GPU actually computes Add. CPU Add from the same encrypted input provides
  an exact RNS coefficient oracle, separate from the plaintext numeric oracle.
- All retained Q counts 12..2 are tested with direct output, source/destination
  aliasing and sequential drops. Metadata and all retained RNS coefficients
  must match the CPU result exactly; the original GPU source must not mutate.
- Decrypt each direct result and compare every complex slot with unchanged
  `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
- Q count 1 is not a valid numeric state at scale 2^40 with a 30-bit modulus;
  the test does not lower scale or relax the tolerance to force this case.

This is an explicit GPU-word-compatible primitive profile, **not a replacement
for the compiler's existing 60-bit-prime SEAL profile** and not a full model
performance or security result. Passing a library security check is not an
independent cryptographic security audit.

Six malformed requests must be rejected: empty ciphertext, upward Q transition,
unknown parameter id, P limbs, mismatched component metadata and partial shard
layout. Additionally, Q=13x30/P=2x30 at N16384/tc128 must fail specifically at
the library security-standard check. No over-budget context is used to encrypt.

## Actual evidence

Native results under `/home/lhy/poseidon-work/results/`:

| Directory | Outcome |
|---|---|
| `poseidon-gpu-build-364gimf5` | Initial executable failed before main: mixed Nix glibc/system loader. No FHE success. |
| `poseidon-gpu-build-2w66m5jz` | Fresh native-link configuration passed, no downloads. |
| `poseidon-gpu-build-uq891l5s` | Full native GPU runtime build passed with -j2. |
| `poseidon-gpu-modswitch-oqv1eh1y` | First real Add/drop pass, 11 Q states, max abs 4.7638965041550745e-7. Predates the additional security counterexample. |
| `poseidon-gpu-build-y6ym1i_l` | Final primitive source increment built without its earlier nodiscard warning. |
| `poseidon-gpu-modswitch-v41o9cps` | Final real GPU pass including security counterexample, all 11 states. MAE 6.182606042590216e-8; max abs 4.7544042363414554e-7. |

Each final Q state has the same decoded error in this run. Prefix drops preserve
the encoded value and scale; they are not a rescale operation. Cryptographic
randomness means a fresh run need not reproduce the same error digits.
The 11 states reuse one encrypted input, not 11 independent model cases.

The final report records source snapshots/hashes, executable/shared-library
hashes and stdout/stderr hashes. It reports `gpu_executed=true`,
`fhe_executed=true`, `hevm_executed=false`, `agent_calls=0`.
The native linkage checks reject `/nix/store/` dependencies or missing libraries.
The test runs with a small environment allowlist and no provider credentials.

## Build failure diagnosis and isolated fix

The initial binary requested `/lib64/ld-linux-x86-64.so.2` but its RUNPATH included
Nix glibc 2.39. Startup failed with `GLIBC_PRIVATE` / `__tunable_is_initialized`.
The build environment allowed Nix compiler/linker settings into a native-GCC
build. This was an environment/linking failure, not an FHE arithmetic failure.

`build_poseidon_gpu.py` now resolves the existing pinned CMake executable first,
then invokes it with native GCC/ld/make, explicit Ubuntu zlib paths, isolated
GMP/CUDA paths, and without Nix compiler/linker search settings. Generated files
live in `build-poseidon/agent-dsl-gpu-native`; the failed original build is kept.
`readelf` and `ldd` on the new binary/shared library show no Nix runtime paths.
The application runs outside the Nix shell; Dacapo's Nix environment is unchanged.
No system package, display driver or global environment was modified.

## Commands

Run in WSL from `/mnt/d/Code Space/Poseidon`:

```bash
# Configure/build use the already installed Nix CMake, GCC/GMP and CUDA components.
timeout -k 3s 220s python3 scripts/baseline/build_poseidon_gpu.py
timeout -k 3s 1850s python3 scripts/baseline/build_poseidon_gpu.py --build

# Fresh real GPU primitive execution with a new evidence directory. No model API.
timeout -k 3s 170s python3 scripts/baseline/run_gpu_modswitch_semantics.py

# Audit the final stored evidence, without executing a new GPU program.
POSEIDON_GPU_MODSWITCH_RESULTS=/home/lhy/poseidon-work/results/poseidon-gpu-modswitch-v41o9cps \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s scripts/baseline -p test_gpu_modswitch_semantics.py -v
```

The optional CTest target is `poseidon_gpu_ckks_modswitch_parity_tests` (90-second
test timeout). The older GPU smoke target is built as a compatibility check but
is not executed as this evidence; it contains small security-disabled fixtures.

Validation of this increment: the actual CTest entry passed 1/1 (1.27 seconds);
the full WSL offline suite discovered 290 tests, passed 265 and skipped 25
tests in the ordinary host Python; skips are not counted as passes. Existing stored CPU evidence
and the final GPU evidence were supplied to the audit tests. This was not a fresh
96-model encrypted rerun and did not make paid API calls. `git diff --check`
passed; its LF/CRLF conversion warnings are not patch errors.

## Next HEVM gate

This experiment supplies real primitive evidence needed to specify opcode 4;
it does not supply its entire adapter mapping. Next establish explicit compiler
profile -> physical Q/P count and encode-level mapping, integrate/verify the
static opcode transition, and validate multiply/relinearize/rescale/rotation as
needed by an actual Linear artifact. Unsupported mappings remain fail-closed.
No decrypt-and-reencrypt bootstrap substitute, implicit CPU HEVM backend,
multi-GPU execution or full ResNet benchmark is introduced.

The semantic inventory now has a separate `poseidon_gpu_primitive` status for
Add and modswitch. Its DSL/HEVM GPU status stays `not_validated`.
