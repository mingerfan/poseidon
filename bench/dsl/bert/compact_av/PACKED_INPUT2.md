# PackedInput2Partial — independently named partial-output AV

**CPU-only evidence; no GPU execution or backend integration in this task.**
Select `AvSpec::algorithm = AvAlgorithm::PackedInput2Partial`, or use the
standalone probe's explicit `--pack-inputs` option (GPU work still requires
`--run-gpu` AND separate authorization). The default remains `Baseline`.
`av_algorithm_name`, `av_output_semantics` and per-ID `AvStats` explicitly report
which algorithm/output contract was prepared. Engine counters remain actual
call-site increments, not scheduled values substituted for measurements.

## Defined output, not a zero-gap AV ciphertext

T128, two heads per tile256, D64 live features per head, baby8, full128 cyclic
score offsets. **All128 score offsets are used**, including offsets64..127.
Client V must have genuine mathematical zeros in each head's feature64..127.
This is an encrypted-input contract: `run()` has no secret key or way to inspect
those amplitudes. The probe rejects dirty public V before encrypting it.

For live feature j<D, independently define

```
A[t,h,j] = sum over d in0..127 with (d-j) even:
           S[t,h,d] * V[(t+d)%128,h,j]
B[t,h,j] = sum over d in0..127 with (d-j) odd: same product
next(t,0)=(t,1); next(t,1)=((t+1)%128,0)
```

The accumulated packed product is `P=[A0,B0,A1,B1]`. Final `out=P+rotate(P,+D)`
has the following **fully defined all-slot** contract:

```
out[t,h,j]   = A[t,h,j] + B[t,h,j]        (j<D; complete original AV)
out[t,h,D+j] = B[t,h,j] + A[next(t,h),j]  (j<D; generally NONZERO)
```

The gap is neither zero nor unspecified garbage. Full32768-slot comparisons
must use the independent defined-partial oracle, not the baseline zero-gap AV
oracle. Separately compare live16384 slots with complete128-key AV.

**Do not connect bare partial output to the old unmasked reciprocal/merge
consumer.** A constant fixture makes the old merge double the result. No
parent Python, lowering, native adapter or backend integration is part of this
module change.

## Genuine packed-input arithmetic

1. Keep all128 V selector CPs, seven original V baby rotations and fifteen V
   giants. Accumulate exactly as baseline, then one actual Rescale2 L8→L6.
   No selector is deleted merely because its V feature is mathematically zero.
2. `Dpack = diag + rotate(diag, tile-D=+192)`. **+192 is not -64.** Make only
   three same-source even babies (token rotations512,1024,1536).
3. For score shift s=8g+b, intersect the original L/R masks with
   `Mlive[j]=(j<64)` **after score rotations**, not on the original score offsets.
   Shift0 uses a genuine Mlive CP, not unity. For s1..64 the live-intersected
   right mask is identically zero: omit exactly64 right CPs and64 right
   rotations. For s65..127 retain both CPs and their add.
4. Every C_b is really CP-normalized by Rescale4 L10→L6, using the same
   canonical common scale as baseline. Pack adjacent offsets as
   `Cpack=C_even+rotate(C_odd,-64)`; **-64 is not +64**. Multiply the packed
   score and corresponding even Dpack baby with ONE CC/relin/Rescale2 L6→L4.
   This is64 true CCs, not128 skipped key offsets or a feature-dimension change.
5. Original giant rotations/sums are followed by `P+rotate(P,+64)`.

No low-scale37 variant is implemented. Canonical scales remain score mask
log2=110.71270836717352, diagonal mask≈55.35255783895382, common≈55.3727119134628,
output≈55.4442212582195. Public mask coefficients are mathematical0/1. Encoding,
CKKS rounding/rescale/keyswitch noise may differ; no bitwise or encrypted exact
zero-gap claim is made.

## Actual schedule checks (not GPU measurements)

| Per pair, direct keys | Baseline (unchanged) | PackedInput2Partial |
|---|---:|---:|
| CP |383|319|
| CC / relin |128 /128|64 /64|
| rescale calls |257|193|
| Rescale2 / Rescale4 |129 /128|65 /128|
| physical prime drops |770|642|
| rotations |298|296|
| adds |381|319|
| genuine same-source hoist batches |19|19|

Packed rotations decompose into7 V babies +15 V giants +1 Dpack +3 Dpack babies
+15 score giants +112 left littles +63 right littles +64 odd packing +15 product
giants +1 unpack. Adds are127 diagonal +1 Dpack +63 L/R +64 Cpack +63 product
accumulation +1 unpack. Scalar tests increment counters at executed operation
sites and check these totals. GPU wrappers likewise instrument every actual
call; GPU counters remain unmeasured until separately authorized execution.

## Cache/key safety and API lifecycle

Mask-bundle identity includes the algorithm and physical-scale tuple. A
baseline full-mask bundle can NEVER satisfy the partial live-mask identity.
Baseline retains383 plaintexts; packed uses319 (all128 diagonals plus191 score
masks), or852,754,432 GPU coefficient bytes (813.25MiB) at N65536, versus
baseline1,037,303,808 bytes (989.25MiB). These are theoretical coefficient
storage sizes, NOT GPU memory measurements. IDs within each mode share one
bundle; two modes have two bundles, not72 copies per mode. Persistent memory
stats count unique bundles only and
exclude opaque scratch, run-local ciphertexts and allocator reserve.

Baseline rotation APIs remain52 direct keys / the old binary inventory. New
`required_packed_input_rotation_steps()` is an explicit55-key superset, adding
-64,+64,+192; the corresponding binary API is also exposed. The packed probe
requires all55 true direct keys. `--prepare-both-algorithms` prepares baseline
first, then packed, checking52→55 owning-key reupload, isolated bundles, all P11,
Q11 storage, and72-ID sharing inside each mode. These GPU-time assertions are
implemented but have not been executed in this CPU-only task.

**Prepare every ID/algorithm before the first run.** Any later `prepare()` throws,
including after a failed started run. Adding a mode before first run forces
owning key reupload with the union inventory, retains highest Q11/all P, and
re-prepares the Q-count views. Runtime parameters remain shared. This avoids
stale views or hoists into replaced owners. All new GPU ciphertext owners
precede `FailureSync`; ordinary adds use independent scratch+swap; rotations
use independent clones and owned workspaces. Successful run has one final
private-stream sync, not a sync per kernel.

## CPU evidence and consumer caveat

New scalar tests compare an independent indexed A/B partial oracle with the
packed BSGS implementation, including T16/D8 and all32768 T128 slots, distinct
heads, wrap, shifts0/64/65/127, d127 and nonzero upper score-offset lanes. Sign
mutations of Dpack/Cpack are negative controls. Baseline scalar direct/BSGS
retain full-feature cyclic arithmetic for dirty V; packed alone rejects dirty
client gaps. The separate strict `validate_values` diagnostic remains available.

Real N1024 CPU CKKS tests encode wide live masks and perform actual Rescale4,
pack two score/diagonal inputs, do one CC/relin/Rescale2, unpack with rotate+8,
and compare with two original terms and the defined partial oracle. A separate
CPU-only N65536 test encodes/decodes all319 production masks and densely verifies
exactly64 omitted right masks. Neither test claims encrypted fullT128 AV or GPU
correctness.

A scalar-only consumer check uses the same three Newton reciprocal iterations,
but the first folded seed is `d*(-M/T²)+2M/T`. The next two updates remain
`r*(2-d*r)`; their correction has gap2, not gap0. The outer r preserves ideal
zero gaps, and the existing context CC can therefore remove partial output
gaps. This verifies real-polynomial scalar equivalence only. An actual CKKS
seed-vector scale near55.43 with coefficient2^-14 has an amplitude budget near
41.43 bits; encrypted gaps are not exactly zero and noise ordering can differ.
The A2 consumer change is NOT implemented here and bare partial AV remains a
separate explicitly partial operator.

## Build without overwriting baseline evidence

Use the standalone CMake with the same read-only mature libraries, but a NEW
build directory under
`test-results/bert-a800-20261115/compact-av-build/packed-input2/`, and `-j1`.
`COMPACT_AV_BUILD_PACKED_CHECKS=ON` (default) adds the scalar, real CPU metadata
and full production CPU-mask tests. CTest never runs GPU. Default/help and
`--pack-inputs` without `--run-gpu` stay behind the device gate; guarded checks
are recorded only in the new evidence directory. No parent targets or frozen
baseline/v6 snapshots are changed.

From the repository root (CPU compilation/linking only):

```sh
out="$PWD/test-results/bert-a800-20261115/compact-av-build/packed-input2/local"
cmake -S bench/dsl/bert/compact_av -B "$out" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCOMPACT_AV_PREBUILT_BUILD="$PWD/build-runtime-gpu-api-release" \
  -DCOMPACT_AV_CPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_shared.so" \
  -DCOMPACT_AV_GPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_gpu.a" \
  -DCOMPACT_AV_DEPS_PREFIX=/usr -DCUDAToolkit_ROOT=/opt/cuda
cmake --build "$out" -j1
ctest --test-dir "$out" --output-on-failure
"$out/compact_av_probe" --pack-inputs --help # no device calls
```

### Completed local evidence

The new `packed-input2/` evidence contains a serial standalone build with **7/7
CTest passes**, a scalar-only **2/2** build, and ASan/UBSan passes for both scalar
suites. The packed scalar suite performs1,514,994 checks across1,264,640 compared
slots (max error2.14e-14), including64 one-hot cases and maskedRecip3 normalized
full-slot equivalence (max error8.33e-17). The actual scalar schedule counters
match the table above. The real N1024 encrypted-pair test passes all512 defined
slots (about1.1e-13 max error), and its packed live lanes match the two original
CC terms (about4.2e-14 difference); every physical drop is checked. The N65536
CPU-mask test really encodes/decodes all319 masks (10,452,992 slots, max complex
error4.55e-16), with all64 omitted right masks independently shown zero.

Guarded API lifecycle, ordinary default/help, packed default/help and mixed
preparation default paths make no CUDA device calls. **No `--run-gpu` was
executed**: GPU results/timing, actual GPU work counters, two-mode key reupload
and72-ID GPU sharing remain unmeasured. A complete source tarball and SHA256
inventory are saved beside `validation.json`; old baseline/v6 evidence remains
untouched.
