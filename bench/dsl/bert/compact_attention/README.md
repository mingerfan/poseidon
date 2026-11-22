# Prepared cyclic QK (one two-head pair)

**Status update: the standalone GPU pair and integrated one-block v4 have now
been measured.** Pair median0.296522763s, all32768-score max error1.57e-10;
integrated full block8.222786937s/23 native Boots (1warmup+3samples), without AV.
See `../../../../test-results/bert-a800-20261115/v4-a800/20261115T140945Z/FINAL-REPORT.md`
and `../PREPARED_QK.md`. Full-model accuracy is not passed. This helper remains
QK only, not AV or an automatic compiler result; backend integration is separate.

## Interface

`prepared_qk.h` exposes the CUDA-header-free `bert::attention::QkSpec` and
`PreparedQkEngine(context, shared_ptr<const RelinKeys>,
shared_ptr<const GaloisKeys>, device=0)`:

- Constructor does no CUDA work. `prepare(spec)` does untimed full CKKS mask
  encoding, real mask upload, parameter creation, and evaluation-key upload.
- `run(id, const GpuCiphertextData &query, const GpuCiphertextData &key)`
  enqueues one complete pair on its private stream. Caller synchronizes inputs;
  engine synchronizes that private stream once before returning the ready GPU
  ciphertext. No per-primitive Runtime calls, secret keys, decrypt/refresh, or
  host-computed output. One engine is serialized, not thread-safe.
- The unchanged `required_rotation_steps()` enables baseline direct keys and
  **two genuine shared hoist batches** (query giants15, key babies7).
  `required_paired_rotation_steps()` adds exactly `-64` and `+63` for paired2;
  `required_packed_input_rotation_steps()` adds `-64`, `+63`, `+192` to the old50
  for pre-CC packed inputs. Their corresponding binary methods
  (`required_binary_rotation_steps()`, `required_paired_binary_rotation_steps()`,
  `required_packed_input_binary_rotation_steps()`) enable exact, slower
  signed-binary fallback; positive-power-only keys are
  also supported by cyclic normalization. No head-reduction hoist is falsely
  reused after the source changes.
- Prepare **all specs before the first run** (later prepare calls are rejected),
  preferably highest physical input level first. Mixed baseline/paired/packed-
  input specs upload their true rotation-step union; adding a lower-Q new mode still
  uploads its extra keys at the existing highest Q and rebuilds all key views.
  Baseline-only setup does not require the paired keys. Owning keys are truly
  cropped to the highest prepared Q count and HYBRID digit count, retaining **all context P**
  (11 for q52-p11). Smaller active views never imply smaller owning storage.
  Each runtime owns parameters for *every* physical Q count from output+1 to
  input+1, including both intermediate rescale steps. The bench uses the stock
  cropped uploader plus the same mature inverse-pre-rotation kernel in a local
  guarded setup helper: this closes the stock double-hoist uploader's nested
  exception-lifetime gap without modifying/rebuilding the core library.
- `qk_stats(id)` / `engine_stats()` provide actual primitive-call counters,
  mask/setup statistics and allocation accounting. `allocated_bytes` is the
  persistent keys + parameter tables + masks; it explicitly **excludes** opaque
  evaluator/rotation scratch, run-local ciphertexts, allocator pool capacity and
  allocator overhead. Counters are cumulative across runs, including operations
  already enqueued by a subsequently failed run.

Inputs are immutable, size2, single-device, single-full-shard, Q-only NTT
ciphertexts. Fixed production geometry is N65536, slots32768, T128, tile256,
head_dim64, baby8. In each token's 256 lanes only `0..63` and `128..191` hold
features; all other input lanes are **zero**. The encrypted gap precondition
cannot be inspected by the engine (the standalone input validator checks it
before encrypting).

## Exact mathematics and physical scale

Left rotation is `rotate(k)[i] = input[(i+k)%32768]`. Output is **cyclic key
distance**, not a conventional row-major score matrix:

```
output[t*256 + h*128 + d] = dot(Q_h[t], K_h[(t+d)%128]) / sqrt(64)
```

For each `g=0..15`, `b=0..7`, the engine performs:

1. Shared `Q.rotate(-2048*g)` and `K.rotate(256*b)` inputs.
2. True CC multiply, true relinearize, true Rescale2 to physical `Lin-2`.
3. Head reduction using successive **current-source** `rotate(1,2,4,8,16,32)`
   and adds.
4. One genuine full CKKS head-start mask CP multiply (`1/8` every128 slots),
   then `rotate(-b)` for packing.
5. Sum babies, `rotate(2040*g)`, sum all giants, and **one** final Rescale2.

In the default baseline no key, head, or CC product is skipped: per pair there
are CC128, relin128, CP128, Rescale2 calls129 (258 actual Q primes), adds895, and direct rotations917
(including the22 hoisted outputs). All ordinary arithmetic destinations are
nonaliasing; adds use independent scratch + swap. Specialized rotate-inplace
runs on independent copies / owned pack/group buffers with retained workspaces.
Run-local GPU owners outlive an exception synchronization guard and release
inside the private stream scope.

Physical Poseidon level `L` means `q_count=L+1`. If input logscale is `s`,

```
intermediate = 2*s - log2(q[Lin]) - log2(q[Lin-1])
mask_scale = requested_output - intermediate
             + log2(q[Lin-2]) + log2(q[Lin-3])
Lout = Lin-4
```

The declared intermediate is validated, and real evaluator results are checked
after each Rescale2. No scale/parms-id metadata is overwritten to fake alignment.
Defaults use actual canonical q52-p11 levels8 ->6 ->4:
`55.35255783895382 ->55.3727119134628 ->55.4442212582195`, not nominal56.
The exact prime pool in `diagnostic_profile.h` is copied from
`../generate_aligned.py`; q52 takes the first52, P takes the **last11**.

## Opt-in paired products (`QkSpec.pair_products = true`)

Default is **false**, retaining the original baseline arithmetic and direct key
set. Paired mode exploits **ideal-message zero gaps**, not lower degree, omitted
keys, or fewer CC products. For each giant and even baby `b=0,2,4,6`:

1. Compute `P0=Q_g*K_b` and `P1=Q_g*K_(b+1)` with two independent genuine
   CC/relin/Rescale2 sequences at the same actual intermediate scale.
2. Add `P1.rotate(-64)` to P0. Each256 row now has four contiguous64-lane
   blocks: head0/b, head0/(b+1), head1/b, head1/(b+1).
3. Share six **changing-source** reduction rotations/adds (1..32), not a hoist.
4. Multiply by two separately encoded full CKKS masks: `/8` at lanes0/128 and
   `/8` at lanes64/192. Rotate the second result by `+63`, add, then pack by
   `rotate(-b)`. Accumulate the four pairs, apply the original giant rotation,
   sum all giants and perform the original single final Rescale2.

| Actual primitive calls per full pair, direct keys | Baseline | Paired2 |
|---|---:|---:|
| CC / relin / CP | 128 / 128 / 128 | 128 / 128 / 128 |
| Rescale2 / physical Q drops | 129 / 258 | 129 / 258 |
| Rotations / adds | 917 / 895 | 597 / 575 |
| Same-source hoist batches | 2 | 2 |

The rotations/adds are counted at the real evaluator callsites, and the scalar
paired reference counts its own operations independently. New direct keys
`-64/+63` are generated explicitly by the paired probe so binary fallback does
not silently undo the direct-key reduction. Binary-policy rotations may require
more physical calls; do not infer the direct counts or a speed gain for them.

Paired mode is **not valid for arbitrary dirty gap messages**: shifting P1 would
mix those gaps into the features. The CPU paired reference and probe reject
nonzero ideal input gaps, and the production caller must guarantee masked gaps.
FHE gaps inevitably contain encoding error/noise and are **not bitwise zero**;
this rewrite changes their propagation, CKKS rounding/noise can change, and no
bitwise ciphertext equality, decrypted gap check, refresh, GPU correctness or
speed is claimed. Engine setup encodes two masks, preserves actual scales, and
reports `schedule=paired2_gap_shared_headreduce` and
`encoding=full_ckks_dual_head_start_masks` in QkStats/probe output.

## Separate pre-CC packing (`QkSpec.pack_inputs = true`)

This second flag is also **OFF by default** and is **mutually exclusive** with
`pair_products`; setting both fails before GPU setup. It does not alter paired2.
The ideal-message zero-gap domain and non-bitwise CKKS noise caveat above apply.
Unlike paired2, this mode genuinely uses otherwise empty SIMD slots to halve the
number of physical CC products while retaining **all128 key offsets, both heads
and all64 features** (no sparse-key/head substitute or polynomial reduction).

Before any CC, perform two real rotations/adds with independent destinations:

```
Qpacked = Q + Q.rotate(-64)
Kpacked = K + K.rotate(256-64) = K + K.rotate(+192)
```

Within a token's four64-lane blocks, the ideal packed messages are
`Qpacked=[Q0[t],Q0[t],Q1[t],Q1[t]]` and
`Kpacked=[K0[t],K0[t+1],K1[t],K1[t+1]]` (tokens cyclic).
The two same-source hoists now read the **actual owned packed inputs**:
15 query giants from Qpacked, and only3 key rotations (`512/1024/1536`) from
Kpacked. No decomposition of raw Q/K is falsely reused for packed sources.
For every giant and even baby, **one** genuine CC/relin/Rescale2 already contains
the two neighboring baby products in its four64-lane blocks. The changing-source
reduction, two genuine masks, B.rotate63/add, even-baby packing, giant correction,
full sum and final Rescale2 are exactly the shared-reduction suffix of paired2.

| Actual primitive calls per full pair, direct keys | Baseline | Paired2 | Packed-input2 |
|---|---:|---:|---:|
| Logical key offsets / heads / features | 128 / 2 / 64 | 128 / 2 / 64 | 128 / 2 / 64 |
| True CC / relin / CP | 128 / 128 / 128 | 128 / 128 / 128 | 64 / 64 / 128 |
| Rescale2 / physical Q drops | 129 / 258 | 129 / 258 | 65 / 130 |
| Direct rotations / adds | 917 / 895 | 597 / 575 | 531 / 513 |
| Hoisted query / key outputs | 15 / 7 | 15 / 7 | 15 / 3 |
| Actual same-source hoist batches | 2 | 2 | 2 |

No input or output scale metadata is changed: the two packing adds preserve
actual input level/scale, the CC path remains physical8->6, the common final
Rescale2 remains6->4. Raw input owners are immutable; packed Q/K and all scratch
live within the private stream/failure guard. Stats use
`schedule=packed_input2_pre_cc_gap`, `pack_inputs=true`, and actual primitive
call counters. The probe's explicit `--pack-inputs` generates and checks genuine
direct `-64/+63/+192` keys. Baseline and paired2 keep their original keys/math.

## Build and CPU-only validation

CMake imports read-only mature libraries; it never rebuilds core Poseidon.
No new CUDA kernel is needed, so this helper compiles as C++17 and links the
existing GPU archive. The default paths match the frozen 20261114 toolchain;
override all matching paths together when building locally:

```sh
cmake -S bench/dsl/bert/compact_attention \
  -B test-results/bert-a800-20261115/compact-attention-build/packed-input2/local \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DATTENTION_PREBUILT_BUILD="$PWD/build-runtime-gpu-api-release" \
  -DATTENTION_CPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_shared.so" \
  -DATTENTION_GPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_gpu.a" \
  -DATTENTION_DEPS_PREFIX=/usr -DCUDAToolkit_ROOT=/opt/cuda
cmake --build test-results/bert-a800-20261115/compact-attention-build/packed-input2/local -j1
ctest --test-dir test-results/bert-a800-20261115/compact-attention-build/packed-input2/local \
  --output-on-failure
```

For a completely dependency-free scalar-only build, set
`ATTENTION_BUILD_GPU=OFF` and `ATTENTION_BUILD_METADATA_CHECK=OFF`.
CPU tests never create a GPU context:

- `attention_math_check`: independent indexed dot-product oracle versus exact
  scalar rotation schedule; T16 nonzero giant/wrap, all32768 T128 score indices,
  two distinct heads, head isolation and real gap zeros, anti-row-major check,
  malformed/nonfinite input rejection and exact-prime physical scale checks.
  Paired checks compare all32768 production slots against both the unchanged
  baseline and independent oracle, cover final odd offset127/wrapped distinct
  heads and nonzero giants, reject dirty gaps/illegal layouts, and count actual
  scalar CC/CP/rotate/add callsites (abstract R2/logical batch counts are labeled).
- `attention_key_plan_check`: dependency-free default-off/direct key contract,
  exact paired-only extras, signed-binary completeness and both mixed-spec
  preparation orders, including all three modes, lower-Q additions, highest-Q
  owner retention and conflicting-flag rejection.
- `attention_metadata_check`: **genuine CPU** encryption, CC, relinearization,
  Rescale2, changing-source head reduction, mask CP, Rescale2, decryption. Uses a
  small N1024 ring and the same actual Q-prime prefix/full11 P, physical8->6->4;
  checks actual metadata without overwrites. It additionally compares two real
  encrypted products with shared reduction/dual masks against the old separate
  term path, including the -D/+D-1 rotations. This tests the crypto scale contract,
  not GPU or a full encrypted T128 pair. The packed-input extension performs
  true encrypted pre-CC Q/K packing followed by one CC chain, compares it with
  two original separately computed adjacent-key CC chains and an independent
  scalar target, and verifies the same actual output scale.
- `attention_qk_probe --help`: exits before any CUDA call, allocator or context.
  Running with no arguments prints help and returns2, also without CUDA.

## Standalone GPU probe — explicit opt-in and bounded resource protocol

`attention_qk_probe` **requires `--run-gpu`** before any CUDA work.
`--pair-products` separately opts into paired2 (all128 CC terms preserved), while
`--pack-inputs` opts into true pre-CC SIMD packing (64 CC terms, all128 logical
key offsets). The flags cannot both be true and never bypass the GPU gate or
change the default baseline. It defaults to
explicitly labeled diagnostic random Q/K (not full-model projections), with two
different heads and genuine gaps. Optional `--query` and `--key` accept32768
packed raw float32 slots each; provenance is labeled merely `provided_gapped_qk`,
never asserted to be the original full model. Inputs must be finite with exactly
zero gap lanes. It performs genuine CPU encrypt -> GPU upload -> prepared QK ->
CPU download/decrypt and compares **all32768** scores against an independent
O(T*T*D) direct oracle. Default context is exact q52-p11/HWT32768; no bootstrap or
ephemeral secret is used by this QK-only operation.

Exactly1 synchronized warmup and3 synchronized samples are collected; no setup,
key generation, encoding, encryption, upload, decryption or CPU oracle is timed.
Actual four-run operation counters are checked. Nonfinite inputs, outputs,
errors, scales and aggregate metrics fail. `--max-abs-error` defaults to1e-4.
`--output` writes decoded cyclic scores as float64. See `--help` for options.

**Do not run with `--run-gpu` until the parent session authorizes the GPU window.**
The completed build/CPU tests are not evidence that the GPU engine is correct or
faster; that validation is deliberately deferred while another worker owns A800.

### Paired2 local evidence (no GPU; preserved historical evidence)

The paired2-stage artifacts live under
`test-results/bert-a800-20261115/compact-attention-build/paired2/`; previous
baseline evidence is not overwritten. Local GCC15 import-only build/link used
`-j1`; CTest passed5/5, dependency-free scalar-only CTest2/2, and the scalar
ASan/UBSan/leak-detection run passed. There are376506 always-on math checks and
275552 compared slots; paired-versus-independent-oracle max error is3.553e-15
and paired-versus-baseline scalar max error is0. The true N1024 CPU encrypted
paired experiment has zero physical scale-log2 error, full-slot decode error
about6.00e-14 and paired-versus-separate starts error about2.30e-15 (randomized
cryptographic noise varies per run). These are CPU tests, not a full encrypted
T128 GPU pair proof. See `paired2/validation.json` and the accompanying logs.

### Packed-input2 local evidence (no GPU)

New artifacts are separate in
`test-results/bert-a800-20261115/compact-attention-build/packed-input2/`; old
paired2/baseline logs and hashes are not overwritten. GCC15/C++17 import-only
`-j1` build/link passes, CTest7/7 and scalar-only CTest2/2 pass, and the scalar
ASan/UBSan/leak check passes. The634103 always-on checks compare515800 slots;
packed-input max error versus the independent oracle is3.553e-15 and versus
both old scalar modes is0. Actual scalar sites independently count64 CC,
128 CP,531 rotations and513 adds;65 R2/two source batches are explicitly
ABSTRACT/LOGICAL in scalar tests, not real hoist measurements.

The true N1024 same-query adjacent-key experiment compares pre-CC ONE-CC
packing against the original TWO-CC pair and the independent scalar target
across all512 slots/64 head starts. Latest logged decode error is4.516e-14,
versus-old-pair error3.772e-14 and final physical scale-log error0 (cryptographic
noise varies per run). No decryption checks a gap or refreshes engine output.
The public flags conflict explicitly before GPU setup; CTest exercises baseline,
paired, packed and binary-packed authorization guards without `--run-gpu`.
See `packed-input2/validation.json`, source hashes and the new source archive.
These remain CPU/source/build evidence only, not production-N GPU validation,
GCC11/nvcc11.8 validation, or a measured speedup.
