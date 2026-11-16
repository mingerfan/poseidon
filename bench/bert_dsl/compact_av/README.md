# Standalone prepared cyclic AV (no backend integration)

`prepared_av.h` exposes CUDA-header-free `bert::attention::AvSpec`,
`PreparedAvEngine`, `AvStats`, and `AvEngineStats`. **Baseline remains the default.**
The separately named `AvAlgorithm::PackedInput2Partial` has complete live AV but
DEFINED NONZERO gap partials, not the baseline full-slot contract; see
[PACKED_INPUT2.md](PACKED_INPUT2.md). Nothing below silently selects that mode.
Constructor does no CUDA
work. `prepare()` uploads real encoded masks, parameters and owning evaluation
keys; prepare ALL IDs/algorithms before first run (later prepare is rejected).
`run()` accepts immutable caller-synchronized GPU ciphertexts; it has no
secret key, host answers, decrypt/refresh, or Runtime/backend calls. A successful
pair synchronizes its private stream once at the end (cold allocator growth may
also synchronize). Not concurrently callable.

**Evidence update:** standalone real GPU probe now passes: pair median0.146162465s,
all32768-slot max abs error6.24e-9, four-run actual counters checked. Preparing72 IDs
shares exactly one1,037,303,808B mask bundle. See
`../../../test-results/bert-a800-20261115/v4-a800/20261115T140945Z/FINAL-REPORT.md`.
This is not full encoder/72-pair latency. Parent has separately integrated the
backend (`../PREPARED_AV.md`), whose full-model GPU validation is still pending.
The standalone still requires explicit `--run-gpu` and bounded resource admission.

## Mathematics and physical arithmetic

Exact real-arithmetic layout from `../cerium_packing.py::_Arithmetic.cyclic_av`:

```
score[t,h,d], V[t,h,j], t,d,j=0..127, h=0..1
out[t,h,j] = sum(d=0..127) score[t,h,d] * V[(t+d)%128,h,j]
slot(t,h,j) = t*256 + h*128 + j
```

V features 64..127 of **each** head are zero; scores have **all 128** key-offset
lanes. Rotations are LEFT: `rotate(k)[i]=x[(i+k)%32768]`. Baby=8, giants=16.
All128 CC products, relinearizations and post-CC Rescale2 operations remain,
including lanes containing zero features. AV is weighted V **before** surrounding
normalization; this standalone diagnostic does not claim full BERT, Softmax or
normalization execution/measurement.

The initial implementation deliberately accepts only N65536 / 32768 slots and
physical L10 score, L8 value, L6 common, L4 output (level L means Q-count L+1).
Scales are checked against actual ordered context primes; they are never
rewritten as an alignment substitute. Defaults are the canonical q52-p11 scales.
Input, plaintext-mask, pre-rescale CP/CC and output scales must also leave32 bits
of actual Q-modulus capacity. This rejects infeasible noncanonical scale plans;
it does not prove absence of overflow for unbounded encrypted amplitudes, which
remain the caller's CKKS message-range responsibility.

1. V: 128 selector CPs accumulate the original BSGS diagonal, then **one real
   Rescale2** L8->L6. Seven common-level baby rotations follow.
2. Scores: each nonzero cyclic shift has two genuine 0/1 periodic masks. Their
   plaintext scale is derived as
   `S6 + log2(q10*q9*q8*q7) - S10` (about110.7). Add both CPs at L10, then one
   **real Rescale4** L10->L6. Shift0 instead uses one full-slot unity CP at that
   same scale and Rescale4; it has no redundant L/R branches.
3. Each normalized score times its normalized diagonal does CC/relin/Rescale2
   L6->L4, before original giant-group accumulation and rotations.

This truly removes the old repeated score alignment `CP(1)+Rescale2`, rather
than altering metadata. Public mask coefficients remain exact mathematical 0/1,
not quantized coefficients. CKKS FFT/encoding and rescale noise can differ from
old two-CP/two-Rescale2 arithmetic; **bitwise identity is not promised**. CPU
metadata tests compare decoded values, real levels and scales, including two
distinct rotated score sources whose masked CPs are added at L10 BEFORE the
single Rescale4 and then passed through real CC/relin/Rescale2.

## Scheduling, sharing, instrumentation

Actual evaluator call sites increment cumulative counters (including calls
completed in a failed run). With direct rotation keys a complete pair has:

| primitive | actual schedule count |
|---|---:|
| CC / relinearize | 128 / 128 |
| CP | 383 = 128 selectors + 254 L/R + 1 unity |
| logical rescale calls | 257 = 129 Rescale2 + 128 Rescale4 |
| physical dropped primes | 770 = 2 + 128*4 + 128*2 |
| rotations / adds | 298 / 381 |
| genuine same-source hoist batches | 19 |

The prior pair had510 CPs and384 rescale calls:127 of each are removed. Hoisting
is only across identical immutable sources: seven V babies, seven normalized
diagonal babies, fifteen score giants, and one L/R little batch for each of the
sixteen score-giant sources (g0 omits the unused shift0 right rotation). The two
sets of fifteen changing-source giant rotations use mature `rotate_inplace`
with owned reusable scratch. Binary-key fallback counts its real component
rotations and does not pretend to hoist changing sources.

All ordinary add/multiply/relinearize/rescale outputs have different owners;
add accumulators use scratch+swap. Private-stream scopes and inner exception
synchronization protect asynchronous GPU and host-buffer lifetimes. The local
inverse-pre-rotation helper uses existing mature kernels, not new core changes.

A physical-scale tuple shares **one MaskBundle across all IDs**:128 diagonal
masks at Q9 and255 score masks at Q11. At N65536 this is1,037,303,808 GPU coefficient
bytes (~989.25MiB), not72 copies for6pairs*12layers. The engine also shares one
runtime and owning Q11-cropped relin/Galois keys, retaining **all P11**. No dynamic
P and no fake full-Q view crop. Persistent byte counts include keys, parameters
and unique mask bundles only, excluding run-local ciphertexts, workspace/pool
reserve and opaque evaluator scratch. The probe prepares72 same-physical IDs
by default and verifies persistent bytes/bundle counts do not grow.

## Build (imports only; no core rebuild)

Local example from repository root:

```sh
cmake -S bench/bert_dsl/compact_av \
  -B test-results/bert-a800-20261115/compact-av-build/local \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCOMPACT_AV_PREBUILT_BUILD="$PWD/build-runtime-gpu-api-release" \
  -DCOMPACT_AV_CPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_shared.so" \
  -DCOMPACT_AV_GPU_LIBRARY="$PWD/build-runtime-gpu-api-release/libposeidon_gpu.a" \
  -DCOMPACT_AV_DEPS_PREFIX=/usr -DCUDAToolkit_ROOT=/opt/cuda
cmake --build test-results/bert-a800-20261115/compact-av-build/local -j2
ctest --test-dir test-results/bert-a800-20261115/compact-av-build/local --output-on-failure
```

CMake uses C++17 and the existing GPU archive's kernel symbols; no CUDA language
or kernel compilation. Cache paths can instead select the A800 GCC11/nvcc11.8
20261114 mature build and matching generated headers. That GCC11/CUDA11.8 build
and standalone GPU probe have now passed in the frozen14:02:22 snapshot. To exclude GPU compilation/linking, set `COMPACT_AV_BUILD_GPU=OFF`.
For dependency-free scalar tests also set `COMPACT_AV_BUILD_METADATA_CHECK=OFF`.
CTest contains only scalar tests, real **CPU** encrypted metadata tests, CPU API
lifecycle/rotation-inventory checks and probe `--help`; it never runs `--run-gpu`.

## Opt-in GPU probe (respect standing authorization and resource protocol)

`compact_av_probe` or `--help` makes no CUDA device API calls. No-argument exits2,
help exits0; both paths also pass an LD_PRELOAD guard that aborts on the linked
device APIs. Static kernel fatbinary registration by the mature archive is not
device execution or context creation. Only explicit `--run-gpu` can construct a
context/allocator and enter GPU work.
The probe encrypts supplied/diagnostic full cyclic scores and zero-gap V, uploads,
runs1warmup+3samples, decrypts, compares **all32768 slots** to the independent
O(T^2*features) direct oracle, rejects nonfinite inputs/results/metrics, verifies
real counters and reports gap error. Optional packed inputs are exactly32768
float32 values each (`--scores FILE --value FILE`); decoded output can be written
as32768 float64 values with `--output FILE`. Default random inputs are clearly
marked **diagnostic**, not model outputs. Timing is single-pair AV only.
