# GPU CKKS Encode benchmark

This is an experiment, not a runtime Encode implementation. It encodes real
full-slot inputs on GPU: slot permutation, double-precision cuFFT, negacyclic
twist/scaling/rounding, exact RNS expansion, and forward NTT. CUDA and Tensor
variants share the FFT and RNS kernels. The Tensor variant uses existing INT8
Tensor Core TAM kernels for eligible NTT stages; remaining stages use CUDA
cores. It does **not** measure a Tensor Core FFT or reduced-precision encoding.

Supported benchmark geometry: N=65536, one GPU, one Q-only level with 30-bit
primes, 1–64 limbs, 1–64 independent plaintexts, scale_log2=20–60. Rounded
coefficient magnitudes must be below 2^64; unsupported and non-finite inputs
are errors. The setup uses no encryption keys and sec_level_type::none: these
are arithmetic measurements, not a secure application parameter selection.

Build through the existing GPU-enabled Poseidon configuration:

```sh
cmake -S . -B build-runtime-gpu-api-release -DPOSEIDON_BUILD_GPU_ENCODE_BENCH=ON
cmake --build build-runtime-gpu-api-release --target poseidon_gpu_encode_bench -j4
OMP_NUM_THREADS=1 build-runtime-gpu-api-release/bin/poseidon_gpu_encode_bench \
  8 1 40 5 30 report.json
```

Arguments: Q limb count, batch count, scale_log2, warmups, repetitions, report.
The benchmark supports the CUDA path on V100; its Tensor path requires SM 7.5+.
An RMM pool with 64 MiB initial size is the default, matching the runtime's
allocation approach. `POSEIDON_GPU_ENCODE_POOL_MB=0` uses direct allocation for
an explicit allocator-overhead comparison. The runner exposes `--pool-mb`.

`individual` submits one FFT/RNS expansion per plaintext and one NTT per
plaintext. `batched` uses one batched FFT and folds all plaintexts into the
permutation, RNS and NTT grids. Individual preparation for the group precedes
individual NTT submission; both run on the same stream.

CUDA NTT uses the existing two-phase N=65536 four-step algorithm, extended with
a batch grid. Tensor NTT uses fusion=4 and existing integer TAM kernels. Both
write contiguous [batch][limb][N] uint32 residues. cuFFT computes the negative
sign DFT followed by exp(-pi*i*k/N); together they implement the CKKS inverse
embedding in the current slot order. cuFFT is unnormalized, so scale/N is
applied explicitly (see [cuFFT conventions](https://docs.nvidia.com/cuda/cufft/index.html#type-definitions-for-transform-direction)).

Timing separates input-already-on-GPU and pinned raw H2D + Encode. CUDA events
report transfer, FFT/RNS preparation, NTT and total GPU timeline; wall timing
includes submission and final event synchronization. GPU error materialization
is outside the timings. Context/table construction, cuFFT plans and persistent
buffers are setup. Existing Tensor NTT scratch allocation and release remain
inside its measured path; allocation overhead is **not** silently removed.
Median, p95, min/max and samples are saved. CPU comparison measures Encode only,
with a warmed output allocation and prebuilt slot vectors, without upload.

Every canonical GPU plaintext is compared with CPU Encode at residue level.
CPU decode checks the first/last plaintext and every plaintext whose FFT
rounding differs from CPU. Other GPU variants must exactly match the canonical
GPU residues for the whole batch, so their decode validation is reused. Differences
from CPU rounding are reported; FFT operation order is not required to be
bit-identical to CPU. Inputs include negative/zero/larger values and differ
across the batch. GPU non-finite and coefficient-overflow rejection is checked
after measurements. Decoder validation and all output downloads are untimed.

## Optional TileLang NTT experiment

The original source was found at
`../experiments/tile_lang_int32/{kernels.py,tensorcore_ntt.py}` relative to the
repository, with an offline source copy at
`/home/xs/Data/windows-sh/tilelang-int32-source/`. That experiment used one
fixed prime and natural output order. `tilelang_ntt.py` adapts its kernels to
the benchmark's actual RNS primes, multiple plaintexts, and Poseidon's
bit-reversed output. It uses the existing negacyclic roots and fusion=4 TAM
matrices; no additional twist or output reorder is necessary.

Enable explicitly using a Python environment with TileLang **0.1.14**:

```sh
cmake -S . -B build-runtime-gpu-api-release \
  -DPOSEIDON_BUILD_GPU_ENCODE_BENCH=ON \
  -DPOSEIDON_GPU_ENCODE_TILELANG_PYTHON=/path/to/venv/bin/python \
  -DPOSEIDON_GPU_ENCODE_TILELANG_ARCH=89
cmake --build build-runtime-gpu-api-release --target poseidon_gpu_encode_bench -j4
```

The generator emits CUDA and a launcher at build time. Python/TileLang are
not loaded at runtime. The MMA translation unit is separate from RMM to avoid
mixing toolkit and vendored CUDA C++ headers. Disable by setting the Python
cache option to an empty string. Other build configurations remain unchanged.
The benchmark rejects GPUs older than the configured TileLang architecture.

Two extra backends, each with individual/batched submission, are measured:

- `tilelang_cuda_ntt`: eight kernels, each fusing two DIF stages.
- `tilelang_tensor_ntt`: three four-stage TAM kernels and two two-stage CUDA
  tails. Each CTA packs four byte planes in shared memory and performs sixteen
  UINT8 MMA products with INT32 accumulation. All limbs and plaintexts share
  the launch grid. There is no global byte-plane scratch or per-limb host loop.

The modulus is a runtime parameter. UINT64 Barrett reduction uses
`floor(2^64/q)` and multiply-high, with one conditional correction for q<2^31.
Each byte-pair dot product is at most `16*255*255`; the sum of sixteen products
weighted by `256^(a+b) mod q` is below 2^55, so integer reconstruction is exact.
The primes, ratios, byte weights and stage matrices are prepared outside timing.
TileLang uses no temporary allocations during NTT. Existing Poseidon Tensor
NTT retains its internal scratch costs; this is a comparison of complete
implementations, not an isolation of the hardware's arithmetic throughput.

Before timing, both TileLang NTT backends must exactly match CUDA four-step
on every coefficient for inputs including 0, 1, q-1, q-2 and independent random
residues across all limbs and plaintexts. The full Encode checks then require
both individual and batched TileLang modes to match canonical GPU Encode.
`scripts/benchmark_gpu_encode.py` automatically summarizes the enabled backends.
