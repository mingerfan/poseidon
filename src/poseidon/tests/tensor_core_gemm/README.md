# Poseidon Tensor Core GEMM Tests

This folder builds and verifies the reusable Tensor Core GEMM API added under
`src/poseidon/gpu/gpu_tensor_core_gemm.h`.

## API Layout

All GEMM inputs are device pointers.

- `A` is row-major, shape `M x K`.
- `B` is column-major, stored as `N x K`.
- `C` is row-major, shape `M x N`.

Main launchers:

- `poseidon::gpu::launch_tensor_core_u8_gemm`: u8 inputs, int32 output.
- `poseidon::gpu::launch_tensor_core_s8_gemm`: s8 inputs, int32 output.
- `poseidon::gpu::launch_tensor_core_u32_low32_gemm`: uint32 inputs, uint32
  output with `mod 2^32` semantics.
- `poseidon::gpu::launch_tensor_core_u32_low32_gemm_from_segments`: same u32
  computation, but reuses pre-split byte segments so benchmark loops do not
  include segmentation.
- `poseidon::gpu::launch_cuda_core_u32_gemm` and
  `poseidon::gpu::launch_cuda_core_s8_gemm`: non-Tensor-Core reference/fallback
  kernels.

The Tensor Core paths require SM 7.5 or newer and `M`, `N`, `K` divisible by 16.
For u8/u32 Tensor Core partial GEMMs, `K <= 33025` is enforced so int32 WMMA
accumulators do not overflow.

## Build And Run

```bash
cmake -S src/poseidon/tests/tensor_core_gemm -B src/poseidon/tests/tensor_core_gemm/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/poseidon/tests/tensor_core_gemm/build -j
ctest --test-dir src/poseidon/tests/tensor_core_gemm/build --output-on-failure
```

Run the benchmark directly:

```bash
./src/poseidon/tests/tensor_core_gemm/build/tensor_core_gemm_bench --size 256 --mode all --warmup 3 --repeat 10
```

If CMake cannot infer a useful CUDA architecture, pass one explicitly:

```bash
cmake -S src/poseidon/tests/tensor_core_gemm -B src/poseidon/tests/tensor_core_gemm/build -DCMAKE_CUDA_ARCHITECTURES=75
```
