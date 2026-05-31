# Bundled Third-Party Dependencies

This directory vendors the GPU test dependencies that are required for offline
builds of `src/poseidon/tests/gpu_data_struct`.

The bundled RMM stack is:

- RMM `v24.12.01`
- rapids-cmake `branch-24.12`
- CPM.cmake `v0.40.0`
- CCCL commit `e21d607157218540cd7c45461213fb96adf720b7`
- fmt `11.0.2`
- spdlog `v1.14.1`
- NVTX `v3.1.0`

The local RMM CMake files are patched to use these sibling directories instead
of downloading RAPIDS/CPM dependencies at configure time. Keep the directory
names unchanged unless the corresponding paths in `third_party/rmm` and
`src/poseidon/tests/gpu_data_struct/CMakeLists.txt` are updated together.

RMM `v24.12.01` requires CMake `3.26.4` or newer and a CUDA toolkit compatible
with the RAPIDS 24.12 dependency set.
