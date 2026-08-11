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

The local RMM CMake files set explicit `CPM_<package>_SOURCE` overrides for
these sibling directories and disable FetchContent downloads while configuring
RMM. Keep the directory names unchanged unless the corresponding paths in
`third_party/rmm/rapids_config.cmake` are updated at the same time.

`third_party/cpm` only contains the pinned CPM.cmake module. Dependency source
caches such as `third_party/cpm/cccl` are local build artifacts and are not part
of the offline bundle.

RMM `v24.12.01` requires CMake `3.26.4` or newer and a CUDA toolkit compatible
with the RAPIDS 24.12 dependency set.
