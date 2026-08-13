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

RMM also requires these repository-owned wrapper files during configure:

- `third_party/rmm/cmake/thirdparty/get_spdlog.cmake`
- `third_party/rmm/cmake/thirdparty/get_cccl.cmake`
- `third_party/rmm/cmake/thirdparty/get_nvtx.cmake`

Do not exclude the RMM `cmake/` directory when producing an offline archive.
The GPU CMake entry point and `scripts/package_offline_gpu_runtime.sh` both
validate these files before continuing.

`third_party/cpm` only contains the pinned CPM.cmake module. Dependency source
caches such as `third_party/cpm/cccl` are local build artifacts and are not part
of the offline bundle.

RMM `v24.12.01` requires CMake `3.26.4` or newer and a CUDA toolkit compatible
with the RAPIDS 24.12 dependency set.

GMP remains a system dependency. When it is installed outside the default
search paths, configure Poseidon with its prefix, for example:

```sh
cmake -S . -B build -DGMP_ROOT=/root/.local
```

The imported `GMP::gmpxx` target propagates the required GMP include and link
paths to both the C++ and CUDA targets.

Create a timestamped offline GPU Runtime archive with:

```sh
scripts/package_offline_gpu_runtime.sh /path/to/windows-sh
```

The script validates the RMM wrappers and the selected 65536 RuntimePlan assets
before writing the archive, then emits both an embedded `SHA256SUMS` manifest
and the archive checksum.

## Offline Migration Checklist

- Always use `scripts/package_offline_gpu_runtime.sh` instead of manually
  selecting RMM files. RMM's small `cmake/thirdparty/get_*.cmake` wrappers are
  required even though all dependency source trees are already present.
- Verify the embedded file manifest after extracting the archive:

  ```sh
  sha256sum -c SHA256SUMS
  ```

- GMP is not bundled. Install the GMP C and C++ development files on the target
  machine and pass their common prefix with `-DGMP_ROOT=/root/.local` when it
  is outside the system search path. Do not rely on shell-specific `CPATH`,
  `CUDAFLAGS`, or `LIBRARY_PATH` settings.
- Treat the included release build as a record of known-good artifacts, not as
  a portable CMake build tree. `CMakeCache.txt`, response files, RPATH entries,
  and dependency metadata contain absolute paths from the source machine.
  Configure a fresh build directory after extraction.
- Keep `POSEIDON_BUILD_DEPS=OFF` on the isolated machine. The RMM stack is
  vendored, while ZLIB, Zstandard, Microsoft GSL, GMP, CMake, and CUDA remain
  target-machine prerequisites according to the selected build options.
- For multi-GPU execution, use the matching 65536 RuntimePlan, plaintext bundle,
  OperatorSpec, and device count together. A plan generated for another device
  topology is not interchangeable merely because the executable is the same.
