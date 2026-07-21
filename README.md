See [Poseidon Doc](https://poseidon-hpu.readthedocs.io/en/latest/) for details.

## Bundled CKKS Runtime

Poseidon pins CKKS Runtime as a submodule. Runtime pins the compatible modified
Dacapo compiler as its own nested submodule. After cloning, initialize the full
toolchain with:

```bash
git submodule update --init --recursive third_party/ckks-runtime
```

Dacapo is then available at `third_party/ckks-runtime/third_party/dacapo`. It
remains source-only in the normal Poseidon build and is built separately when
compiler artifacts need to be regenerated.

## Optional CKKS Runtime CPU/GPU Api

The integration is disabled by default. When disabled, Poseidon does not inspect
or build the Runtime submodule.

Configure the Runtime Api with the bundled Runtime:

```bash
cmake -S . -B build-runtime-cpu \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON \
  -DPOSEIDON_BUILD_EXAMPLES=OFF
cmake --build build-runtime-cpu --target poseidon_runtime_cpu_api_tests
ctest --test-dir build-runtime-cpu -R poseidon_runtime_cpu_api_tests --output-on-failure
```

`POSEIDON_CKKS_RUNTIME_SOURCE_DIR` can still point at another Runtime checkout
for local development. CPU MPI and CUDA test paths are controlled by
`POSEIDON_BUILD_CKKS_RUNTIME_MPI` and
`POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS`.
