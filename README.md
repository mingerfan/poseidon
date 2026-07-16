See [Poseidon Doc](https://poseidon-hpu.readthedocs.io/en/latest/) for details.

## Optional CKKS Runtime CPU Api

The integration is disabled by default. When disabled, Poseidon does not inspect
or build a CKKS Runtime source tree.

Configure the first single-process Host-only CPU Api with an explicit Runtime
source directory:

```bash
cmake -S . -B build-runtime-cpu \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON \
  -DPOSEIDON_CKKS_RUNTIME_SOURCE_DIR=/path/to/ckks-runtime \
  -DPOSEIDON_BUILD_EXAMPLES=OFF
cmake --build build-runtime-cpu --target poseidon_runtime_cpu_api_tests
ctest --test-dir build-runtime-cpu -R poseidon_runtime_cpu_api_tests --output-on-failure
```

This first Api supports one process, Host values, synchronous Encode/compute,
and no communication backend. Multi-process CPU and GPU work is tracked in
[`docs/ckks-runtime-api-next-steps.md`](docs/ckks-runtime-api-next-steps.md).
