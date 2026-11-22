# Portable BERT build

The maintained build uses the current source tree by default. No SSH account,
private data directory, previous build, GCC version or CUDA installation path is
selected automatically. Builds never execute a benchmark or a GPU probe.

## Prerequisites and scope

- GPU runners: Linux, C++20 compiler, CMake >=3.26.4, a compatible NVIDIA CUDA
  toolkit, GMP C/C++ development files and spdlog/fmt development packages.
- Keep the bundled `third_party/{ckks-runtime,rmm,cccl,nvtx,rapids-cmake,cpm}`
  dependencies available; see `../../../third_party/README.md`. SOURCE builds do
  not require any prior `.so` or `.a` files.
- Python plan generation/tests additionally require Python 3 and NumPy. Hecate
  is optional and only needed for the separate compiler-export path.
- This change removes machine binding; it does not port the CUDA runtime to
  AMD/Metal, or establish Windows/macOS/ARM support for the full Poseidon stack.

## Build from source (recommended)

From the repository root, for an SM80 target:

```sh
cmake -S bench/dsl/bert -B build-bert \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-bert --parallel 2
```

`build-bert/bin/bert_native_runner` includes prepared Linear/QK/AV by default.
Set `-DBERT_BUILD_PREPARED=OFF` for the ordinary RuntimePlan-only runner, or
`-DBERT_BUILD_MULTI_PLAN=ON` to additionally build `bert_multi_plan_runner`.

The equivalent convenience entry is:

```sh
bash bench/dsl/bert/build.sh build -B build-bert -- \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
```

Use standard `CC`, `CXX`, `CUDACXX`, `CUDAHOSTCXX`, `CMAKE_GENERATOR`,
`CMAKE_BUILD_PARALLEL_LEVEL`, `CMAKE_TOOLCHAIN_FILE`, `CUDAToolkit_ROOT`,
`CMAKE_PREFIX_PATH` and `GMP_ROOT` to select the target environment. Arguments
containing spaces must be shell-quoted. Select GPU architectures explicitly;
when neither CMake nor `CUDAARCHS` specifies one, the fallback is SM75, not
local-device detection. The toolkit must support the requested architecture
(e.g. V100/SM70 requires a toolkit that still supports Volta).

`build_a800.sh` and `compact_native/build_a800.sh` are now **local compatibility
wrappers**, not private remote-deployment scripts. They do not connect by SSH,
copy private bundles or download Python wheels. `multi_plan/build-local.sh`
builds from the maintained source rather than a historical frozen snapshot.
The standalone component CMake entry points remain available. `aligned_native/`
is a compatibility entry for the unified prepared-runner build.

## CPU-only checks (no CUDA or Poseidon libraries)

```sh
cmake -S bench/dsl/bert -B build-bert-cpu -DBERT_BUILD_GPU=OFF
cmake --build build-bert-cpu --parallel 2
# Optional, explicit test invocation:
ctest --test-dir build-bert-cpu --output-on-failure
```

This builds independent layout/QK/AV arithmetic checks, not a CPU BERT inference
runner. `-DBERT_BUILD_CPU_METADATA_CHECKS=ON` additionally builds real Poseidon
CPU encoding/encryption checks and therefore requires the CPU dependencies.
Use a fresh build directory when switching GPU/dependency configurations.

## Explicit prebuilt-library mode

For an ABI-compatible build on the target machine:

```sh
cmake -S bench/dsl/bert -B build-bert-prebuilt \
  -DBERT_DEPENDENCY_MODE=PREBUILT \
  -DBERT_PREBUILT_BUILD=/path/to/compatible/build \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-bert-prebuilt --parallel 2
```

AUTO mode also selects PREBUILT when an explicit build/library is supplied.
There is no implicit search in old experiment build directories. Both ordinary
Poseidon output layouts and the older standalone BERT layout are supported.
For split layouts, explicitly set `BERT_CPU_LIBRARY`, `BERT_GPU_LIBRARY`,
`BERT_API_LIBRARY`, `BERT_CORE_LIBRARY`, `BERT_JSON_LIBRARY` and
`BERT_CONFIG_INCLUDE_DIR` (the directory containing `poseidon/util/config.h`).
Set `POSEIDON_ROOT` and `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` to matching headers.
Legacy `COMPACT_*`, `ATTENTION_*`, and `COMPACT_AV_*` library/path overrides
remain supported for standalone components. Imported APIs built with MPI/NCCL
require `-DBERT_PREBUILT_WITH_NCCL_MPI=ON` and matching target dependencies.

SOURCE mode also retains the optional `BERT_PREBUILT_CPU_LIBRARY` shortcut:
only the CPU `.so` is imported; GPU/API/Runtime are rebuilt. Matching the CPU ABI
and build options is the caller's responsibility. Prefer a full SOURCE build
when moving to another machine/toolchain. Old CMake caches, generated build
files and binary ABI compatibility are not portable artifacts.

## Validation status and local artifacts

Before the user stopped further validation, the dependency-free CPU build and
its five CTests passed; the existing Python suite passed; explicit PREBUILT
mode linked both runners and the no-device guard for SM75. The full SOURCE
build was stopped during compilation, not completed. Final packaging/script
changes were not retested at the user's request. No GPU inference was run.

Historical timings in README/result summaries remain experimental and full HE
accuracy is not passed. Large evidence archives, raw plans/weights, private
lifecycle scripts and frozen source copies are deliberately local-only; their
historical filenames/paths in reports are provenance, not fresh-clone inputs.
`test_gpu_admission.py` belongs to that local orchestration harness and is not
part of the distributed algorithm tests. Do not copy old credentials or keys
into the source package.
