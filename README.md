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
`POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS`. The experimental MPI/NCCL GPU backend
is enabled separately with `POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL`; it requires
an MPI implementation, `nccl.h`, and `libnccl`.

The backend uses MPI for process control and NCCL bootstrap, while GPU payloads
use NCCL. `Place.rank` remains the MPI rank and `Place.index` is the local GPU
index. A raw transport smoke test can be built without running the ordinary
GPU API tests:

```bash
cmake -S . -B build-runtime-nccl \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=OFF
cmake --build build-runtime-nccl --target poseidon_nccl_mpi_smoke
mpiexec -n 4 build-runtime-nccl/bin/poseidon_nccl_mpi_smoke \
  --device-counts 1x1x1x1
```

Generate and run the small compiler-to-runtime MPI GPU smoke plan with one MPI
process managing one GPU on the first node and one process managing four GPUs
on the second node:

```bash
python3 scripts/generate_mpi_gpu_smoke_plan.py \
  --output-dir build-runtime-nccl/mpi-gpu-smoke
cmake --build build-runtime-nccl --target poseidon_gpu_mpi_plan_e2e
mpiexec --host node0:1,node1:1 \
  build-runtime-nccl/bin/poseidon_gpu_mpi_plan_e2e \
  build-runtime-nccl/mpi-gpu-smoke/mpi-gpu-smoke.mpi_gpu_fanout.runtime-plan.json \
  build-runtime-nccl/mpi-gpu-smoke/operator-spec.json \
  build-runtime-nccl/mpi-gpu-smoke/report.json \
  --rank-to-node 0x1
```

The generator checks that Dacapo placement uses all five GPUs, exercises an
ordered V2 communication rule, emits cross-rank Device transfers, and does not
request an unsupported remote transfer direction.

## Nsight Systems GPU Kernel Gap Analysis

After collecting one or more Nsight Systems reports, compare actual GPU kernel
execution, per-stream idle gaps, and end-to-end compute wall time with:

```bash
python3 scripts/analyze_nsys_kernel_gaps.py \
  1GPU=/path/probe_1gpu.nsys-rep \
  4GPU=/path/probe_4gpu_4w.nsys-rep \
  8GPU=/path/8gpu-probe-strong.nsys-rep \
  --output-dir nsys-kernel-analysis
```

The tool invokes `nsys export` as needed and writes a Markdown report, combined
summary CSV files, and one per-kernel interval CSV for each input report. Run
`python3 scripts/analyze_nsys_kernel_gaps.py --help` for anchor overrides, gap
filtering, SQLite input, and overwrite options.
