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

## Waterline plaintext preparation

The late Dacapo option `--runtime-plan-plaintext-schedule=waterline` emits V4
Device Prepare operations after the existing HEFT placement and communication
passes. Supply positive GPU, Host, pinned and raw budgets, a reserve and a
waterline fraction. The runtime admits complete direct-dependency groups in
one topological prefix and reuses its existing per-device executor. Explicit
Fence operations remain boundaries; the compiler inserts no batch Fence.

Prepare supports complete CPU Encode plus upload, complete GPU Encode from
real raw slots, and `auto` queue-based selection among legal candidates. GPU
Encode uses the context degree, level and scale, with explicit cuFFT scratch,
RNS reduction and CUDA NTT. Wide finite FP64 integer coefficients are supported
within the context modulus range. Foreground streams have higher CUDA priority
than the preparation stream, with one active preparation per GPU.

The backend checks the configured reserve against actual parameter/key storage
and bounds for retained workspaces. Evaluation keys share one owning allocation
across active Q levels. `POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB` caps the managed
RMM pool; its cap plus measured outside-pool GPU usage must fit `M`. Active
allocation peaks and retained pool bytes are reported separately. Custom memory
resources used by allocation tests must provide their own physical limits.
Shared Host credits cover runtime objects, raw payloads, encoder temporaries
and pinned staging; process RSS also includes context, Host keys and graph
metadata and is measured separately.

For complete fixture inputs and outputs, build and run:

```bash
cmake --build BUILD --target poseidon_runtime_gpu_qwen_e2e
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v \
  BUILD/bin/poseidon_runtime_gpu_qwen_e2e \
  PLAN OPERATOR_SPEC FIXTURE REPORT DECODED \
  --bundle BUNDLE_DIR --backend auto --trace TRACE
```

The fixture contains explicit Q/P bits, all inputs and reference outputs. The
runner verifies every real and imaginary slot with fixed `atol=0.001` and
`rtol=0.0001`. It sets a pool cap from the per-device capacities and records
CPU encoding wall time and actual CUDA Encode/H2D event durations. Transport
H2D counters include Host cipher and CPU plaintext uploads; GPU raw uploads
are counted separately, and direct setup key uploads are excluded.

V4 Waterline currently supports one rank and ciphertexts with at most three
components. Native Boot resource bounds are not implemented and cause an
explicit error. Host decrypt/re-encrypt Boot uses the secret key and is reported
as a test refresh, with its actual semantics stated in each model result.
Validation status and reproducible evidence are in
[the Waterline results](test-results/waterline-20261013/README.md).

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

For timing an arbitrary multi-rank GPU RuntimePlan (including the MLP and
1999-operation probe plans), build the generic runner:

```bash
cmake --build build-runtime-nccl --target poseidon_gpu_mpi_runtime_e2e
mpiexec --host node0:1,node1:1 \
  build-runtime-nccl/bin/poseidon_gpu_mpi_runtime_e2e \
  PLAN OPERATOR_SPEC BUNDLE_DIR REPORT \
  --rank-to-node 0x1 --warmups 1 --iterations 3
```

Use `BUNDLE_DIR=-` for plans without a plaintext bundle. The runner reports
the maximum online execution time across ranks per iteration; it does not
apply the fanout smoke graph's fixed `10 negate + 9 add_cc` numerical check.

For a single-process run, use `--local` instead of `--rank-to-node`:

```bash
build-runtime-nccl/bin/poseidon_gpu_mpi_runtime_e2e \
  PLAN OPERATOR_SPEC BUNDLE_DIR REPORT \
  --local --warmups 1 --iterations 3
```

Local mode does not call `MPI_Init_thread` and uses the non-MPI GPU API, so it
does not create an NCCL communicator. A one-rank MPI launch also selects the
non-MPI GPU API, although MPI initialization and finalization remain visible.

GPU communication publishes Device outputs after submission, with CUDA events
carried by the values. Consumers wait on those events in their own streams, so
ordinary copy/NCCL dependencies do not block a CPU worker. Cross-rank actions
use one ordered communication issuer thread per MPI rank; same-rank copies stay
on device workers.

For a compact Nsight Systems trace, use `--warmups 0 --iterations 1`; retain
`--warmups 1 --iterations 3` for stable timing measurements. The runner emits
`setup`, `warmup`, and `online.iteration.N` NVTX ranges so these scopes can be
selected directly in the timeline.

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
