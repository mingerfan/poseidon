# Poseidon Multi-GPU Static Schedule Runtime

This directory contains the optional multi-GPU static scheduling work.

V1 scope:

- ciphertext-level parallelism only;
- each compute op uses a full ciphertext/plaintext object on one GPU;
- no multi-shard operator execution;
- static placement and explicit copy insertion before interpretation;
- existing single-GPU GPU operators remain the execution backend.

Subdirectories:

- `ir/`: internal schedule representation and debug text support;
- `compiler/`: Dacapo adapter, placement, copy insertion, verification;
- `runtime/`: device contexts, object store, schedule interpreter;
- `comm/`: object-level GPU communication backends;
- `third_party/dacapo/`: planned Dacapo submodule location.

The Dacapo submodule is source-only in this build. Poseidon does not build
Dacapo or MLIR as part of `poseidon_mgpu`; use the adapter boundary in
`compiler/` to translate captured Dacapo output into the internal IR once the
output format is fixed.

## Dacapo Dependency Isolation

Use Nix for Dacapo/MLIR dependency experiments instead of installing packages
into the system environment:

```bash
nix-shell src/poseidon/mgpu/nix/dacapo-shell.nix
```

The shell points `DACAPO_ROOT` at `src/poseidon/mgpu/third_party/dacapo` and
keeps Dacapo source-only with respect to the normal Poseidon build.

The upstream Dacapo helper functions in `config.sh` currently expect the
compiler binary at `$DACAPO_ROOT/build/bin/hecate-opt`. If you use a different
CMake build directory, either invoke `hecate-opt` directly or provide the same
path before using helpers such as `hc-trace`, `hopts`, and `hbt`.

## ResNet20 Artifact Runbook

Generate real Dacapo ResNet20 artifacts inside the isolated Dacapo shell, not
from the normal Poseidon build:

```bash
nix-shell src/poseidon/mgpu/nix/dacapo-shell.nix
cd "$DACAPO_ROOT"
cmake -S . -B build -G Ninja
cmake --build build -j2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./install.sh
source config.sh
hc-trace ResNet
hbt dacapo 40 ResNet HEAAN GPU
```

Dacapo's ResNet test script loads artifacts using this naming convention:

```text
$DACAPO_ROOT/examples/traced/_hecate_ResNet.cst
$DACAPO_ROOT/examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm
```

Check that both files exist before handing them to Poseidon:

```bash
test -f "$DACAPO_ROOT/examples/traced/_hecate_ResNet.cst"
test -f "$DACAPO_ROOT/examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm"
```

Build the Poseidon CPU-side mgpu diagnostics and external-artifact CTest binary
separately:

```bash
cd "$POSEIDON_ROOT"
cmake -S . -B /tmp/poseidon-mgpu-tools \
  -DPOSEIDON_BUILD_MGPU=ON \
  -DPOSEIDON_BUILD_MGPU_TOOLS=ON \
  -DPOSEIDON_BUILD_MGPU_TESTS=ON
cmake --build /tmp/poseidon-mgpu-tools \
  --target poseidon_mgpu_dacapo_hevm_dump \
           poseidon_mgpu_external_hevm_artifact_tests -j2
```

On the single-node 8-GPU machine, build the optional CUDA communication probe
before attempting execution:

```bash
cmake -S . -B /tmp/poseidon-mgpu-cuda-comm \
  -DPOSEIDON_BUILD_MGPU=ON \
  -DPOSEIDON_BUILD_MGPU_TOOLS=ON \
  -DPOSEIDON_BUILD_MGPU_TESTS=ON \
  -DPOSEIDON_BUILD_MGPU_CUDA_COMM=ON
cmake --build /tmp/poseidon-mgpu-cuda-comm \
  --target poseidon_mgpu_cuda_peer_probe \
           poseidon_mgpu_cuda_peer_probe_tests \
           poseidon_mgpu_cuda_comm_tests -j2
/tmp/poseidon-mgpu-cuda-comm/bin/poseidon_mgpu_cuda_peer_probe \
  --require-devices 8 \
  --require-full-peer-access
ctest --test-dir /tmp/poseidon-mgpu-cuda-comm --output-on-failure \
  -R 'poseidon_mgpu_cuda_peer_probe_tests|poseidon_mgpu_cuda_comm_tests'
```

Then inspect the ResNet20 artifact without executing GPU operators:

```bash
/tmp/poseidon-mgpu-tools/bin/poseidon_mgpu_dacapo_hevm_dump \
  --hevm "$DACAPO_ROOT/examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm" \
  --constants "$DACAPO_ROOT/examples/traced/_hecate_ResNet.cst" \
  --devices 8 \
  --upload-device 0 \
  --compute-devices 0,1,2,3,4,5,6,7 \
  --download-device 0 \
  --opcode-summary \
  --communication-plan \
  --communication-execution-preflight \
  --execution-cuda-peer-available \
  --poseidon-gpu-preflight \
  --preflight-comm-available \
  --preflight-relin-keys \
  --preflight-galois-keys \
  --require-ready \
  --summary-json \
  --no-schedule
```

Use the skipped-by-default external CTest for a repeatable check:

```bash
POSEIDON_MGPU_EXTERNAL_HEVM="$DACAPO_ROOT/examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm" \
POSEIDON_MGPU_EXTERNAL_CST="$DACAPO_ROOT/examples/traced/_hecate_ResNet.cst" \
POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=8 \
POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0 \
POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=0,1,2,3,4,5,6,7 \
POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0 \
POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1 \
POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1 \
POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1 \
POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1 \
POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1 \
ctest --test-dir /tmp/poseidon-mgpu-tools --output-on-failure \
  -R '^poseidon_mgpu_external_hevm_artifact_tests$'
```

Interpret the diagnostics conservatively:

- Unsupported HEVM opcodes, especially `ModswitchC` or `UpscaleC`, mean the
  Dacapo adapter must be extended only after the Poseidon GPU scale/modulus
  semantics are verified.
- A `BootstrapFallback` preflight failure means the artifact can be loaded and
  scheduled, but the current Poseidon GPU execution path cannot run it end to
  end until a fallback or native bootstrap path is designed.
- Missing relinearization or Galois keys must be fixed by key provisioning, not
  by changing placement in the interpreter.
- `communication_plan` entries marked `inter_node` are diagnostic only today.
  Single-node 8-GPU execution should use CUDA peer communication first; cluster
  transport is a later backend behind the same mgpu communication interface.
- `--require-ready` and `POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1` turn these
  CPU-side diagnostics into a hard gate. Use them before execution attempts; do
  not use a passing readiness gate as a substitute for a real GPU correctness run.

## External HEVM Artifact Check

`poseidon_mgpu_external_hevm_artifact_tests` is a skipped-by-default CTest for
real Dacapo `.hevm + .cst` output, including ResNet20 artifacts. It loads the
files, runs static placement and copy insertion, builds the HEVM I/O plan, and
prints schedule/device summaries plus a Poseidon GPU executor preflight. It
does not execute GPU operators.

```bash
POSEIDON_MGPU_EXTERNAL_HEVM=/path/to/model.hevm \
POSEIDON_MGPU_EXTERNAL_CST=/path/to/model.cst \
POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=8 \
POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0 \
POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=0,1,2,3,4,5,6,7 \
POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0 \
ctest --test-dir /tmp/poseidon-mgpu-json --output-on-failure \
  -R '^poseidon_mgpu_external_hevm_artifact_tests$'
```

Optional environment variables:

- `POSEIDON_MGPU_EXTERNAL_DEFAULT_DEVICE`
- `POSEIDON_MGPU_EXTERNAL_ROUND_ROBIN_COMPUTE=1`
- `POSEIDON_MGPU_EXTERNAL_DEBUG_DUMP=1`
- `POSEIDON_MGPU_EXTERNAL_RICH_MOCK_ARTIFACT=1`
- `POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1`
- `POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1`
- `POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1`
- `POSEIDON_MGPU_EXTERNAL_NODES=4`
- `POSEIDON_MGPU_EXTERNAL_DEVICES_PER_NODE=8`
- `POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1`
- `POSEIDON_MGPU_EXTERNAL_EXECUTION_INTER_NODE_AVAILABLE=1`
- `POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1`

`poseidon_mgpu_external_hevm_mock_artifact_tests` exercises the same CTest
binary with a generated mock `.hevm + .cst` artifact, preflight availability
flags, and a 2x2 topology. It is intended to keep the artifact diagnostics path
covered on machines that do not have a real ResNet20 artifact yet.

`poseidon_mgpu_external_hevm_rich_mock_artifact_tests` uses a richer generated
artifact with rotate, ciphertext add/multiply, rescale, plaintext add, explicit
copy insertion, GaloisKeys preflight, and readiness gating. It still avoids
unsupported `ModswitchC` and `UpscaleC`; those remain real adapter work once
their Poseidon GPU semantics are verified.

The dump tool can run the same preflight without CTest:

```bash
poseidon_mgpu_dacapo_hevm_dump \
  --hevm /path/to/model.hevm \
  --constants /path/to/model.cst \
  --devices 8 \
  --compute-devices 0,1,2,3,4,5,6,7 \
  --opcode-summary \
  --communication-plan \
  --communication-execution-preflight \
  --execution-cuda-peer-available \
  --poseidon-gpu-preflight \
  --preflight-comm-available \
  --preflight-relin-keys \
  --preflight-galois-keys \
  --require-ready \
  --summary-json \
  --no-schedule
```

Preflight is CPU-only. It reports whether the schedule needs the mgpu
communication layer, relinearization keys, Galois keys, or unsupported
Poseidon GPU operations such as bootstrap fallback before attempting execution.
Without `--require-ready`, the dump tool prints diagnostics and exits
successfully if loading and schedule construction succeeded. With
`--require-ready`, unsupported HEVM opcodes, Poseidon GPU preflight failures,
communication plan failures, or unavailable communication execution routes
produce a non-zero exit code.

## Communication Topology Planning

`comm/topology.*` is a CPU-only planning layer for copy ops. It models logical
devices as `(node_id, local_device)` pairs and classifies scheduled copies as:

- `same_device`
- `cuda_peer` for copies inside one node
- `inter_node` for copies across nodes

This does not add NCCL or MPI. It fixes the V1 planning interface for the
single-node 8-GPU path and the later 4x8 cluster path while keeping execution
owned by the existing mgpu communication layer.

The Dacapo HEVM dump tool can include this plan with `--communication-plan`.
For a 4x8 logical cluster preview, pass `--nodes 4 --devices-per-node 8`.
Use `--opcode-summary` to print the raw HEVM opcode distribution before
execution planning; this is useful when a real artifact contains unsupported
opcodes such as `ModswitchC` or `UpscaleC`.

Add `--communication-execution-preflight` to distinguish planned routes from
routes executable by the currently available backend. For the single-node path,
also pass `--execution-cuda-peer-available`. Do not pass
`--execution-inter-node-available` until a real cluster transport backend has
been implemented and wired behind the mgpu communication layer.

## CUDA Peer Probe

`POSEIDON_BUILD_MGPU_CUDA_COMM=ON` builds the optional CUDA peer-copy backend
and, when tools are enabled, `poseidon_mgpu_cuda_peer_probe`. This target is not
part of the default mgpu build.

Use it on the 8-GPU node before running a scheduled artifact:

```bash
poseidon_mgpu_cuda_peer_probe \
  --summary-json \
  --require-devices 8 \
  --require-full-peer-access
```

The probe prints visible CUDA devices, basic device properties, and the
destination-by-source peer-access matrix. `--require-devices 8` fails when fewer
than eight devices are visible. `--require-full-peer-access` fails when any
required off-diagonal peer route is unavailable; in that case the CUDA comm
backend can still host-stage copies, but the run should not be treated as a
clean CUDA P2P validation.
