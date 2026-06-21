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

The dump tool can run the same preflight without CTest:

```bash
poseidon_mgpu_dacapo_hevm_dump \
  --hevm /path/to/model.hevm \
  --constants /path/to/model.cst \
  --devices 8 \
  --compute-devices 0,1,2,3,4,5,6,7 \
  --poseidon-gpu-preflight \
  --preflight-comm-available \
  --preflight-relin-keys \
  --preflight-galois-keys \
  --summary-json \
  --no-schedule
```

Preflight is CPU-only. It reports whether the schedule needs the mgpu
communication layer, relinearization keys, Galois keys, or unsupported
Poseidon GPU operations such as bootstrap fallback before attempting execution.

## Communication Topology Planning

`comm/topology.*` is a CPU-only planning layer for copy ops. It models logical
devices as `(node_id, local_device)` pairs and classifies scheduled copies as:

- `same_device`
- `cuda_peer` for copies inside one node
- `inter_node` for copies across nodes

This does not add NCCL or MPI. It fixes the V1 planning interface for the
single-node 8-GPU path and the later 4x8 cluster path while keeping execution
owned by the existing mgpu communication layer.
