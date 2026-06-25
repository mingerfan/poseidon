# Multi-GPU Communication Layer Guide

This directory owns all multi-GPU data movement. Compute code should not insert
ad hoc CUDA copies; scheduled cross-device movement must enter through this
communication layer.

## Current Structure

`gpu_comm.h/.cpp` is the main execution interface. It contains:

- `GpuComm`: object-level copy API used by runtime copy dispatch.
- `SameDeviceGpuComm`: trivial same-device implementation for tests and
  single-device fallback paths.
- `GpuObjectCopyRequest`: materialized full-object buffer-copy request.
- `GpuObjectCopyMaterializer`: converts opaque runtime object handles into
  destination objects plus full-object buffer copies.
- `GpuObjectCopyBackend`: local buffer-copy backend interface.
- `MaterializedGpuComm`: logical copy to materialized local object-copy bridge.
- `PlannedMaterializedGpuComm`: static-plan-aware copy executor. It checks the
  precomputed route, materializes the object copy, validates topology, and
  dispatches local or inter-node backends.
- `InterNodeTransportBackend`: future cluster transport boundary.
- `MissingInterNodeTransportBackend`: default backend that fails clearly when an
  inter-node route is planned but no real backend is configured.

`topology.h/.cpp` is the CPU-only planner. It validates logical topology,
classifies scheduled copy ops as `same_device`, `cuda_peer`, or `inter_node`,
and emits plan dumps/JSON.

`execution_preflight.h/.cpp` checks whether a valid communication plan can run
with the currently declared backend set. It reports missing same-device,
CUDA-peer, or inter-node support before schedule execution starts.

`cuda_peer_comm.h/.cpp` is the optional CUDA local backend. It implements
same-device device-to-device copies, CUDA peer copies, and host-staged fallback.
It is built only when `POSEIDON_BUILD_MGPU_CUDA_COMM=ON`.

`cuda_peer_probe.h/.cpp` is the optional CUDA diagnostic path for visible device
and peer-access reporting. It is also built only under
`POSEIDON_BUILD_MGPU_CUDA_COMM=ON`.

`gpu_object_materializer.h/.cpp` is the optional CUDA/RMM-gated bridge for
Poseidon GPU ciphertext/plaintext objects. It is built only when
`POSEIDON_BUILD_MGPU_GPU_OBJECTS=ON`.

`nccl_comm.h/.cpp` is the optional NCCL communication wrapper used by the
multi-GPU transfer benchmark. It is built only when
`POSEIDON_BUILD_MGPU_NCCL_COMM=ON` and is not wired into the mgpu runtime path.

## Execution Flow

The normal planned execution path is:

1. Placement and copy insertion produce explicit `CopyPlain` or `CopyCipher`
   operations in the static schedule.
2. `plan_schedule_communication` reads those copy ops plus `MgpuTopology` and
   produces an `MgpuCommunicationPlan`.
3. `preflight_communication_execution` checks that the declared backend set can
   execute all planned routes.
4. `CopyDispatchingExecutionBackend` handles each copy op and calls `GpuComm`.
5. `PlannedMaterializedGpuComm` looks up the planned route by
   `(source_id, destination_id)`, validates the runtime request, materializes the
   copy buffers, validates the route against topology, then dispatches:
   - `same_device` and `cuda_peer` routes to `GpuObjectCopyBackend`.
   - `inter_node` routes to `InterNodeTransportBackend`.

This keeps device placement and transport choice tied to the static schedule,
not to evaluator or kernel code.

## Build Boundaries

The default `poseidon_mgpu` target is CPU-only and must not require CUDA or RMM.
It includes only the interfaces, planning, preflight, and CPU-side dispatch
logic.

CUDA peer communication is optional:

```bash
-DPOSEIDON_BUILD_MGPU_CUDA_COMM=ON
```

Poseidon GPU object materialization is optional and RMM-gated:

```bash
-DPOSEIDON_BUILD_MGPU_GPU_OBJECTS=ON
```

The Poseidon GPU executor is separate from communication:

```bash
-DPOSEIDON_BUILD_MGPU_GPU_RUNTIME=ON
```

NCCL communication for benchmarks is optional:

```bash
-DPOSEIDON_BUILD_MGPU_NCCL_COMM=ON
```

Do not move CUDA runtime, NCCL, RMM, or existing single-GPU evaluator
dependencies into the default communication target.

## V1 Invariants

- V1 copies full single-device objects only.
- Materializers must reject multi-field or partial-shard objects.
- Copy ops must be explicit in the schedule; runtime communication must not
  invent missing copies.
- The interpreter executes the static schedule and must not make placement
  decisions.
- Inter-node support is only an interface and diagnostic boundary until a real
  NCCL/MPI-style backend is wired in.

## Common Tests

Run the focused communication tests after editing this directory:

```bash
cmake --build build-codex-mgpu --target \
  poseidon_mgpu_comm_tests \
  poseidon_mgpu_object_copy_tests \
  poseidon_mgpu_materialized_comm_tests \
  poseidon_mgpu_planned_materialized_comm_tests \
  poseidon_mgpu_topology_tests \
  poseidon_mgpu_communication_execution_preflight_tests \
  poseidon_mgpu_planned_communication_executor_tests -j2

ctest --test-dir build-codex-mgpu \
  -R 'poseidon_mgpu_(comm|object_copy|materialized_comm|planned_materialized_comm|topology|communication_execution_preflight|planned_communication_executor)_tests' \
  --output-on-failure
```

When touching public mgpu headers or CMake wiring, also run the broader mgpu
test group:

```bash
cmake --build build-codex-mgpu -j2
ctest --test-dir build-codex-mgpu -R 'poseidon_mgpu_' --output-on-failure
```
