# GPU Runtime Communication

This directory contains the local GPU transfer primitives used by
`PoseidonGpuApi`:

- `device_buffer.h` describes one device-buffer copy;
- `gpu_object_copy.*` validates and materializes complete single-device
  Poseidon plaintext and ciphertext copies;
- `cuda_local_transfer.*` performs same-device, CUDA peer, or pinned Host-staged
  copies.
- `nccl_mpi_transport.*` (when `POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL=ON`)
  bootstraps a global NCCL clique through an MPI communicator and provides
  grouped asynchronous uint32 device-buffer send/receive operations.

`PoseidonGpuApi` owns RuntimePlan communication actions, logical device mapping,
value validation, and completion ordering. These helpers do not perform
placement, insert transfers, interpret plans, or own values by `ValueId`.

Device outputs are published as soon as their CUDA work has been submitted.
Each output keeps the transfer or NCCL completion event alive. A later compute
or communication operation inserts `cudaStreamWaitEvent` into its own stream;
the CPU does not wait for the payload. Host materialization and the final drain
use the blocking request wait so asynchronous CUDA/NCCL errors are surfaced.

In `PerDeviceWorkers` mode, same-rank D2D copies are submitted by the source
worker, H2D copies by the destination worker, and D2H copies by the source
worker. One dedicated issuer thread per MPI rank submits cross-rank NCCL actions
in RuntimePlan order. A mixed Replicate is split between those two paths while
keeping its transfer id.

V1 copies complete single-device objects. Multi-field and partial-shard values
are rejected. Every movement must come from an explicit RuntimePlan Transfer or
Replicate action.

The distributed GPU path supports complete Device-to-Device objects across MPI
ranks. The receiver allocates the destination directly from the RuntimePlan
ValueDesc, and the field payload moves with NCCL without an MPI object header.
A Host source may also
target a Device on another rank: the source rank uploads the object to its
logical device 0 and uses that device as NCCL staging. Cross-rank Device-to-Host
and Host-to-Host actions remain unsupported.
