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

V1 copies complete single-device objects. Multi-field and partial-shard values
are rejected. Every movement must come from an explicit RuntimePlan Transfer or
Replicate action.

The first distributed GPU path supports complete Device-to-Device objects across
MPI ranks. It exchanges a compact object header over MPI, allocates the
destination object locally, and moves the field payload with NCCL. Host values
whose source and destination ranks differ remain outside this initial backend.
