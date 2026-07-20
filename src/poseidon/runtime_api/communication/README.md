# GPU Runtime Communication

This directory contains the local GPU transfer primitives used by
`PoseidonGpuApi`:

- `device_buffer.h` describes one device-buffer copy;
- `gpu_object_copy.*` validates and materializes complete single-device
  Poseidon plaintext and ciphertext copies;
- `cuda_local_transfer.*` performs same-device, CUDA peer, or pinned Host-staged
  copies.

`PoseidonGpuApi` owns RuntimePlan communication actions, logical device mapping,
value validation, and completion ordering. These helpers do not perform
placement, insert transfers, interpret plans, or own values by `ValueId`.

V1 copies complete single-device objects. Multi-field and partial-shard values
are rejected. Every movement must come from an explicit RuntimePlan Transfer or
Replicate action.
