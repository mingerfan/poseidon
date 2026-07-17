# GPU Runtime Communication

This directory contains local GPU communication primitives shared by the
Runtime API and the legacy `mgpu` compatibility layer.

- `device_buffer.h` describes one device-buffer copy without schedule or value
  identifiers.
- `gpu_object_copy.*` validates and materializes full single-device Poseidon
  plaintext and ciphertext copies.
- `cuda_local_transfer.*` performs synchronous same-device, CUDA peer, or
  explicitly selected host-staged copies.
- `cuda_topology.*` probes visible CUDA devices and the directional peer-access
  matrix.

These files do not perform placement, insert communication, interpret a plan,
or own values by `ValueId`. The Runtime plan remains the source of all movement.
The synchronous transfer entry point is the compatibility implementation; the
multi-GPU `PoseidonGpuApi` will add CUDA event-backed asynchronous handles on
top of this boundary.
