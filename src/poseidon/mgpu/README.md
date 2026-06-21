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
