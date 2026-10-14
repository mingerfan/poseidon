# Public Tensor.gather draft

Confirmed code gap: trusted capture registers index_select, but not torch.gather/Tensor.gather.
The draft normalizes static public int64 indices to existing reshape/public_gather/reshape operations.
It preserves per-coordinate indices, repetitions and partial nonindexed dimensions.
Production, DSL, backend and model files are unchanged while r1570 is frozen.

Next gates: pinned Torch differential and core decomposition; v37 capture/provenance integration;
bounded Poseidon and current Agent acceptance in three distinct contexts.
Scalar/empty/secret indices and dynamic axes remain explicit unsupported cases.

Pinned draft checks: 27 passed, 0 failed/skipped. Six math/Torch/core contexts pass.
No production capture, frontend compilation, encrypted execution or real Agent claim.
