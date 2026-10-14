# Ordered PyTorch capture draft

Agent-only staging code in the ARM Ubuntu development checkout. Production source is frozen for the running authorized batch.

pytorch_ordered_capture.prepare_module() reuses the existing registered PyTorch capture and operator decomposition. It tracks tensor views, shared storage and ordered writes; pytorch_ordered_reference.validate() compares original and normalized Torch with independent mathematical references using separate input clones and copied outputs.

Tested operations: Linear, Conv1d/2d, BatchNorm, LayerNorm, GroupNorm, mean, matrix products and concat combined with full writes, partial writes and clone. Additional cases cover split, basic versus advanced indexing, transposed views, reshape copies, contiguous, broadcast copy, alpha and in-place ReLU with an explicit approximation.

The tests also cover rank 32 logical admission, a nonrepeating 10,240-element public weight, original dtype identity, weight/source binding changes, resource rejection, overlap, public-state mutation, dynamic branches, unsupported signatures and default capture compatibility.

## Acceptance

results/agent-alias-capture-tests-r1170: 32 tests passed, 0 failed/errors/skipped. Torch 2.0.1, NumPy 1.25.2 and Nix Python 3.10.14 run in an isolated bubblewrap process, with the existing portable store mounted read-only. No second Nix launcher is started.

- 42 capture contexts: 672 groups and 8,848 values, using the bounded local probe fixture.
- 28 reference contexts: 448 groups and 6,000 values, using the retained trusted 16-probe schedule.
- These sets overlap; they are separate checks, not unique model counts.
- Original Torch versus independent ideal reference, and approximate reference versus decomposed graph, are checked separately.
- The test polynomials' approximation error does not certify application accuracy.
- This is Torch FX and plaintext/reference validation. Hecate tracing, compilation, encrypted execution and new Agent generation remain pending.

Each result has source snapshots, hashes, runtime and sandbox receipts. Prior failures and drafts remain preserved.

## Entry

Run the staging tests with a fresh result name:

    python3 -B scripts/baseline/benchmarks/tools/staged_alias_capture_v1/run_isolated.py agent-alias-capture-tests-rNNNN

The runner uses the existing ARM environment and unchanged resource limits. Integration status and source hashes are in integration.json; the review diff is changes-r1171.diff.

The application and CLI still use the retained input versions. Source-manifest routing, worker/reference provenance and representative Poseidon acceptance are the next integration steps. Public copy mathematics and other untested signatures have no executable capability claim.
