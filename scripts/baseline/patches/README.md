# Dacapo local patch bundle

The parent repository keeps Dacapo pinned at
`4616402710f39df3e5f5bd7930a6c036025aaac3`.
Agent validation uses the following local fixes in addition to that upstream
commit. They are distributed as patches in the parent repository; no unpublished
submodule commit or Dacapo remote write is required.

Apply in this order to a **clean checkout of the pinned commit**, checking each
patch before applying it:

1. `dacapo-plaintext-subtraction.patch`
2. `dacapo-concat-layout.patch`
3. `dacapo-bootstrap-containers.patch`
4. `dacapo-native-function-calls.patch`
5. `dacapo-refined-function-input-types.patch`
6. `dacapo-augmented-operand-order.patch`

From the parent repository root, inspect `git submodule status third_party/dacapo`
and `git -C third_party/dacapo status --short` first. Do not overwrite a modified
submodule or reapply patches to an already patched checkout.
For each filename above, use `git -C third_party/dacapo apply --check` with its
absolute patch path, then the same command without `--check`. Stop on any failure.
Rebuild affected frontend/compiler binaries in the isolated environment afterwards.
Applying source patches alone does not rebuild or validate those binaries.

The bundle covers four modified upstream files: `tools/frontend.cpp`,
`python/hecate/hecate/expr.py`, `python/poly/poly/MPCB.py`, and
`lib/Dialect/Earth/Transforms/Common.cpp`. The last patch preserves operand order
for augmented arithmetic, particularly `x -= y`; it records an existing local
fix rather than introducing a new evaluator behavior in this publication.

The bootstrap container patch preserves frontend dataflow only. It does not add
real bootstrap support to the Agent's SEAL CPU execution chain.
