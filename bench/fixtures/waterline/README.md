# Frozen Waterline MLP inputs

These are intentionally frozen benchmark inputs, **not per-run results**. They
were copied byte-for-byte from source commit
`5ef32ba8da97742e4942b99dd6d3a0b735e29f8e` (`5ef32ba8`) so that
`scripts/benchmark_waterline.py generate --kind mlp` does not depend on retaining
`test-results/`. Keep generated suites, reports and logs outside this directory.
The generator's existing spec/plan adaptations apply only to its output copies.

## Source-to-fixture mapping

All destinations below are relative to `bench/fixtures/waterline/`.

| Historical source at `5ef32ba8` | Destination |
| --- | --- |
| `test-results/e2e-new-profile-e63ec18/profiles/operator-spec.json` | `operator-spec.json` |
| `test-results/e2e-new-profile-e63ec18/fixture/input.json` | `input.json` |
| `test-results/e2e-new-profile-e63ec18/fixture/mock-result.json` | `mock-result.json` |
| `test-results/e2e-new-profile-e63ec18/artifacts/1gpu/mlp.optimized.ckks.mlir` | `mlp.optimized.ckks.mlir` |
| `test-results/v100-profile-e63ec18/applied-profile/compiler-profile.json` | `compiler-profile.json` |
| `test-results/memory-v1-review/1gpu/mlp.reuse._hecate_MLP.runtime-plan.json` | `mlp.reuse._hecate_MLP.runtime-plan.json` |
| `test-results/memory-v1-review/1gpu/mlp.reuse._hecate_MLP.bundle/` | `mlp.reuse._hecate_MLP.bundle/` (entire tree) |

The frozen payload comprises **107 files / 4,586,662 bytes**: six standalone
files plus 101 bundle files (one manifest and 100 data blobs). The plan and bundle
basenames must stay paired: the generator derives the `.bundle` path by replacing
`.runtime-plan.json`. The MLIR and compiler profile are retained for the existing
two-device compilation branch, not as evidence that that branch was validated.

`input.json` and `mock-result.json` provide fixed inputs and a historical FP64/mock
reference. Their expected values, `passed` field and tiny FP64 comparison error
are **not an encrypted CKKS accuracy result or acceptance criterion**. Preparing
suites likewise performs no encrypted execution and establishes no such accuracy.

## CPU-only checks

From the repository root, the standard-library regression test checks the frozen
file hashes and every bundle content hash, then generates a single-device suite
in a temporary repository skeleton with no `test-results/` directory:

```sh
python3 bench/fixtures/waterline/test_fixture.py
```

The test stubs the unused NumPy import in the MLP branch and rejects external
compiler execution. To exercise the actual CLI as well (requires NumPy):

```sh
python3 scripts/benchmark_waterline.py generate --kind mlp --devices 1 \
  --backends cpu --output "$(mktemp -d /tmp/poseidon-waterline.XXXXXX)/suite"
```

Both commands write generated suites only under temporary directories. Neither
runs a GPU benchmark, invokes a compiler, or validates the two-device branch.
