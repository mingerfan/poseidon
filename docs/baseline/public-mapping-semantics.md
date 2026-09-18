# Public dictionaries and actual upstream shape helpers

Current integration update: public mappings are now wired through formal request
v19 / AST v18, the CLI, sandbox and tracing. A manual golden and the targeted
paid Agent cohort passed real Dacapo/SEAL CPU execution. The frozen cohort covers
30 construction exercises and 117 observation items, not every mapping edge case
or all upstream DSL semantics. See [targeted Agent evidence](construction-agent-coverage.md)
for the 55 API calls, retained failures, per-item matrix and exact scope.

Historical status on 2026-09-12: implemented behind the explicit experimental
`normalize(..., public_mappings=True)` option, construction metadata schema 13.
It includes the preceding object-array semantics without changing request v18
or any earlier contract. Formal request/CLI wiring and new Agent/FHE cases are
still pending. This milestone is public construction evidence, not encrypted
execution evidence.

## Why this is needed by the real project

`third_party/dacapo/python/poly/poly/MPCB.py` uses shape dictionaries to compute
packing-related dimensions before constructing ciphertext operations.
`CascadeDS` and `CascadePool` both start with `shapes.copy()`, modify the copy,
then call `InferShapes`. The preceding checker only accepted object-array copy
and therefore rejected this actual upstream path.

The implementation is in `scripts/baseline/function_construction.py`:
public AST expressions produce bounded dicts and view handles; checked mutation
changes construction data, not ciphertext slots. Public branch/loop expansion
then produces ordinary flat Hecate arithmetic for the existing compiler.
The shape dictionary is not automatically encoded or emitted as runtime IR.

## Newly implemented paths

- `dict()` and `dict(source, **keywords)`; source dicts or public iterables of
  two-element entries, bounded to the existing construction resource limits.
- `copy`, `get`, `setdefault`, `pop`, `popitem`, `clear`, `update`.
- `keys`, `values`, `items` as live views, not detached list snapshots.
- View length/truth, forward/reverse iteration and public membership.
- Python-compatible insertion order, duplicate-key replacement and LIFO popitem.
- Shallow copy preserves nested references while separating the outer mapping.
- Receiver/arguments are evaluated once in source order. Defaults are evaluated
  even when an existing key is selected; unused defaults are not inserted.
- Size-changing mutation invalidates an existing iterator; value replacement is
  observed by a live values iterator.
- Cycles through dict views are detected before insertion, including a dict
  storing its own values/items view. Arbitrary attributes remain inaccessible.

Methods other than update use positional-only arguments. Update supports a
single positional source plus string-keyed keywords; keyword unpacking remains
checked. Object-array copy continues working under the new option.

One differential test caught and fixed a real implementation discrepancy:
`(1, 2) in {1: 2}.items()` is true, but `[1, 2] in {1: 2}.items()` is false.
An items view does not accept a list as a tuple-like membership item.

Still excluded: view set algebra/equality, fromkeys, dict union operators,
arbitrary mapping classes, secret keys/comparisons and cyclic containers.
These exclusions remain work toward the full goal, not evidence of completion.

## Upstream helper verification

Tests extract **unchanged ASTs** of `cint`, `fint`, `InferShapes`, `CascadeDS`
and `CascadePool` from the actual dependency. They compare:

1. Native execution of those trusted upstream definitions.
2. An independent reference using integer ceil-division and bit-length-based
   powers of two, rather than copying the upstream NumPy/log2 formula.
3. The candidate AST interpreter evaluating each expected dictionary field.

36 parameter configurations are checked for each of the 3 shape functions:
**108 comparisons**. nt spans 4/16/64, spatial dimensions 2/4, ki 1/2 and output
channels 1/3/8. These are public metadata configurations, not new CKKS security
parameters or demonstrated executable packing layouts. DS stride changes and
the repetition-factor clamp are covered. Cascade functions must not modify the
original dict; InferShapes intentionally fills fields in its argument.

The full shapeClosure, Torch/einops weight construction and arbitrary layouts
remain unverified. Correct shape arithmetic alone does not prove a full Conv or
packing algorithm correct.

## Test results and reproduction

- New fixed-environment checks: **10/10 passed**, no skips, NumPy 1.25.2 asserted.
- New/old semantic and existing encrypted-artifact audits: **239/239 passed**,
  no skips. The historical FHE evidence is re-audited, not rerun or extended.
- Full system-Python offline suite: **659 total; 500 passed, 159 conditional
  skips; no failures**. New NumPy tests ran separately in the fixed environment.
- Paid API calls: **0**. New FHE executions: **0**. No installation, key generation,
  security-parameter change, GPU work or remote write.

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 160s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_mapping_checks.py
```

Report: `/home/lhy/poseidon-work/results/public-mapping-checks-ytokeimh/report.json`.
It records actual source hashes, unchanged-source check, test counts and explicit
construction-only / no-Agent-contract / no-FHE markers. The checked implementation
hash is `23b4a20fc3f5ca537f396277cd256ff31c5f6c4fb955e8d5b05bed71d122af0a`.

Next gates: formal versioned request/sandbox/CLI wiring; manual positive and
negative programs exercising these helpers through the established FHE path;
separate Agent generation/repair measurements. All remaining DSL and backend
requirements stay open.
