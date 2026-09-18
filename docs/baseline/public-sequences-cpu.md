# Public sequences in Hecate construction

## Purpose and upstream evidence

The fixed upstream `third_party/dacapo/python/poly/poly/Poly.py:17-19` slices a
list of coefficient text lines (`coeffStr[0:16]`, `[16:32]`, `[32:60]`). MPCB.py additionally
uses multidimensional tensor slicing, for example line 115. This change supports
the public Python sequence building block, **not** that complete tensor helper,
NumPy/Torch arrays, public floating-point arithmetic or GenPoly.

Opt-in CLI: `--public-sequences`; request `hecate-function-synthesis-v13`, AST
contract `hecate-function-v12`. Earlier versions and saved requests are unchanged.
The new version includes previous construction/function/lambda rules.

## Implemented semantics

- Public list, tuple and string indexing and `start:stop:step` slicing, including
  omitted/None bounds, clipping, negative bounds/steps and public Boolean indices.
  Zero steps and symbolic/private bounds are rejected.
- Same-type sequence concatenation and integer repetition in either operand
  order. Slices, concatenations and repetitions are shallow: a new outer
  sequence still shares contained lists, closures and symbolic operands.
- Named-list `+=` and `*=` mutate in place and preserve aliases. Ordinary `+`
  and `*` create new outer objects. Self-extension by the list snapshots; an
  iterator over that same list observes appended items and reaches the budget.
- List slice assignment evaluates the RHS first, then target and bounds in
  order, then materializes replacement before mutation. Contiguous slices may
  resize; extended slices must preserve the number of selected positions.
- Public string len, iteration, reverse iteration and list/tuple conversion.

Generated Python is never executed. The bounded AST interpreter expands the
construction to the existing checked straight-line Hecate arithmetic program.
Existing 128-item/character container limits, 4096-step expansion budget,
bounded integers, acyclic containers and emitted-operation limits remain.

Named public constants are still opaque, read-only symbolic values. Slicing
`[c0,c1]` selects whole weight operands; slicing `c0` remains rejected. Likewise,
slicing `[x,y]` selects ciphertext objects, not slots inside `x`. This does not
add general shape/broadcast/packing, ndarray slicing, encrypted indexing,
tuple/string mutation or augmented item assignment.

## Tests and reproducible commands

`test_public_sequences.py` compares 216 slice-read boundary combinations and
180 slice-write combinations with native Python list results, including expected
extended-slice length errors. Additional tests cover shallow aliases, direct
self-extension versus iterator extension, evaluation order, resource limits,
cycles, private bounds, strings, immutable sequences and version routing.

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 100s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest test_public_sequences test_function_literals \
  test_public_iteration test_construction_calls test_closure_construction \
  test_function_construction test_public_construction test_extended_arithmetic -q
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_sequence_goldens.py
```

Manual golden roles (each has an intentional numerical counterexample):

1. Linear(4,2): reverse a public weight-row list, then use real cross-slot
   multiply/rotation/add reductions. The wrong program keeps rows reversed.
2. Affine `1.5*x+0.375`: update a shared nested list created by repetition. The
   wrong program creates independent nested lists and loses the shared update.
3. The same affine reference: write through a negative-step slice. The wrong
   program writes through a forward slice and changes the selected coefficient.

The runner first checks an independent plain formula, then uses the existing
isolated tracing/compiler/SEAL HEVM path. Neither mock arithmetic nor the
preflight is counted as encrypted execution. No paid API call, new backend,
bootstrap substitute, dependency installation or tolerance change is involved.

## Completed actual CPU evidence

Report: `/home/lhy/poseidon-work/results/public-sequence-goldens-w084hfaa/report.json`

SHA256: `70b919cc1f362f09ca824d3ed5b7d364e1de1b1239cc2e5b99c5a62d130982ff`

All six roles matched on their first run: **3 positive programs passed and 3
intentional mistakes were rejected at numerical comparison**, after actual
encrypted execution. Total: 24 input groups / 80 compared output values.

| Role | MAE | Maximum absolute error | Outcome |
|---|---:|---:|---|
| Linear row selection by slice | 3.0501920e-9 | 8.0696271e-9 | Pass |
| Shallow repetition with shared nested update | 5.6977305e-9 | 2.9603147e-8 | Pass |
| Negative-step slice assignment | 9.1598619e-9 | 2.7302213e-8 | Pass |
| Wrong Linear row order | 0.23237257 | 0.61699026 | Numerical rejection |
| Independent lists instead of shared repeat | 0.51763045 | 1.00000000 | Numerical rejection |
| Forward instead of reverse slice assignment | 0.25881523 | 0.50000002 | Numerical rejection |

The unchanged profile is SEAL 4.0.0, N=32768, fourteen 60-bit moduli, tc128;
elementwise tolerance is `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
Reports retain individual outputs, relative error and cosine similarity. No
bootstrap or GPU execution is claimed.

The eight-group saved-evidence audit passed **136/136, no skips**, in the pinned
Nix environment. It verifies producer/source/artifact hashes, exact old-version
normalization, original requests/weights/inputs, security settings, actual
execution flags and independently recomputed decrypted numerical comparisons.
Enable this group's audit with `POSEIDON_SEQUENCE_REPORT` set to the report above;
the other groups use their documented evidence environment variables.

Full offline regression: **573 tests, 461 passed, 112 conditional skips, no
failures** (`python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -q`
with `PYTHONPATH=scripts/baseline` and `PYTHONDONTWRITEBYTECODE=1`). Skip conditions
are not counted as passing coverage. Expected argparse rejection messages and
mock-provider retry diagnostics are exercised by negative unit tests; no actual
API was called. The separate 136-test real-evidence audit above has no skips.

Producer `function_construction.py` SHA256:
`147b1e25d47620746cf9e32be158fa8a1978c0d013ac38e5cf0010e602f0e7f0`.

All six key-cleanup outcomes are complete, freeing **4,073,880,060 bytes (about
3.79 GiB)**. Original random keys cannot be recovered, but equivalent fresh test
keys can be regenerated. Source, reports, arrays, IR, HEVM/CST and decrypted
outputs are retained. No unrelated data was removed.

An initial offline command misspelled `test_construction_calls` as
`test_call_binding` and failed module import; corrected invocation passed. This
was not a DSL/compiler/backend failure. No dependencies were installed and no
paid API, commit, push or PR was performed.

## Remaining complete-DSL requirements

This is one construction-semantics addition, not a complete-DSL claim. General
public numeric/array operations, remaining controls and container operations,
general packing and shapes, all upstream helpers, IR function calls, actual
bootstrap/upscale and online Agent coverage of new syntax remain outstanding.
