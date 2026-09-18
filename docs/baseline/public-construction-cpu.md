# Public Hecate construction: loops, containers and branches

## Status and scope

Confirmed: an opt-in request contract now accepts bounded public construction
syntax, expands it without executing candidate Python, and sends the resulting
straight-line Hecate function through the existing Dacapo/SEAL CPU pipeline.
Three manual positive programs pass real encrypted execution; two intentionally
wrong programs are rejected by the independent numerical comparison.

This is progress toward **all project DSL semantics**, not completion of that
goal. It is not a new paid Agent cohort, a Poseidon GPU result, a formal proof
over all programs/inputs, or a performance claim.

Source motivation: `third_party/dacapo/examples/benchmarks/MLP.py` constructs
weighted rotations using public loops, indexing, and conditional expressions.
`third_party/dacapo/python/poly/poly/MPCB.py` also uses containers and public
loops to construct polynomial expressions. Those entire upstream helpers are
NOT claimed supported by this smaller execution gate.

## Representation boundary

`--public-construction` selects request `hecate-function-synthesis-v7` and AST
contract `hecate-function-v6`. This includes the previous arithmetic expansion.
Old request/AST versions and default selection remain unchanged.

1. The immutable request supplies model semantics, public constants, input and
   output layouts. It contains neither hidden inputs nor reference outputs.
2. `public_construction.normalize` parses the submitted source as data. It binds
   encrypted inputs and named constants to immutable symbolic value IDs.
3. Public integers, local lists/tuples, loops and conditions are interpreted
   with explicit resource bounds. Ciphertext operations emit fresh SSA values.
4. The emitted straight-line source is checked by `hecate-function-v5`, including
   ciphertext types, provisioned rotations and the unchanged packing rules.
5. The sandbox independently repeats validation/normalization, saves
   `normalized-source.py` and `construction.json`, binds public operands as
   Hecate Plain inside the active function, and traces real Hecate objects.
6. The existing compiler/artifact gates and real SEAL runtime execute the result;
   the trusted harness compares decrypted outputs with the frozen reference.

Only the new normalizer module is added to the existing explicit read-only
sandbox module mounts. No workspace/home/network capability is added. Candidate
Python is never passed to `exec` or `eval`. Original candidate source, expanded
source, source hashes, construction counters, Earth/CKKS IR and HEVM/CST survive
as separate evidence. The named-constant manifest is not rewritten.

## Implemented semantics

| Construct | Meaning and boundary |
|---|---|
| `for i in range(...)` | One to three public integer arguments; positive/negative step, empty ranges and nested loops. No encrypted trip counts. |
| `for item in values` | Iterate a local list/tuple. List mutation during iteration follows Python's advancing-index behavior, subject to limits. |
| `if/else`, `a if condition else b` | Choose at graph-construction time using public integer/bool values and integer comparisons; not encrypted conditional execution. |
| Integer `+ - * // %`, unary `+ -` | Public indexing/control calculations only; division by zero and oversized integers fail. |
| Lists/tuples, nested containers, exact-length unpacking | Hold symbolic operands or public construction values, not ciphertext slots. |
| `container[index]`, negative indices, `len(container)` | Public container operations; bounds checked. A named packed weight vector is still an opaque operand, not an indexable local container. |
| `list[index] = value`, `list.append(value)` | Shared list aliases observe writes. Rebinding a list variable leaves other aliases intact. Cyclic containers fail. |
| `x = ...`, `x += ...`, `x -= ...`, `x *= ...` | Cipher operations produce new values. Captured ciphertext aliases retain the old value, distinct from mutable list aliases. |
| `x.rotate(public_step)` | Public expression is evaluated and replaced by a signed literal; only existing provisioned steps are allowed. |
| Final ciphertext/list/tuple return | Must match the immutable output ciphertext count and selectors. A tuple is normalized to an ordered return list. |

Current engineering bounds: 65,536 source bytes, 4,096 input AST nodes,
4,096 evaluator steps, nesting depth 32, 128 entries per container/range,
public integers within ±1,048,576, and 256 emitted ciphertext operations.
These are execution resource limits, not a definition of the complete upstream DSL.

Still pending: helper definitions/calls, closures, comprehensions, public array
element access/arithmetic and derived constants, slicing, augmented list-item
assignment, while/break/continue/early-return/for-else, further Python container
semantics, generalized shape/packing and the full high-level helper library.
Ciphertext-dependent control flow requires its own explicit FHE semantics; it
must not be implemented by reading or decrypting a value during construction.
True bootstrap/upscale runtime gaps remain unchanged.

## Real encrypted validation

From the WSL source root `/mnt/d/Code Space/Poseidon`:

```bash
timeout -k 3s 1600s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_construction_goldens.py
```

Evidence: `/home/lhy/poseidon-work/results/public-construction-goldens-urebgj54/report.json`.
Five programs, four input groups each, 64 output values in total:

| Program | Expected outcome | Maximum absolute error |
|---|---|---:|
| Nested-loop Linear(4,2), public weight selection, append and return list | pass | 4.6424835232637505e-9 |
| Shared list item write, negative index and public branch | pass | 3.382830571219131e-8 |
| Pair unpacking with two independently encrypted inputs | pass | 8.920237526708215e-9 |
| Linear deliberately missing the second reduction rotation | numerical rejection | 0.6249999980813912 |
| Deliberate list copy in place of a shared alias | numerical rejection | 0.5000000153602886 |

The Linear's fixed independent formula is `X @ W.T`, with rows
`[0.5,-0.25,0.125,0.75]` and `[-0.375,0.25,0.5,-0.125]`.
The other independent formulas are `1.5*x+0.375` and `left-right`.
Reference values are recomputed from original inputs, not decoded intermediates.

Security/numerics unchanged: SEAL 4.0.0, N=32768, 14×60-bit modulus, tc128,
existing scale/waterline profile, atol=1e-5 and rtol=1e-4, no bootstrap.
Each correct program compiled, encrypted, evaluated and decrypted in the real
isolated CPU backend. Both wrong programs also executed, then failed only at
the intended numerical comparison gate. Neither is an infrastructure failure.

`test_public_construction` passes 18/18 with no skips when
`POSEIDON_PUBLIC_CONSTRUCTION_REPORT` points to the above report and the tests
run in the existing pinned Nix/Python environment. Its evidence test rechecks
immutable data/artifact hashes, recomputes the normalized source, independently
recomputes formulas and metrics, reparses HEVM/CST, verifies security parameters
and confirms there are no retained private-key directories.

Full offline regression: 467 tests, 361 passed and 106 conditional skips, no
failures. Skipped tests are not counted as passes. No paid API, installation,
download, commit or push occurred. Five temporary private-key directories were
cleaned after their individual runs: 3,394,900,050 bytes (about 3.16 GiB), while
all durable non-key evidence was retained.

## Coverage interpretation

The coverage analyzer reports ciphertext dependencies on the expanded graph.
Construction counters (iterations, branches and writes) are reported separately:
executing a public loop does not prove its result influences the returned value.
A regression checks that a loop computing an unused negation chain contributes
no output-reachable negation coverage. No percentage here uses these finite
partitions as the denominator for all Hecate semantics.
