# Scoped comprehensions and lazy public iteration

## Why this belongs to the actual DSL work

Pinned Dacapo `python/poly/poly/MPCB.py` uses nested list comprehensions when
building polynomial trees, an enumerate-based filtered comprehension for
`calc_order`, and zip/reversed comprehensions over baby-step polynomials.
These are Python **graph-construction** semantics, not encrypted control flow.
Supporting their scope and evaluation behavior removes real frontend gaps.
It does not by itself support all of GenPoly: string parsing, NumPy/Chebyshev,
floating-point public expressions, integer-key mappings, lambda/sorted,
general packing and runtime operations remain separate work.

`run_candidate.py --public-iteration` selects request
`hecate-function-synthesis-v11` / AST `hecate-function-v10`. It includes earlier
function/closure/parameter features without changing old versions or defaults.
The normalizer interprets only checked AST and produces a straight-line Hecate
function for the existing compiler and SEAL CPU executor. No generated Python
function or generator expression is executed.

## Implemented semantics

| Feature | Behavior |
|---|---|
| List/dict comprehensions | Nested public for clauses and short-circuiting public filters; at most 128 result entries. |
| Outer iterable | Evaluated once in the enclosing frame before entering the comprehension scope. |
| Comprehension targets | Separate implicit lexical locals, not assignments to enclosing inputs/variables. |
| Nested comprehension | Creates another independent scope; later clauses see earlier target bindings. |
| Dict comprehension | String keys evaluate before values; later duplicate keys replace values without changing insertion order. |
| enumerate | Lazy iterator with optional bounded public integer start. |
| zip | Lazy iterator, shortest input wins; inputs advance left-to-right, including shared-iterator exhaustion effects. |
| reversed | Lazy reverse traversal of permitted list/tuple/dict, without mutating the original. |
| iter/next | Iterator aliases share consumption; next's default expression is evaluated even if not needed, as with ordinary arguments. |
| list/tuple | Bounded materialization from permitted public iterables. |
| Public for/unpack/star calls | Consume permitted public iterators, retaining alias and mutation behavior. |
| Dict mutation while iterating | Invalidated iterator produces a clear construction error, not a guessed snapshot. |

Internal iterator objects wrap only trusted builtins over type-checked local
containers. No candidate-defined iteration method is invoked. Every advance
consumes the existing construction step budget, even if all elements are later
filtered out. Container-size, AST/source-size, call/depth and 256-cipher-operation
limits remain. Ciphertexts and named opaque scalar/vector constants cannot be
iterated, indexed as slots or used as filter decisions.

New builtin names are reserved. Calls to these builtins are positional only.
Generator expressions, yield, async iteration, sets, general strings, dict
methods, arbitrary iterables, while/break/continue and richer public numerical
array computation are not claimed here. Those gaps remain in the full goal.

## Tests and real encrypted fixtures

`test_public_iteration` includes 19 semantic/security tests, one independent
plain-formula preflight and one optional real-artifact audit. Cases include
outer-name capture, nested scopes, non-leaking targets, filter order, dict
key-before-value evaluation, lazy list mutation, shared iterator consumption,
zip exhausting a shared input, reverse traversal and resource-limit rejection.
These symbolic tests are not counted as FHE runs or Agent calls.

Run the six manual positive/negative fixtures:

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_iteration_goldens.py
```

The fixtures use existing immutable model/reference data:

- Linear(4,2): two row dot products, real rotation reduction, enumerate/reversed
  and a filtered output comprehension; negative misses one rotation.
- `1.5*x+0.375` through lazy zip: mutate a list after zip creation, consume it
  in a comprehension; negative materializes zip before the mutation.
- The same affine reference through a dict comprehension: duplicate key keeps
  the final scaled value; negative reverses the entries and keeps the wrong value.

This preserves all numerical/security gates and contains no bootstrap or GPU
execution. Passing manual fixtures does not establish online Agent generation
success for this new grammar. A full DSL claim remains unproven.

## Completed real evidence

Report: `/home/lhy/poseidon-work/results/public-iteration-goldens-w8n08sbg/report.json`

SHA256: `71b528620052db1bc1b70f0db9821977b6f6aee75e90e46d3c52af9b77ad787c`

All six roles matched expectations on the first run: **3 positive programs and
3 numerical counterexamples**, 24 input groups and 80 compared output values.
All counterexamples parsed, traced, compiled and completed encrypted execution
before failing only at numerical_comparison. No rerun or threshold adjustment.

| Manual program | Expected | MAE | Maximum absolute error |
|---|---|---:|---:|
| Linear with output comprehension and enumerate/reversed reduction | Pass | 2.3590053e-9 | 6.2637556e-9 |
| Lazy zip reads modified list values | Pass | 8.5312089e-9 | 1.9648672e-8 |
| Dict comprehension keeps last duplicate-key value | Pass | 4.5051677e-9 | 1.2163652e-8 |
| Missing a reduction rotation | Numerical rejection | 0.33686968 | 0.62500000 |
| Premature list(zip(...)) snapshot | Numerical rejection | 0.25881522 | 0.50000002 |
| Reversed duplicate-key entries preserve wrong value | Numerical rejection | 0.25881523 | 0.50000001 |

SEAL4.0.0, polynomial degree 32768, fourteen 60-bit moduli, tc128 and existing
compiler scale settings were retained. Each value must satisfy
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`. Reports retain individual
decrypted values/errors, nonzero-reference relative error and cosine similarity.
References are independently recomputed from original inputs, not intermediate
decrypted values or the expanded candidate.

Set `POSEIDON_ITERATION_REPORT` to this report path in the existing pinned
Nix/Python environment to enable the real-artifact audit. It verifies current
producer hashes, immutable request/model/arrays, normalized source, construction
metadata, IR/HEVM/CST hashes, actual-encryption flags and recomputed metrics.
The six semantic groups with saved real evidence passed **105/105, no skips**.
Full offline regression: **542 tests, 432 passed, 110 conditional skips, no
failures**. Expected invalid-argument and mocked retry output is not a live API
failure. Skipped tests are not counted as passed.

Older parameter-binding results retain their original generator hashes; the
audit recognizes only those specific historical producers and separately
reproduces their exact old-mode expansion. It does not relabel old execution
as an experiment with the new iterator engine.

Six temporary private-key directories were removed by the existing per-case
cleanup, totaling **4,073,880,060 bytes (about 3.79 GiB)**. Random test keys are
regenerable. Inputs, reference, source, IR, compiler artifacts and decrypted
results remain. No unrelated cleanup, package install, paid API, commit, push
or PR occurred. No GPU/backend implementation was changed.
