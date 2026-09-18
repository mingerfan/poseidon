# Lexical construction helpers: CPU correctness evidence

## Confirmed implementation, not full DSL completion

`run_candidate.py --function-composition` selects request
`hecate-function-synthesis-v8` / AST `hecate-function-v7`. It accepts one
decorated golden function and bounded undecorated top-level construction helpers.
The trusted data-only interpreter produces a straight-line Hecate function;
the existing Dacapo compiler and real SEAL CPU runtime execute it.

This supports **ordinary Python-style graph-construction helpers**, not multiple
decorated Hecate IR functions or the unresolved upstream `Func.__call__` /
`createCall` path. No generated Python function is executed. No arbitrary module
imports, external calls, network/filesystem access or new backend is enabled.
The earlier `public_construction.py` engine remains unchanged for historical
request-v7 reproducibility; `function_construction.py` is the new versioned engine.

## What the function boundary preserves

| Semantic requirement | Implemented behavior |
|---|---|
| Argument evaluation | Resolve the declared callable, then evaluate arguments left-to-right exactly once. |
| Local scope | A fresh frame per call, including calls repeated from a loop. Locals are initially unbound, so an earlier invocation cannot supply a missing value. |
| Lexical globals | Helpers see module public constants and declared helpers, not the caller's local bindings. |
| Name shadowing | Helper parameters/locals may shadow a public name without modifying the immutable constant registry. A read before local assignment fails rather than falling back to the global. |
| Ciphertext arguments | Immutable symbolic values; rebinding a parameter does not rebind the caller's variable. |
| Local list arguments/results | Shared references; element writes and append are visible through aliases. Rebinding the parameter to a new list is local. |
| Composition | Helpers can call other helpers, accept declared helper values as parameters, and return a declared helper for later use. |
| Early return | Publicly selected returns unwind the current call even from nested loops, then restore the caller's frame. |
| Void helper | Implicit/bare return is None, usable for list side effects. Golden must still return the prescribed ciphertext results. |
| Recursion | Public, bounded recursive construction is interpreted before tracing; it is not recursive ciphertext execution. |

There are at most 16 helper declarations, 16 positional parameters per helper,
128 total helper calls and 16 live call frames. The existing 4096-step, depth-32,
256-cipher-operation, source-size, container-size and provisioned-rotation gates
also apply; whichever resource limit is reached first stops construction.

Nested definitions/closures, defaults, keyword/variadic arguments, richer public
array construction, comprehensions and other pending syntax are not yet enabled.
Generalized packing, all upstream helpers, IR-level calls, true bootstrap and
upscale remain part of the full goal, not excluded from its denominator.

## Tests and independent formulas

The manual goldens exercise:

1. `Linear(4,2)` implemented by a linear helper receiving a reduction helper as
   an argument; each row calls that helper to perform real cross-slot reduction.
2. `0.5*x**4+0.125` constructed using recursive squaring and early return.
3. `1.5*x+0.375` using a void helper that updates a shared list, followed by a
   helper that returns from inside a public loop.

Each has a targeted wrong program: missing one reduction rotation, one too few
recursive squarings, or rebinding a list parameter instead of modifying its item.
The original-input references are independent of the generated/expanded DSL.

### Initial failure and bounded repair

The first quartic golden accidentally implemented only `x**4`, omitting the
existing model's coefficient 0.5 and offset 0.125. Trace, compile and real FHE
execution succeeded; the numerical comparison correctly failed, maximum error
0.3750000186136837. This was a **manual golden source error**, not a compiler,
TLS, runtime or numerical-parameter failure.

The new plaintext preflight reproduced it at zero input: actual `[0,0,0,0]`
versus expected `[0.125,0.125,0.125,0.125]`. Only the manual positive/negative
quartic sources were corrected to include the supplied constants. The negative
still uses the wrong recursion depth, isolating that fault. References, weights,
inputs, compiler settings, security and tolerances were not changed.

The original six-run report remains failed and untouched:

- `/home/lhy/poseidon-work/results/function-composition-goldens-_44vgx7x/report.json`
- SHA256: `2f3086c1cd1439dee017dacd48bcafe4a9306a8c0b5deb972e5239b2385007f0`

Only the quartic pair was rerun, both reaching their expected gates:

- `/home/lhy/poseidon-work/results/function-composition-goldens-nclk_e80/report.json`
- SHA256: `d5cb8d0a35db2686ec7bba6efa2c1eefeadd0df6472748b9e1163d979b0f72cc`

Final evidence selects the latest result for each `(case, counterexample)` pair;
it does not overwrite or hide failed runs and does not count retries as extra
successful cases. There are six selected roles, 24 input groups and 80 output
comparisons. Eight actual runs occurred including the targeted rerun.

| Selected program | Expected outcome | Maximum absolute error |
|---|---|---:|
| Higher-order Linear helper | pass | 1.3013331814915041e-8 |
| Recursive quartic with coefficient/offset | pass | 2.553001965388546e-8 |
| Shared list + return from loop | pass | 1.2664070958567919e-8 |
| Missing reduction rotation | numerical rejection | 0.6249999979184588 |
| Too few recursive squarings | numerical rejection | 0.12489675478612322 |
| Wrong list rebinding | numerical rejection | 0.5000000036057435 |

These are **manual** programs, not new live Agent generations. No paid API calls
occurred. The direct, higher-order and recursive calls produce real Hecate IR,
not simulation. The FHE backend remains SEAL 4.0.0 CPU, N=32768, 14×60-bit
modulus, tc128, existing scale/waterline, atol=1e-5 and rtol=1e-4, no bootstrap.

## Reproduction and durable checks

From `/mnt/d/Code Space/Poseidon` in WSL:

```bash
timeout -k 3s 1950s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_function_composition_goldens.py
```

The script now runs the independent plaintext golden preflight before generating
keys or running native code. `--quartic-only` reproduces the targeted pair rather
than rerunning unchanged cases. This script does not call any provider.

`test_function_construction` contains 16 scoped-interpreter unit tests, one
independent golden-formula preflight and one opt-in real-evidence audit.
Set `POSEIDON_FUNCTION_REPORTS` to a JSON array containing the original and rerun
report paths in that order, and run the module in the existing pinned Nix/Python
environment. The audit verifies original source, immutable requests and artifacts,
recomputed expansion, actual decrypted arrays, independent formulas, error
metrics, security parameters and key cleanup.

New and historical real-evidence suites passed **48/48 without skips**:
`test_function_construction`, `test_public_construction`, `test_extended_arithmetic`,
with their explicit saved-result environment variables. Full offline regression:
485 tests, 378 passed, 107 conditional skips, no failures. Skips are not passes.

All eight temporary private-key directories were cleaned individually:
5,431,840,080 bytes (about 5.06 GiB). Original failed/successful sources, arrays,
Earth/CKKS IR, HEVM/CST, decrypted outputs and diagnostics remain available.
No install, large download, Git commit/push or GPU integration occurred.
