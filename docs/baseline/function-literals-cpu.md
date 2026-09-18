# Lambda construction functions and stable public-key sorting

## Source basis and scope

Pinned upstream `python/poly/poly/Poly.py:36` returns a lambda from genRelu6;
`python/poly/poly/MPCB.py:47` builds calc_order with
`sorted(..., reverse=True, key=lambda x: x[0])`. These are real construction
language features, not newly invented FHE instructions.

`run_candidate.py --function-literals` selects request
`hecate-function-synthesis-v12` / AST `hecate-function-v11`, including earlier
iteration, parameter and closure support. Defaults/older contracts remain
unchanged. The feature interprets lambda AST using the existing lexical frames;
no generated Python function is created or executed. The compiler still sees
the rechecked straight-line Hecate expansion, not runtime lambda instructions.

## Semantics

- Lambda parameters/defaults follow the existing helper binding rules, including
  positional-only, keyword-only and bounded varargs/kwargs.
- Free variables are shared late-bound lexical cells; defaults retain the value
  or reference computed when that lambda expression is evaluated.
- Lambdas in a comprehension capture its implicit scope. Default capture can
  intentionally freeze each iteration's value; ordinary closure capture does not.
- Function values can be called directly, returned then called, selected from a
  public container or chosen by a public conditional. Only declared AST function
  objects can be invoked; arbitrary Python callables/attributes remain blocked.
- `sorted` first materializes the bounded public iterable, computes each key
  once in input order, then sorts stably. Equal-key elements retain their order
  even with reverse=True; reversing an ascending result is not equivalent.
- Key callbacks may be declared helpers or lambdas. Key values must be public
  int/bool/str or nested list/tuple structures of those types. Ciphertext and
  opaque plaintext symbolic values are rejected, including within key containers.
- Shared key containers are revalidated after all callbacks, since a later
  callback may mutate an earlier key. Reverse must also be public.

There are at most 17 named/lambda declarations in total and 128 nested/lambda
instances, with the existing 128 calls, 16 call frames, 4096 steps, 128 container
entries and 256 emitted cipher operations. Sorting reorders public construction
objects, never secretly compares ciphertext values or permutes encrypted slots.

This does not implement full genRelu6/GenPoly: public numerical arrays, floating
expressions, integer-key dictionaries, other control syntax, general packing,
complete high-level helpers and true bootstrap/upscale remain separate gaps.

## Verification interface

`test_function_literals` covers direct/returned/container call targets, lambda
defaults versus comprehension cells, independent factories, default side effects,
stable forward/reverse sorting, key-once behavior, input materialization before
callbacks, mutable-key revalidation, forbidden secret sorting and resource limits.
Symbolic tests and plain-formula preflight are not FHE or Agent execution evidence.

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_function_literal_goldens.py
```

The manual fixtures are:

1. Linear(4,2): sort public row indices, then call lambda/helper dot products with
   actual cross-slot rotation reduction; negative reverses output-row order.
2. Affine `1.5*x+0.375`: lambdas capture per-iteration default weights; negative
   captures the shared late-bound loop variable instead.
3. The same affine reference: equal-key reverse sorting must preserve item order;
   negative incorrectly reverses an ascending sorted result.

All use the existing immutable model/input/reference, security profile and
numerical tolerance, with no paid API, new backend or bootstrap substitute.

## Completed actual CPU evidence

Report: `/home/lhy/poseidon-work/results/function-literal-goldens-rm7dglir/report.json`

SHA256: `33390a01c0c06d20b4b61720698e7ea20d3e6c8bd963989e383215e53b4e761c`

All six roles matched expectations on the first run: 3 positive programs and
3 deliberately wrong programs, **24 input groups / 80 compared values**. Every
wrong program completed parsing, tracing, compilation and actual encrypted
execution, then failed only at numerical_comparison. No rerun was needed.

| Program | Expected | MAE | Maximum absolute error |
|---|---|---:|---:|
| Public row sort + lambda dot-product calls | Pass | 1.6415731e-9 | 5.6090353e-9 |
| Per-iteration lambda default captures | Pass | 5.8722613e-9 | 1.5131283e-8 |
| Stable reverse sorting with tied keys | Pass | 3.8237272e-9 | 1.1853426e-8 |
| Incorrect reversed Linear row order | Numerical rejection | 0.23237256 | 0.61699026 |
| Incorrect shared late-bound weight | Numerical rejection | 0.06470381 | 0.12500001 |
| Reversing an ascending sort instead of stable reverse sorting | Numerical rejection | 0.25881523 | 0.50000001 |

The profile remained SEAL4.0.0, N=32768, fourteen 60-bit moduli, tc128, with
unchanged scale/compiler settings. Elementwise tolerance remains
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`. Reports retain per-element
outputs, absolute/relative error and cosine similarity. There is no GPU,
bootstrap or simulated-encryption claim.

Set `POSEIDON_LITERAL_REPORT` to the report path in the existing pinned Nix
environment to enable the actual evidence audit. It checks producer hashes,
immutable requests/arrays/reference, original and normalized source, IR and
HEVM/CST hashes, security parameters, real execution flags and recomputed
numerical metrics. New and historical seven-group audit: **122/122 passed,
zero skips**. Old iterator reports retain their original producer hashes and
their expansion is separately reproduced using the current compatibility mode.

WSL emitted a systemd user-session startup warning once, but the Nix audit
completed with exit zero and all tests passed. No system repair was attempted
for this non-blocking warning.

All six temporary private-key directories were cleaned by the existing per-run
policy, **4,073,880,060 bytes (about 3.79 GiB)**. Random test keys can be generated
again; source, original inputs/reference, artifacts and decrypted results remain.
No unrelated files were deleted. No dependency installation, paid API call,
commit, push or PR occurred. These manual goldens are not online Agent cases.

The full DSL goal remains incomplete: broader public numerical/array semantics,
remaining control syntax, general shape/packing, all actual upstream helpers,
IR calls, real bootstrap/upscale and online generation coverage remain pending.
