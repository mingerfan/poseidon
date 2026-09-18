# Public control construction: rules and actual CPU evidence

## Scope and upstream source

`--public-control` selects request `hecate-function-synthesis-v15` and AST
`hecate-function-v14`. The single-case and batch Agent entrypoints accept it;
the model receives the corresponding versioned rules and semantic guidance.
It includes preceding construction semantics without changing historical
requests, weights, reference, encryption profile or period-four packing.

Pinned Dacapo `python/poly/poly/MPCB.py:GenPoly` uses integer keys in
`giantTs`, tuple keys in `tmpPoly`, membership, pass and public control
when constructing Chebyshev expressions. This extension addresses that real
construction gap. It does NOT implement all of GenPoly: coefficient string
parsing, np.polynomial.Chebyshev, polynomial division/remainders, np.log2 and
other helper dependencies remain separate work.

## Implemented semantics

- Bounded public while, for/while-else, break, continue and pass.
  Else runs on exhaustion/false condition, not break or return.
  Loop-control ownership is lexical: an inner loop's else may break/continue
  its outer loop; a called helper cannot break its caller's loop.
  Invalid break/continue are rejected statically, including dead code.
- and/or short-circuit and return the selected operand, not a coerced bool.
  Only actually truth-tested operands must be public; the final selected
  operand can be ciphertext. not returns a public bool.
- Truth of None, public scalars/containers, functions and iterators matches
  Python. An exhausted iterator remains truthy. Cipher/array truth is rejected.
- Immutable public None/bool/int/float/string/recursive tuple dictionary keys,
  including Python numeric collision behavior (1 == True == 1.0).
  Comprehensions, reads and writes use the same key checks.
  Keyword ** unpacking still requires string keys.
- Public scalar/list/tuple/dict equality; numeric/string/lexicographic ordering;
  chained comparisons short-circuit. Dictionary membership checks keys;
  string membership checks substrings. List/tuple/iterator membership compares
  public elements, consuming an iterator through the first match or exhaustion.
- No cipher comparison, encrypted condition, cipher-slot indexing, mutable/
  array/function keys, cyclic containers or arbitrary candidate Python execution.
  Existing expansion/depth/operation/encoding bounds remain enforced.

Location and data flow:

1. candidate_contract creates a versioned public model/rules request.
2. hecate_contract validates the candidate's AST using function_construction.
3. The bounded interpreter resolves public control into a flat arithmetic
   Hecate function plus a separately hashed derived-constant manifest.
4. candidate_trace binds actual Hecate values in the existing isolated process.
5. Dacapo emits Earth/CKKS IR and HEVM/CST; existing SEAL HEVM CPU executes
   real ciphertexts. Decrypted outputs are compared with frozen reference.

The local construction interpreter is not the HEVM runtime. Adding a public
while loop does not add a secret-dependent loop instruction to HEVM.

## Actual manual encrypted experiment

Report:
`/home/lhy/poseidon-work/results/public-control-goldens-y31s5rso/report.json`

SHA256:
`7d1af5d43f88b1cae2d617525d33137f84665e2cf043dd279b33ffcd1521cfe8`

All six roles matched their expected outcome on the first execution:
3 positive programs passed and 3 counterexamples completed encrypted execution
but failed numerical comparison. Four input groups per role, 24 groups total,
80 output values including positive and negative roles.

| Program | MAE | Maximum absolute error | Outcome |
|---|---:|---:|---|
| Chebyshev T3 via tuple-keyed recurrence and while-else | 1.39951e-8 | 3.36346e-8 | Pass |
| Linear via integer-keyed rows and for/break | 2.89468e-9 | 5.87769e-9 | Pass |
| Affine via continue and exhaustion else | 4.79107e-9 | 1.15574e-8 | Pass |
| Recurrence omits subtraction, yielding 4x^3 | 1.55289 | 3.00000 | Numerical rejection |
| Linear swaps two output rows | 0.232373 | 0.616990 | Numerical rejection |
| Affine replaces continue with break | 0.0647038 | 0.125000 | Numerical rejection |

The Chebyshev reference is independently represented by the graph
`4*x*x*x - 3*x`, not evaluated using the candidate recurrence. Linear uses
independent W*x; affine uses 1.5*x+0.375. The audit recomputes these formulas
against original arrays and independently recomputes comparison metrics.

Unchanged profile: SEAL 4.0.0, N=32768, fourteen 60-bit moduli, tc128.
Per-element tolerance: abs(actual-reference) <= 1e-5 + 1e-4*abs(reference).
No bootstrap, decrypt/re-encrypt simulation or Poseidon GPU claim.

All six terminal-run key cleanups succeeded: 4,073,880,060 bytes (~3.79 GiB).
Original random private keys are removed and unrecoverable; equivalent fresh
test keys can be generated. Requests, references, source, IR, HEVM/CST,
decrypted outputs, diagnostics and hashes remain available.

## Validation and reproducibility

Offline tests include 1,176 public comparison combinations and 250 loop
configurations against independent native Python behavior, plus short circuit,
iterator consumption, scope, key collisions, resource limits and rejection
tests. These symbolic tests are not encrypted-execution evidence.

Final offline suite: 607 tests, 492 passed, 115 conditional skips, no failures.
New plus historical semantic/artifact audit: 187/187 passed, no skips.
Historical expansion metadata is reproduced exactly; added key validation
must not increment old-contract counters.

From the source root in WSL:

```sh
timeout -k 3s 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -q
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_control_goldens.py
```

The golden runner uses the existing isolated Nix environment and performs
no external model API calls. Do not overlap Nix/native batches.
To audit the preserved batch, set POSEIDON_CONTROL_REPORT to its report path
and run test_public_control in the existing pinned NumPy/Nix environment.
Producer hashes are fixed in ControlEvidenceTests; all artifact/reference
hashes, expansion metadata, security profile and numerical results are checked.

During development a Windows test invocation could not import Linux fcntl;
the full suite was rerun in WSL. A rule declaration inserted before its
dependency caused an import NameError; declaration order was corrected before
any actual encrypted test. Neither error is a model generation failure.

## Remaining full-goal requirements

This is manual golden evidence, NOT an online Agent generation success rate.
The new paid batch flag is wired and its plan is tested without credentials;
changing a grammar contract cannot be disguised as retrying an old prompt.
No paid API call was made in this experiment.

Full DSL support remains incomplete: further public arrays/mutation/numerical
operations, string/dictionary/generator construction, general shape/packing,
complete upstream helpers, multi-decorated IR calls, true bootstrap/upscale
and online Agent validation of the expanded grammar are still required.
The prior specifically blocked paid disclosure is not retried by a workaround.
