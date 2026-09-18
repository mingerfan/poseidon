# Public Chebyshev data and upstream GenPoly validation

## What changed

`--public-polynomial` selects request `hecate-function-synthesis-v17` / AST
`hecate-function-v16`. It includes previous construction contracts. Both
single-case and batch Agent entrypoints carry the versioned rules; changing
the contract cannot be disguised as retrying an unchanged prompt.

This milestone goes beyond hand-written recurrence examples: the fixtures
contain AST-identical copies of `fint` and `GenPoly` extracted from
`third_party/dacapo/python/poly/poly/MPCB.py`. Tests compare their function ASTs
against the current upstream file, not merely their names.

The Dacapo gitlink remains 4616402710f39df3e5f5bd7930a6c036025aaac3.
MPCB.py has no local Git diff; file SHA256 is
6dc4d64c648a2ad00159186e5cd35c5ed9f22d64c8e1a7e300774fbd933e65e8.
Formatting/comments are normalized when copying the functions, not their AST.

## System location and responsibilities

1. public_polynomial.py stores immutable, checked public coefficient/domain/
   window/symbol data. It delegates whitelisted arithmetic to the existing
   NumPy Chebyshev implementation, never exposes an unrestricted native object.
2. function_construction.py recognizes the allowed constructor, coefficient
   attributes, numeric functions and dtype markers; it resolves public constants
   before using them. Polynomial objects stay on the public construction side.
3. GenPoly returns a closure which builds cipher arithmetic. The bounded
   interpreter expands that closure into flat Hecate +,-,* operations and
   separately hashed derived constants.
4. Existing isolated tracing creates Earth IR; Dacapo lowers to CKKS/HEVM/CST.
   Existing SEAL HEVM CPU executes real ciphertexts. No new execution backend.
5. Frozen independent monomial reference graphs and original arrays provide the
   differential oracle. No reference is computed from candidate coefficients or
   decrypted intermediate values.

NumPy 1.25.2 is already installed in the pinned Nix/venv. No package was installed.
The golden precheck uses that environment and releases its lock before the
per-case compiler runs. Do not overlap Nix/native batches.

## Supported public operations and boundaries

- np.polynomial.Chebyshev(coef, domain=None, window=None, symbol='x'), with
  explicit positional/keyword arguments. Nonempty scalar/1D real coefficients,
  at most 128 values. Construction preserves trailing zeros. Domain/window
  contain two endpoints; symbol is a bounded identifier.
- Polynomial +,-,*,//,%, unary +/- and integer powers 0..16. // and % are
  series quotient/remainder, not elementwise division. Incompatible polynomial
  domain/window/symbol metadata is rejected by native series arithmetic.
- .coef/.domain/.window expose read-only public arrays. Original named constants
  work as constructor keyword arguments and as public arithmetic operands.
- np.double/np.float64 dtype markers for existing array constructors.
- Public scalar/array np.floor, np.ceil and np.log2, with finite-domain checks.
- Existing finite magnitude and expansion bounds remain. Product/power result
  degree is below 128. This is bounded support, not arbitrary-size NumPy.

No unrestricted NumPy, mutable coefficient views, object calls p(x), direct
polynomial/cipher arithmetic, roots, fitting, differentiation, integration,
serialization or encrypted numeric ufuncs are enabled. Cipher operations still
use the existing Hecate lowering rules.

### Important upstream application restriction

GenPoly's leaf loop accesses `leaf.coef[2*k+1]`: only odd coefficients.
It is therefore not a general Chebyshev evaluator for arbitrary models.
The rules explicitly warn the Agent about constant/even terms and tree/length
compatibility. The implementation is not silently rewritten to hide this fact.

A model including 0.25*T2 was tested through the unchanged helper: compilation
and encrypted execution succeeded, but numerical comparison correctly rejected
the missing even term (maximum error about 0.25). That failure is a model/helper
semantic mismatch, not an environment, compiler or SEAL failure.

## Actual encrypted evidence

Initial seven-role report:
`/home/lhy/poseidon-work/results/public-polynomial-goldens-b38hl0i7/report.json`

SHA256:
`cfec772874f1584ab68d995292cf83b430f0ab0850733c4012bac0762cced691`

One additional branch-coverage report, not a restart of the first batch:
`/home/lhy/poseidon-work/results/public-polynomial-giant-mcigw32r/report.json`

SHA256:
`11bd2c9a0b2d434d4a2744b7c8f04d1f164c8ea632b2e44a8e9cca6b2fd2f0d4`

All 7 initial roles and the 1 added role matched on first execution. Combined:
4 positive programs, 4 numerical counterexamples, 32 input groups, 128 output
values. They are manual goldens, not online Agent generation attempts.

| Role | MAE | Maximum absolute error | Outcome |
|---|---:|---:|---|
| Cubic, single leaf | 9.74506e-9 | 3.41110e-8 | Pass |
| Cubic, quotient/remainder tree | 1.50217e-8 | 6.08092e-8 | Pass |
| Seventh degree with scale=2 | 1.13993e-8 | 9.05247e-8 | Pass |
| Wrong cubic coefficient | 0.159521 | 0.250000 | Numerical rejection |
| Wrong scale in decomposition | 0.172942 | 0.375000 | Numerical rejection |
| Wrong fifth-basis coefficient sign | 0.153308 | 0.250000 | Numerical rejection |
| Even-term model sent through odd-only helper | 0.196669 | 0.250000 | Numerical rejection |
| Fifth degree, generate missing giant degree | 7.69677e-9 | 5.10997e-8 | Pass |

Independent reference expressions:

- Cubic: 2*x^3 - 1.25*x.
- Fifth degree: 2*x^5 - 0.5*x^3 - 0.625*x.
- Seventh degree: 4*x^7 - 5*x^5 + 2*x^3 - 0.4375*x.
- Even-term negative: cubic + 0.5*x^2 - 0.25.

The extra fifth-degree case uses tree [4; 0 0], length=4: giantTs initially has
degree 2, and the later missing-degree branch constructs degree 4. Initial
positive cases did not cover that branch; it was tested separately.

Unchanged profile: SEAL4.0.0, N=32768, fourteen 60-bit moduli, tc128;
atol=1e-5, rtol=1e-4. No bootstrap, decrypt/re-encrypt simulation or GPU claim.
All temporary key cleanups completed: 5,431,840,080 bytes (~5.06 GiB).
Original random keys are unrecoverable; new equivalent test keys are regenerable.
Reference, requests, source, IR, HEVM/CST, decryptions and diagnostics remain.

## Verification and known development failures

- Independent recurrence evaluates Chebyshev coefficient vectors; arithmetic
  tests use 36 operand pairs for +,-,* and quotient/remainder reconstruction for
  30 nonzero-divisor pairs at five sample points. No NumPy polynomial evaluator
  is used as the sole mathematical oracle.
- Fixed NumPy 1.25.2 new/historical semantic and artifact audit: 210/210 passed,
  no skips. This includes actual coefficient/decomposition tests, original helper
  AST identity, frozen hashes, exact expansions, security and numerical checks.
- System-Python offline suite: 630 tests, 500 passed, 130 conditional skips.
  New NumPy-dependent tests are skipped there because system Python lacks NumPy;
  they are actually run in the separate pinned-environment audit, not counted
  as successful from the skipped suite.
- The first system-Python invocation failed with ModuleNotFoundError: numpy.
  Root cause was environment selection; existing Nix/venv was used, no installation.
- Supplementary checks found missing conversion of named public constants in
  constructor keywords and polynomial arithmetic. Both were fixed and covered
  by regression. Historical GenPoly expansions reproduce exactly after the fix.
  Original seven-role producer and supplemental producer hashes are both
  explicitly retained; historical encrypted executions are not relabeled.

## Reproduction

From the WSL source root:

```sh
timeout -k 3s 2700s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_polynomial_goldens.py
timeout -k 3s 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -q
```

The current golden runner runs all eight roles. To audit the preserved split
experiment, set POSEIDON_POLYNOMIAL_REPORT to a JSON list of the two report paths
and run test_public_polynomial in the existing pinned NumPy/Nix environment.
A future single full eight-role report may instead be supplied as one path.

## Remaining full goal

The measured GenPoly configurations and newly admitted public operators are
supported; arbitrary helper inputs/tree layouts are not claimed correct.
Online Agent generation under request-v17 remains unmeasured. This turn made
zero paid API calls and did not bypass the previous specific disclosure gate.
General arrays/mutation, remaining public/container/string semantics, general
shape/packing, other complete upstream helpers, IR calls and true bootstrap/
upscale remain part of the full unfinished goal.
