# Public string parsing and actual CPU evidence

## Purpose and source

`--public-strings` selects request `hecate-function-synthesis-v16` and AST
`hecate-function-v15`, including previous construction semantics. Single-case
and batch Agent entrypoints both accept it and carry versioned semantic rules.
Old requests and old contract behavior remain unchanged.

In pinned Dacapo `python/poly/poly/MPCB.py:GenPoly`, public tree text is parsed
using `token.strip().split(" ")` and nested int conversion; coefficients use
`float(token.strip()) / scale`. The previous contract rejected these string
method calls. They are public graph-construction work, not encrypted string
processing or a new HEVM instruction.

## Implementation and semantic boundaries

- `public_strings.py` dispatches a fixed set of native str operations on exact
  public str receivers: strip/lstrip/rstrip, split/rsplit, partition/rpartition,
  replace and join. It has no file, process, network or candidate-code execution.
- `function_construction.py` resolves receiver first, then positional/starred
  arguments, then keyword expansions. Only direct method calls are allowed;
  capturing a bound method or accessing arbitrary attributes is not enabled.
- strip's argument is a character set, not a prefix/suffix. Explicit split
  separators retain empty fields; whitespace split collapses runs and discards
  leading/trailing empty fields. rsplit applies maxsplit from the right.
- Preserve pinned Python 3.10 method signatures: only split/rsplit accept
  sep/maxsplit keywords; other methods are positional-only. Duplicate keywords,
  invalid receiver/argument types and empty split separators are rejected.
- join consumes public iterables (including shared iterators) and requires
  strings only. It does not stringify ciphertexts or arbitrary objects.
- Inputs/results remain bounded to 128 characters/items and bounded public
  integer arguments. join checks aggregate output length before allocation.
  Other methods act on bounded inputs and check output bounds. Numeric parsing
  still rejects nonfinite values and preserves existing encoding limits.
- The normalizer emits flat Hecate arithmetic and hashed derived constants.
  Original request, weights, reference, security profile and packing remain fixed.
- `candidate_sandbox.py` supplies the new module to the existing isolated
  tracing worker; `candidate_trace.py` and the contract/coverage dispatch handle
  the new version. No unrestricted execution fallback was introduced.
- Batch reports record public_strings and reject changing a DSL contract while
  labeling the run as a same-prompt retry. Offline plan tests never load keys.

This is not all Python string semantics. format, encode/decode, bound method
values and other methods remain unsupported. Nor does it complete GenPoly:
np.polynomial.Chebyshev, polynomial // and %, coefficient attributes, np.log2,
other public NumPy operations and complete helper validation remain work.

## Actual encrypted experiment

Report:
`/home/lhy/poseidon-work/results/public-string-goldens-tauf65i2/report.json`

SHA256:
`ac394ece0049a17e11143d6751d89c55af763a668f0c3b00140135334b144f03`

All six manual roles matched on first execution: three correct programs passed,
three incorrect parsers completed actual encrypted execution then failed
numerical comparison. There are four input groups per role, 24 groups total,
80 compared values including positive and negative roles.

| Role | MAE | Maximum absolute error | Outcome |
|---|---:|---:|---|
| Parse signed coefficient text, construct Chebyshev T3 | 1.40579e-8 | 6.21398e-8 | Pass |
| Parse comma-separated Linear rows, rotate-and-sum | 2.99081e-9 | 6.28493e-9 | Pass |
| Parse explicit-space affine coefficients | 4.32537e-9 | 1.82398e-8 | Pass |
| Incorrect strip removes coefficient sign | 1.27617 | 2.00000 | Numerical rejection |
| Incorrect replacement removes negative weight signs | 0.282280 | 0.509618 | Numerical rejection |
| Whitespace split changes empty-field-dependent coefficient | 0.129408 | 0.250000 | Numerical rejection |

References remain independent: 4*x^3-3*x, matrix multiplication with original
fixed weights, and 1.5*x+0.375. They are not recomputed from parsed candidate
coefficients or decrypted intermediates.

Backend is existing Dacapo/SEAL HEVM CPU, not Poseidon GPU. Unchanged SEAL4.0.0,
N=32768, fourteen 60-bit moduli, tc128, atol=1e-5 and rtol=1e-4.
No bootstrap or decrypt/re-encrypt simulation.

All terminal-run key cleanups succeeded: 4,073,880,059 bytes (~3.79 GiB).
The original random keys are removed and unrecoverable; fresh equivalent test
keys can be generated. Requests, arrays/reference, source, Earth/CKKS IR,
HEVM/CST, decryptions, diagnostics and hashes remain.

## Tests and reproduction

- 616 combinations of string methods/inputs/arguments compared with native
  Python and also passed through the actual AST interpreter.
- 12 join combinations plus keyword, iterator-consumption, evaluation-order,
  invalid capability/type, nonfinite conversion and resource-bound tests.
- Final full offline suite: 616 tests, 500 passed, 116 conditional skips.
  The suite total and string matrix count happen to coincide; they are different
  denominators. Skipped tests are not counted as passed coverage.
- New and historical semantic/actual-artifact audit: 196/196 passed, no skips.
  It verifies source identities, frozen hashes, exact normalized source and
  metadata, derived constants, actual compilation/execution and comparisons.
- Intentional invalid argparse values and mocked transport retries in unit
  logs are negative tests, not failed paid requests. No API call was made.

From the WSL source root:

```sh
timeout -k 3s 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -q
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_string_goldens.py
```

The golden runner enters existing pinned Nix/Dacapo dependencies. Do not overlap
native/Nix batches. For a preserved-evidence audit, set POSEIDON_STRING_REPORT
to the report path and run test_public_strings in the pinned NumPy/Nix environment.
StringEvidenceTests contains the actual producer hashes, including
function_construction 6ec591c71a7f96bcdc42c2fb131db2799aac743359ab5c0ae1979685493cc9d0.

## Remaining goal

This is manual validation, not measured online Agent success under request-v16.
No additional paid API calls, environment installation or GPU integration were
performed. The prior specifically blocked paid disclosure was not bypassed.

General arrays/mutation, more numeric/string/container/generator semantics,
general shape/packing, complete upstream helpers, IR calls, genuine bootstrap/
upscale and Agent generation validation remain part of the full open goal.
