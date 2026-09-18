# Public numeric construction and derived constants

## Scope, location and source evidence

CLI `--public-numbers` selects request `hecate-function-synthesis-v14` / AST
`hecate-function-v13`. It includes earlier construction contracts; historical
requests are unchanged. This is a bounded part of full Hecate semantic support,
not all Python/NumPy or all upstream helpers.

Upstream evidence (in the pinned Dacapo checkout):

- `python/hecate/hecate/expr.py:resolveType` converts int, float, lists, NumPy
  arrays and Torch tensors into Plain; the existing harness supplies float64
  length-one/four arrays for its original named constants.
- `python/poly/poly/MPCB.py:GenPoly` parses and divides public coefficients
  before constructing expressions such as `2 * poly * giantTs[idx]`.
- `expr.py:toBinary` lists add/sub/mul, not cipher division or exponentiation.
  Public division/power support must not invent a cipher lowering for these.

Implementation flow:

1. `function_construction.py` interprets permitted AST, never generated Python.
2. `public_numeric.py` computes bounded public values, shape-aware broadcasting,
   indexing and C-order reshape. It has no filesystem/network/process access.
3. At a cipher operation, the normalizer materializes scalar or 1D length-one/
   four operands into a separate `derived_constants` registry. The original
   constants dictionary, request, weights and reference are unchanged.
4. The checked flat source references original and derived names.
   `candidate_trace.py` saves `derived-constants.json`, binds the effective
   manifest through actual Hecate `resolveType`, then uses existing compilation.
5. Artifact hashes include the derived manifest. The evidence auditor recomputes
   the original expansion and manifest, checks input integrity and independently
   compares actual decrypted arrays with the frozen plaintext reference.

## Supported operations and retained boundaries

- Finite public int/float arithmetic: +, -, *, /, //, %, **, unary +/-, and
  bounded `float`, `int`, two-argument `pow`. Public scalar floats may be used
  in conditions and sort keys. Complex/nonfinite/zero-division results fail.
- Checked `np.array/asarray(value, dtype="float64")`, with optional dtype.
  Without dtype, infer real integer/floating data; preserve dtype for empty
  arrays as well. No arbitrary NumPy or external calls.
- Public arrays: rank <=4, <=128 elements; trailing-dimension broadcast,
  integer/slice/tuple indexing, iteration, `.shape`, `.reshape(...)`, `.flatten()`.
  Python list +/* keep sequence semantics unless interacting with an array.
- Current arrays are read-only values: item, slice and augmented array writes
  are rejected. In particular, NumPy `a += b` must not be faked by rebinding an
  immutable value and leaving aliases unchanged.
- Conversion/traversal budgets are checked before expanding repeated nested
  containers. Public magnitude <=1048576; exponent magnitude <=16; encoded
  magnitude <=1024; original plus derived constants <=128.
- Encoding still requires scalar or 1D length one/four. Multidimensional arrays
  require explicit reshape/flatten; no silent packing change. General tensor
  packing, arbitrary slot counts and encrypted indexing are not implemented.
- Cipher operations remain +, -, *. Array comparisons, reductions, matmul,
  advanced/Boolean indexing, complex/Boolean dtypes, array mutation, general
  Torch/NumPy and np.polynomial remain gaps, as do true bootstrap/upscale.

`constants_changed=False` means the input registry was preserved. New-contract
metadata additionally records `input_constants_changed=False`, the derived
count and a canonical derived-manifest SHA256. It does not claim that no new
encoding constants were produced.

## Actual manual CPU evidence

Report: `/home/lhy/poseidon-work/results/public-number-goldens-ot5bw8xa/report.json`

SHA256: `c1dbb2cf84acf24af24046355dea4069b5403ba61d15c44b3883d63203cd5729`

All six manual roles matched on their first run, 24 input groups / 80 compared
values. Three positives passed; three intentional mistakes completed real
encrypted execution and were rejected at numerical comparison, not parsing.

| Role | MAE | Max absolute error | Result |
|---|---:|---:|---|
| Linear via public matrix arithmetic and row iteration | 3.06118e-9 | 4.99807e-9 | Pass |
| Scalar-derived coefficient and offset | 6.62910e-9 | 1.62536e-8 | Pass |
| Broadcast matrix, explicit reshape, affine | 4.91112e-9 | 1.50526e-8 | Pass |
| Reversed Linear rows | 0.232373 | 0.616990 | Numerical rejection |
| Wrong scalar coefficient | 0.258815 | 0.500000 | Numerical rejection |
| Wrong broadcast column | 0.133343 | 0.500000 | Numerical rejection |

Backend: existing Dacapo/SEAL HEVM CPU. Unchanged SEAL4.0.0, N=32768,
fourteen 60-bit moduli, tc128, atol=1e-5 and rtol=1e-4. No GPU or bootstrap claim.
All six temporary key cleanups completed: 4,073,880,060 bytes (~3.79 GiB).
Original random keys are not recoverable; fresh equivalent test keys can be
generated. Reports, source, original arrays, IR, HEVM/CST and decryptions remain.

The original numeric-golden producer was function_construction SHA256
`d2b6dd9102b40645784d3b1dc09eaaaa7d34d5600c578afe65483fa4e24c3490`
and public_numeric SHA256
`e1529a9f3db3a9ce38bb75107b7ad70a4e6d143b4f18a6d976ebc3c00085ef8c`.
The subsequent augmented-array rejection and explicit empty-array dtype fixes
are separately regression-tested. Audits retain those historical fingerprints
and reproduce every original expansion/derived manifest with current code;
historical encrypted executions are not relabeled as new executions.

## Tests and reproduction

`test_public_numbers.py` includes scalar arithmetic, 175 combinations against
actual pinned NumPy operations, multidimensional slicing, reshape, original
constant integrity, name collision/dedup, encoding/resource bounds and capability
rejections. An initial NumPy test adapter discarded empty-array trailing shape
through `tolist()` in its expected-failure path; fixing the adapter resolved that
failure without changing the broadcasting algorithm or numerical tolerance.
Empty-array dtype required its own implementation fix and regression test.

Final pinned-environment audit (nine semantic groups plus batch routing and
manifest checks): **170/170 passed, no skips**. This includes replaying the
historical expansions and derived manifests against current code.
Full offline regression: **590 tests, 476 passed, 114 conditional skips, no
failures**. Skip conditions do not count as verified semantic coverage. The
argparse rejections and mocked provider-retry messages are expected negative
tests, not actual paid calls. AST/whitespace checks passed for all 21 touched
Python files. No dependencies, system configuration, GPU code or Git history
were changed by this extension.

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_public_number_goldens.py
```

Set `POSEIDON_NUMBER_REPORT` to the report above in the pinned Nix environment
to enable its actual-artifact audit. The eight preceding groups keep their
existing documented evidence environment variables.

## Paid Agent validation: prepared, NOT started

The user authorized paid tests within the full-DSL goal. A proposed new cohort
uses the existing `cases/advanced-agent-12-manifest.json` (SHA256
`a398e1ec1ade2d1639d793b41cf8fe8523df4cb3a8b0c901b0b0e869cb4d5678`):
four width-varying MLPs, six convolution variants, and two explicit polynomial
graphs. It uses the v14 contract, DeepSeek `deepseek-flash`, high reasoning,
384000 output-token cap, 1200s request deadline, 10 API workers and 2 native
workers, with at most 3 feedback repairs and 3 transport retries.

The 6478 proxy completed verified TLS to api.deepseek.com. The unauthenticated
HEAD returned HTTP401, proving connectivity, NOT credential validity.

**Execution-environment approval rejected process creation for the paid batch**,
requiring explicit approval for external disclosure of this specific cohort's
model descriptions/public weights, DSL rules and diagnostic feedback. No batch
process or paid request was started. Do not retry via a workaround. Ask for that
specific external-payload approval before launching this cohort.

The batch runner now forwards `--public-numbers`, records it, and audits the
actual request task. Cross-contract retries are refused: use a fresh batch so
new prompts are not mislabeled as repair results from old prompts. After launch,
measure actual generated-syntax use separately; enabling a grammar does not
prove the Agent used every rule.
