# Multi-input golden v3: verified compiler/runtime ABI, Agent integration pending

Subsequent increment: [the public schema-3 and isolated candidate path are now
integrated](schema3-multi-input-agent-path.md). The pending-integration statements
below describe this earlier golden-only milestone. Live v3 LLM generation and
Poseidon GPU validation remain pending.

## Confirmed outcome

Eight manually authored Hecate programs with two, three or four independently
encrypted inputs passed real tracing, Dacapo compilation, HEVM/CST inspection,
actual SEAL HEVM CPU execution and independent numerical comparison. An ordered
subtraction counterexample deliberately swaps its two arguments; it compiles and
executes but fails numerical comparison, as expected.

This is NOT an Agent generation result, a Poseidon GPU result, arbitrary-shape
support or completion of the overall goal. Public JSON model-loader and Agent
request integration remain to be implemented. All existing v0/v1/v2 requests
still require a single encrypted input.

## Why the change is needed

Two parameters in source must remain two independent inputs through every
representation. Otherwise, code could silently calculate x-x instead of x-y,
or encrypt one input twice. In the pinned upstream frontend,
`third_party/dacapo/python/hecate/hecate/expr.py`, `Func` splits the decorator's
comma-separated parameter description and creates that many input expressions.
Thus the correct declaration is `@hc.func("c,c")`, NOT `@hc.func("cc")`.

In `third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp`, the C ABI encrypts input i to
register i using that argument's scale and level. The existing runtime supports
this; no new FHE backend was added.

```python
@hc.func("c,c")
def golden(x, y):
    return x - y
```

Current bounded layout: each parameter is one separately encrypted vector with
four logical values repeated across 16384 slots. This is not four shards, four
devices or four slots masquerading as four ciphertexts. The v3 checker accepts
2..4 named ciphertext parameters and 1..4 ciphertext results. These fixtures
exercise one result or two results; not every allowed result combination is
proven by this suite.

## Changed boundaries

- `hecate_contract.py`: v3 requires an explicit ordered input-name manifest,
  exactly matching function arguments and decorator. Reject duplicate names,
  default/variadic parameters, plaintext input declarations, constant shadowing
  and rebinding. Earlier contract signatures remain unchanged.
- `candidate_trace.py`: trusted AST construction can bind a checked dictionary
  of distinct Hecate input objects. No candidate Python is evaluated or executed.
  The existing live Agent tracing entry still uses its single-input contract.
- `seal_artifact_gate.py`: expected input count must be explicitly supplied.
  HEVM arity, argument metadata and initialized register tracking are checked;
  callers that omit the new argument still require exactly one input.
- `seal_cpu_golden.py`: validates the input array shape before loading native
  runtime code, encrypts each argument separately, checks each input's physical
  level/scale/polynomial metadata and preserves existing single-input reporting.
- `multi_input_fixtures.py`: independent Python scalar reference and a separate
  trusted Torch implementation. Neither reads translated or decrypted answers.
- `run_multi_input_goldens.py`: no provider calls or key discovery; golden-only
  tracing, bounded child processes, frozen source/input/reference/artifact hashes,
  real encrypted execution and per-layer failure reporting.

The trusted golden runner is not the sandboxed live Agent pipeline. It is a
prerequisite for safely wiring that pipeline's public multi-input model schema,
request manifest and tracing sandbox. This distinction must remain explicit.

## Actual results

Evidence root:
`/home/lhy/poseidon-work/results/multi-input-golden-q_a8yain`

`report.json` records source/binary/profile hashes and all nine attempts.
Each `case-NNN` includes `golden.py`, `manifest.json`, frozen `arrays.npz`, trace
and compile logs, Earth/CKKS IR, HEVM/CST, `decrypted.npy` and `execution.json`.
Do not share the whole directory because `private-keys` contains FHE secret keys.

| Manual case | Ciphertext inputs | MAE | Max absolute error | Expected result |
|---|---:|---:|---:|---|
| dual_add | 2 | 2.540983055e-9 | 1.042306064e-8 | pass |
| ordered_subtract | 2 | 2.851084269e-9 | 9.855547933e-9 | pass |
| dual_product | 2 | 2.815480847e-9 | 1.620355350e-8 | pass |
| weighted_merge | 2 | 3.296374108e-9 | 1.119003989e-8 | pass |
| dot_difference | 2 | 1.206801179e-8 | 2.215396358e-8 | pass |
| triple_merge | 3 | 3.018394986e-9 | 1.201624888e-8 | pass |
| quad_merge | 4 | 4.782664310e-9 | 1.463159860e-8 | pass |
| dual_outputs | 2 | 1.957847943e-9 | 4.703638701e-9 | pass |
| reversed ordered_subtract | 2 | 1.275180848 | 3.999999997 | numerical rejection |

Eight correct cases x four input sets = 32 input tuples, 76 separate input
encryptions, 108 compared output values. The negative case adds four tuples and
16 output comparisons, excluded from positive correctness denominators.

Inputs cover zero, asymmetric signed vectors, fixed-seed random vectors, and
declared [-1,1] boundaries. Symmetric input alone cannot detect exchanged
arguments. `dot_difference` includes genuine weighted cross-slot reduction with
rotations +1 and +2; `dual_product` uses two different input ciphertexts, not a
square rewritten to look like a binary-input operation.

Frozen tolerance remains `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
N=32768, SEAL tc128, fourteen 60-bit key moduli and the stock CPU compiler profile
are unchanged. All observed input/output ciphertexts have two polynomials;
real key-file preflight is performed, and no bootstrap is executed.

These eight test computations are not automatically eight new model families.
Operator and family doubling are still separate incomplete requirements.

## Reproduction and regression

Use existing isolated dependencies from the WSL source root:

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_multi_input_goldens.py

# Recheck saved evidence without generating new keys or requesting an LLM:
POSEIDON_MULTI_INPUT_RESULTS=/home/lhy/poseidon-work/results/multi-input-golden-q_a8yain \
  python3 -m unittest discover -s scripts/baseline -p test_multi_input.py -v

# Fixed Torch environment also cross-checks all independent references:
python3 scripts/baseline/run_model_batch.py --unit-tests
```

Full ordinary-interpreter regression with all historical and current evidence:
234 discovered, 218 passed, 16 skipped. Separately, fixed Torch regression:
36 discovered, 35 passed, one evidence-dependent test skipped because the pure
Nix launcher does not inherit arbitrary environment variables. The same real
evidence test passed in the ordinary-interpreter regression with an explicit
results environment variable. This is not a missing runtime or numerical failure.

## Next integration gate

Extend the data-only user model schema with explicit ordered inputs and per-input
shapes; propagate them through Torch/FX, Agent public request, candidate validation,
sandboxed tracing, artifact gate and runtime. Add live-generation evidence only
after deterministic and manual paths pass, and retain separate rule/Agent rates.
Larger packing, broader Linear/MLP, Conv/Pool, approximation error decomposition,
doubled operator/model coverage and Poseidon GPU remain open. No installation,
system change, API call, commit, push or PR occurred for this increment.
