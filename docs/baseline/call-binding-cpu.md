# Definition-time defaults and function argument binding

## Confirmed scope

`run_candidate.py --call-binding` selects request `hecate-function-synthesis-v10`
and AST contract `hecate-function-v9`, including the preceding closure features.
It works through prepare/manual/replay/live paths without changing the default
or any old request's rules. The new syntax has manual real-FHE evidence, not a
new online Agent cohort or a formal equivalence proof for all programs/inputs.

This is relevant to actual upstream construction code: in pinned Dacapo,
`python/poly/poly/Func.py` defines `HE_Linear(..., p=1.0, scale=1.0)` and
`HE_ReshapeLinear(..., p=1.0, scale=1.0, reshape={})`. Supporting the language's
parameter binding is necessary, but does **not** make those entire helpers
available: their tensor computations, public numerical expressions, packing
and MPCB construction still need separate verification.

## Representation and semantics

The trusted AST interpreter stores each construction function's lexical frame,
positional defaults and keyword-only defaults. It evaluates defaults once, at
definition time, before binding the new function name. Calls first evaluate
arguments, then `construction_calls.bind` maps symbolic values to fresh locals.
No generated Python function is executed. The resulting straight-line program
is rechecked and sent to the existing Hecate/Dacapo/SEAL CPU chain.

| Rule | Implemented behavior |
|---|---|
| Positional-only `/` | Cannot be filled by keyword; the same spelling may separately occur in **kwargs when declared. |
| Positional-or-keyword | Can be filled either way; assigning it twice fails. |
| Keyword-only `*` | Must be passed by keyword unless it has a saved default. |
| Default evaluation | Positional defaults then keyword-only defaults, left-to-right, once per executed definition. |
| Definition binding | Defaults can see the prior binding of a redefined nested function. Top-level helpers are bound in source order. |
| Default versus closure | A saved symbolic default keeps the definition-time value; a closure cell observes later rebinding. |
| Mutable defaults | Repeated calls share the saved list/dict; separate factory definitions create separate defaults. |
| Argument evaluation | Callable is resolved first; positional/star expressions evaluate before keyword expressions. Keywords preserve expression order, not parameter order. |
| *args / **kwargs | Tuple of remaining positionals and a fresh string-keyed dict of remaining keywords; contained values retain aliases. |
| Forwarding | Bounded list/tuple *expansion and dict **expansion; duplicates fail instead of being overwritten. |
| Mapping support | Explicit string-keyed dict literals, indexing, item writes and len; cycles and oversized mappings rejected. |

The existing bounds remain (16 parameters, 128 expanded positional/keyword
arguments, 128 container entries, 4096 construction steps, 16 call frames,
256 emitted cipher operations). String keys are bounded public data, never
ciphertext arithmetic. No new filesystem/network/process capability is enabled.
Plain constants remain immutable opaque operands, not generally indexable arrays.

These helpers still obey the current symbolic expression subset. General
floating-point/public-array computation, lambda, comprehension, dict iteration
and methods, mapping-unpacking literals and general iterable unpacking remain
pending. The golden decorated entry function retains its exact harness ABI;
new parameter kinds apply to construction helpers, not arbitrary encrypted I/O.

## Tests and actual encrypted execution

`test_construction_calls` has 17 semantic/negative tests, one independent plain
golden preflight and one actual-artifact audit. A handwritten trusted Python
function supplies an independent binding oracle for **192 combinations** of
positional counts and keyword sets. This is one parameterized unit test, not
192 FHE executions or 192 model API requests.

Full run (no API):

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2700s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_call_binding_goldens.py
```

Original report, retained **failed**:

- `/home/lhy/poseidon-work/results/call-binding-goldens-bogj5d6d/report.json`
- SHA256 `b7ab4afe7a3dcad15f70774481cd3ba5e68a56a6704e23ce18c47f2566b825f1`

One deliberate wrong-default fixture accidentally generated `-x+x`. It parsed,
traced and compiled, but SEAL aborted with `result ciphertext is transparent`:
failure layer `seal_runtime`, exit 134, **no completed encrypted result**. Its
original source, normalized source, IR, HEVM/CST and execute.log remain in
`/home/lhy/poseidon-work/results/candidate-replay-acqs6a3b`.

That failure is not numerical-comparison success. No transparent-ciphertext
check was disabled and no fresh-zero/decrypt-reencrypt substitute was added.
The paired positive/negative fixtures now rebind outer `value=x` instead of
`value=-x`: this preserves the default-versus-closure distinction without exact
cancellation. Their fixed reference remains `1.5*x+0.375` and was not changed.

Only that pair was rerun:

```bash
timeout -k 3s 700s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_call_binding_goldens.py --snapshot-only
```

- `/home/lhy/poseidon-work/results/call-binding-goldens-q4cur850/report.json`
- SHA256 `885c9047a4a08d94adb5a74e35341bc7f2cd5182d40209d9c3ae39039826aa5d`

The selected final eight roles all match expectations: **4 correct programs,
4 numerical counterexamples, 32 input groups, 112 compared output values**.
There were ten total attempts including the rerun, not an initial 8/8 success.

| Program | Expected result | MAE | Maximum absolute error |
|---|---|---:|---:|
| Linear(4,2), captured default row/rotation tuple, positional and keyword calls | Pass | 3.0783655e-9 | 5.9373463e-9 |
| Default snapshot unaffected by outer rebinding (rerun) | Pass | 6.5825376e-9 | 1.4023028e-8 |
| Mutable default list shared across calls | Pass | 4.7968586e-9 | 1.7764313e-8 |
| Positional-only / keyword-only / vararg / kwarg forwarding | Pass | 5.0784608e-9 | 2.2549508e-8 |
| Wrong default rotation tuple | Numerical rejection | 0.33686968 | 0.62500000 |
| Wrong closure read instead of saved default (rerun) | Numerical rejection | 0.25881523 | 0.50000002 |
| Resetting mutable default list on every call | Numerical rejection | 1.03526089 | 2.00000000 |
| Wrong subtraction of keyword bias | Numerical rejection | 0.74999999 | 0.75000001 |

SEAL4.0.0, N=32768, fourteen 60-bit moduli, tc128, compiler configuration and
elementwise `atol=1e-5, rtol=1e-4` were unchanged. Per-element outputs/errors,
nonzero-reference relative error and cosine similarity are retained in reports.
These are real SEAL CPU runs, not Poseidon GPU, bootstrap or simulation results.

## Evidence audit and retention

Set `POSEIDON_CALL_BINDING_REPORT` to a JSON array containing the original and
rerun report paths in that order. The audit replaces only matching **golden
program identities**, not every entry sharing a model name. It checks all eight
selected sources, current generator hashes, frozen inputs/request/reference,
IR/artifact hashes, security settings and independently recomputed errors.
Original failed reports are never rewritten or treated as passed.

Historical closure evidence retains its actual pre-parameter-binding engine
and lexical-scope hashes. The audit accepts those specific historical producers
and separately reproduces exact old-mode expansion with today's code; it does
not claim that old artifacts were produced by the new engine.

All five semantic groups with real saved evidence: **84/84 passed, zero skips**.
Full offline suite: **521 tests, 412 passed, 109 conditional skips, no failures**.
Expected invalid-argument diagnostics and mocked provider retries are test
output, not live API/network failures. Skipped conditions are not passes.
All ten temporary test-key directories were cleaned by the existing per-run
policy: **6,789,800,100 bytes, about 6.32 GiB**. Random test keys are regenerable;
source, arrays, diagnostics and compiler/decryption artifacts remain. No unrelated
files were deleted. No install, sudo, paid API, commit, push or PR occurred.

The full DSL goal is still active: public numerical arrays, broader control and
construction syntax, general packing, all actual upstream helpers, IR calls,
true bootstrap/upscale and online Agent coverage are not completed by this work.
