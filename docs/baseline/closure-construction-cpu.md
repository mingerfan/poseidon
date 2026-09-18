# Nested functions and lexical closures: real CPU evidence

## Scope and representation

`run_candidate.py --closures` selects request `hecate-function-synthesis-v9`
and AST contract `hecate-function-v8`. Earlier versions and default selection
are unchanged. This extends Python-style **graph construction**, not the
unresolved multi-function Earth/CKKS runtime call mechanism.

The original source is parsed and checked, interpreted using symbolic operands
and lexical frames, and expanded to straight-line arithmetic. The expansion is
checked again before Hecate tracing. Dacapo then compiles to CKKS/HEVM/CST and
the existing SEAL CPU path performs real encryption, evaluation and decryption.
No generated Python function is executed. Static `compile(AST, ..., 'exec')`
inside `analyze_scopes` only checks Python symbol-table legality; its code object
is discarded, never executed. It catches illegal nonlocal declarations even
inside helpers that are never called. Candidate AST limits and allowlists run
before this static check. The existing isolated tracing boundary is retained.

## Preserved closure semantics

| Requirement | Implementation / test |
|---|---|
| Lexical capture | A nested function stores the defining frame, not its caller's frame. |
| Late binding | Reading a free variable observes its current outer binding, not a definition-time snapshot. |
| Escaping closure | Captured frames survive their outer call; two factory calls have distinct frames. |
| Sibling closures | Functions created in the same invocation share outer cells. |
| Loop capture | Definitions created in a loop share that invocation's loop variable and observe its latest value. |
| nonlocal | Writes the nearest enclosing function binding, including across an intermediate scope; module globals are not writable. |
| Static local analysis | A nested function's assignments do not become its parent's locals. Uninitialized local/free cells fail rather than falling back to globals. |
| Mutable captured lists | Item mutations remain visible through aliases without requiring nonlocal; rebinding is separate. |
| Ciphertext values | Symbolic ciphertext IDs are immutable; rebinding changes a cell, not previously saved values. |
| Nested recursion | Captured self-binding permits bounded recursive construction. Existing step/call/depth gates still apply. |

At most 17 total function declarations (including golden), 128 instantiated
nested functions, 128 helper calls, 16 active helper frames and 4096 construction
steps. Existing source/AST/container/operation bounds, fixed period-four packing,
provisioned rotation steps and immutable public registry remain enforced.
Harness-owned bindings remain protected. An unused helper cannot hide arbitrary
attribute access, imports, external calls or global mutation.

No defaults, keyword/variadic parameters, lambda, comprehensions, generalized
public array computation, arbitrary shape/packing or ciphertext control flow is
claimed here. Complete upstream high-level helpers, IR calls, true bootstrap
and upscale remain incomplete and in the full goal's scope.

## Actual run

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 2100s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_closure_goldens.py
```

Batch: `/home/lhy/poseidon-work/results/closure-goldens-fk14qbbf/report.json`

SHA256: `147e7edec3a4e1a8a8b8074dffb7652d8b7a306765a9d384d6e93042329309ad`

All six roles matched their expectations, without reruns. Three positive
programs and three deliberate counterexamples each used four fixed input
groups: **24 groups / 80 compared values** total. Correctness is judged against
the unchanged model reference, never against decrypted intermediate values.

| Manual program | Expected result | MAE | Maximum absolute error |
|---|---|---:|---:|
| Linear(4,2), two independently captured row weights and real rotation reduction | Pass | 8.9931294e-10 | 2.9674055e-9 |
| Shared state: nonlocal scaler and reader survive factory return; old alias retained | Pass | 5.1486502e-9 | 1.3491041e-8 |
| Late-bound outer value after reassignment | Pass | 5.3541724e-9 | 1.5370192e-8 |
| Linear missing the second rotation | Numerical rejection | 0.33686968 | 0.62500000 |
| Scaler accidentally creates a local instead of updating the captured cell | Numerical rejection | 0.25881522 | 0.50000000 |
| Reader captures a separate old-value variable rather than the updated variable | Numerical rejection | 0.25881522 | 0.50000001 |

The latter two correct programs implement `1.5*x + 0.375`. Linear uses fixed
rows `[.5,-.25,.125,.75]` and `[-.375,.25,.5,-.125]`. Plain preflight checks these
independent formulas before entering native execution; preflight alone is not
counted as FHE success. All counterexamples parsed, traced, compiled and ran
under encryption before failing only at `numerical_comparison`.

SEAL 4.0.0, polynomial degree 32768, fourteen 60-bit moduli, tc128 security and
existing compiler scale settings were retained. The elementwise criterion is
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`; it was not relaxed.
Reports retain per-element outputs/errors, nonzero-reference relative error and
cosine similarity. There was no bootstrap, GPU execution or paid model call.
This batch establishes manual DSL execution evidence, not online Agent success.

## Verification and historical provenance

`test_closure_construction` has 15 symbolic/security tests, one independent
plain-golden test and one opt-in real-artifact audit. Set
`POSEIDON_CLOSURE_REPORT` to the report above and run it in the existing pinned
Nix/Python environment. The audit checks actual generator hashes, immutable
request/model/input hashes, original source, normalized source, construction
metadata, compiler artifacts, security configuration, real-encryption flags,
independently recomputed reference/metrics and key-directory removal.

Multiple closure programs intentionally share a reference model. The audit
matches the full ordered plan and original source, not a dictionary keyed only
by model name, which would silently discard these distinct programs.

Combined closure/function/public-construction/arithmetic tests with their saved
real evidence: **65/65 passed, no skips**. General offline suite:
**502 tests, 394 passed, 108 conditional skips, no failures**. Skips are not
passes. The suite's expected invalid-CLI and mock-provider retry messages are
negative-test output, not live provider failures.

The first closure security run found that an arbitrary attribute in an unused
helper escaped static rejection. New-contract attribute checks now reject it;
the targeted regression passed. No external access was executed.

Older request-v8 evidence was produced by function_construction.py SHA256
`8eea9b31450d19c9ff5f3bccaeca6d3f8e881bdac8a92e7b73244f168e8e1155`.
That exact historical producer is explicitly recognized by its audit, without
changing old reports or claiming execution by the new engine. The audit also
recomputes the old-mode expansion using current code and checks exact source
and metadata equality. New closure evidence requires the current engine and
lexical_scope.py hashes instead.

Six temporary private-key directories were removed by the existing per-run
cleanup: 6 x 678,980,010 = **4,073,880,060 bytes (about 3.79 GiB)**. Random test
keys are regenerable; source, IR, artifacts, arrays and decrypted results were
retained. No unrelated files were deleted.

WSL `git diff --check` passed; fixed Dacapo gitlink is unchanged. Windows Git's
submodule shell failed because basename/sed/git-sh-setup were unavailable in
that host invocation; WSL Git confirmed the state without any system repair.
No install, sudo, commit, push or PR was performed.
