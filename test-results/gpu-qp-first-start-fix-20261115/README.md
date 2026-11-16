# QP first-start GPU preparation validation

The detailed methodology, baseline, limitations and reproduction instructions are
in `docs/gpu-qp-first-start-fix-20261115.md`.

Tracked compact evidence:

- `summary.json`: A800 public-constant fingerprints and per-stage startup A/B
  measurements across two reference, three optimized 4-worker and three optimized
  6-worker runs, plus a single full-CRT oracle run and a single 1-worker run.
- `results/oracle-t4/stderr.log`: direct all-Q/P word equality checks of all 316
  actual manual Boot profile diagonals; this is a validation run, not a timing sample.
- `results/exact-regression-final/stdout.log`: 24-configuration/264-case GPU exact
  oracle check on A800.
- `merge-review/local-gpu-regression.*`: follow-up tests of fault injection, wide
  lazy verification and concurrent callers on RTX4060.
- `merge-review/bootstrap-{reference,optimized}.log` and
  `merge-review/bootstrap-cross-compare.json`: full GPU Boot/decrypt smoke with
  N4096, Q50/P8, 2048 complex slots; this is not the N65536 ResNet20 Boot.
- `merge-review/a800-final-*`: A800 revalidation of final source after concurrency
  hardening, including a new full-CRT bytewise oracle for all 316 diagonals,
  fault-injection coverage, GPU exact regression, and one-sample latency A/B.

These artifacts document the original optimization experiment. Later pre-merge
concurrency/thread-failure and wide-lazy-oracle hardening is validated separately;
see the report for its final code-specific evidence and performance. The rest of
this directory remains deliberately gitignored, including full launch/environment
logs, public inventories, generated binaries/source bundles and manifests. Complete
raw experiment directories are retained locally in the main worktree and on the
experiment server, and should not be confused with committed artifacts.
