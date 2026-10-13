# Integer Rescale compiler review

Compiler commit d5ed984, pushed on integration/multi. No HEFT or RescaleAttrs
changes. Existing user README/docs changes are preserved.

- `MaterializePhysicalLevels.cpp`: reject negative logical scale reduction;
  require dropped nominal bits >= logical reduction + 30; retain physical drop
  capability bound; require target scale < remaining nominal modulus sum.
- `check.py` and lazy physical fixtures: Q50/factor5 maps logical9→8→0 to
  physical49→44→4; 3 logical levels map to a 15-prime drop. Negative checks cover
  factor4 with no alignment space, only29 extra bits, target scale equal to
  remaining nominal modulus bits, level underflow and 15-prime drop vs max14.
- `docs/runtime-plan/v4.md`: protocol rationale; actual-prime precision must still
  be verified by the backend. Other pending changes in this file belong to root.
- Native compiler build succeeded; `ctest` passed 10/10 in 2.52s. Evidence:
  `compiler-integer-rescale-build.log`, `compiler-integer-rescale-tests.log`.
- Isolated Q50 profiles and small q graph are in
  `build-waterline-sync/qwen3-q50-rescale-candidate`. Old source bytecode and
  fixed dense fixture reused, synthetic q2048×1024/8192 slots/dirty tail;
  compile exit0, Boot0, Prepare1025, groups8353, Rescale1025, inputlevel49 and
  outputlevel14. Evidence: `compiler-q50-q-proj.log`, `candidate-manifest.json`.
- New GPU API q graph passed all8192 complex slots, maxabs9.90988e-10;
  see `q50-q-proj-corrected-report.json`. No full-model/pretrained
  numerical claim. Old Q40 API q/down runs failed original tolerance with the
  proven 1.090078312498109 amplitude bug; they remain failure evidence.
- Q50 physical latency rows40..49 inherit old last-row estimates; unmeasured.
  Logical cost rows unchanged. Actual persistent keys, workspace and memory
  limits must be reevaluated. Context is numerical-only, no security claim.
