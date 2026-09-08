# EvalMod degree-30 + three double-angle experiment

This directory isolates the low-degree EvalMod experiment from the normal
bootstrap test path. It has its own CMake build tree and result logs.

The candidate uses `K=16`, requested polynomial degree 30, and three
double-angle steps. `K=16` is required because the discrete approximation
starts with `2*K-1` interpolation nodes; retaining the production `K=25`
would still generate an approximately degree-59 polynomial.

Run the EvalMod screening test:

```bash
./run.sh evalmod
```

Run the full one-warmup/one-measurement bootstrap validation:

```bash
./run.sh full
```

Generated build files and logs stay below this experiment directory and must
not be used as the production baseline.

## Phase-1 result (N=65536, 2026-08-13)

The candidate is **not approved as a production default**.

- The active polynomial was degree 30 with 4 leaf blocks, 3 combines, and
  three double-angle steps.
- EvalMod decreased from the production baseline of about 74.61 ms to
  55.17 ms (about 26.1%).
- Full bootstrap decreased from about 148.96 ms to 128.39 ms (about 13.8%).
- Staged EvalMod CPU/GPU agreement passed with a maximum error of about
  7.9e-11, after correcting the experimental dynamic leaf-level planner.
- SlotToCoeff CPU/GPU error increased to about 9.18e-3 and the full
  GPU/source error increased to about 1.90e-2. The production path is about
  4.6e-3 to 6e-3 under the same known N=65536 precision limitation.
- The candidate also leaves q=14 after EvalMod instead of the production
  q=15, so it consumes one additional physical Q modulus.

The useful conclusion is that reducing the polynomial graph has substantial
performance potential, but degree/K/double-angle selection and the
EvalMod-to-S2C scale contract must be optimized together. This experimental
result is intentionally not copied into the production optimization log.
