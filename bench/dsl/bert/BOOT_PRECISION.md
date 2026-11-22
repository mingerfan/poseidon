# Opt-in BERT Boot precision

Build `bert_native_runner` with the matching Poseidon GPU library using [the
normal SOURCE build](BUILDING.md). No alternate objects or build flags are
needed. Add `--boot-precision` to `generate_aligned.py` to regenerate a cyclic,
selective q52-p11/r256 encoder plan at input L12. For example:

```sh
python3 bench/dsl/bert/generate_aligned.py --output-dir bert-precise --layers 12 \
  --boot-precision --prepared-qk --qk-pack-inputs --prepared-av --av-pack-inputs \
  --algebra-rewrite --shared-rotation-rescale
```

This generates a distinct native Boot profile and inserts real public
multiplications around attention denominator refreshes: `D/T → Boot → ×T`,
where `T` is the model's token count. The profile preserves the actual
ModRaise period, folds the output gain into the S2C diagonals, and retains
EvalMod working precision for leading quotients. Existing plans and other
profiles keep their original behavior. Do not enable just the profile flag on
an old plan: Boot precision alone does not fix out-of-range denominators.

When using `--profile-report`, generate the probe fixture with
`--profile-fixture-only --boot-precision` first and use a report from the
matching build and precision mode.
