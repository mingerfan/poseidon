# Compact BERT / Cerium alignment work

This is a new experimental path, not a claim that Poseidon now executes the
Cerium compiler's exact nonlinear/bootstrap implementation. See
[`CERIUM_ALIGNMENT.md`](CERIUM_ALIGNMENT.md) for the audited original system.

## Workload and layout

- BERT-base encoder: 128 tokens, hidden768, heads12×64, FFN3072, independent
  dense weights and the same seed/input as the earlier benchmark.
- `cerium_packing.py`: hidden **3 CT × [128,256]**, FFN12 CT. Q/K/V each use
  six two-head/gapped CTs in Cerium's `(0,2,1,3,...)` head order.
- QK directly produces **six full cyclic score CTs**; AV uses the corresponding
  BSGS value-diagonal/within-head rotations. It does not evaluate 384 repeated
  score ciphertexts. Tiny independent score/AV/full-block oracles cover this.
- Exp4 with encrypted mean subtraction, three reciprocal Newton steps, GELU4,
  true mean/variance post-LayerNorm, both residuals and native Boot remain.
  These are **our previous approximations**, not Cerium's sign/max, exp6,
  isqrt6 or piecewise GELU. Numerical accuracy is not an acceptance criterion.

## Execution and physical scheduling

`generate_aligned.py` → `CompactGraph` → `MultiPrimeLowering` → verified V7
RuntimePlan → standard Runtime / Poseidon GPU API. Ordinary nonlinearity,
attention and alignment use standard compute instructions. Dense Linear uses
an explicitly named `bert.compact_linear` prepared ApiCall; its true GPU BSGS
implementation is in `compact_native/`. This is **not** an automatically
planned Hecate/DaCapo full-size compilation result.

The prepared engine preserves every float32 dense weight and bias. It uses
exact, nonquantized periodic NTT/RNS storage, shared hoisted baby rotations,
one fused Q MAC over all input tiles per giant group, real rotations, one
multi-prime rescale per output tile and real bias addition. Macro counts must
not be presented as all scalar work: the default one-block matrices contain
**55,188 plaintext products** inside 27 calls.

Candidate contexts use N65536, the actual Cerium prime-pool prefix Q52 and
suffix P11 (or Q49/P13), main secret weight32768 and ephemeral weight32.
Working canonical scales are about56 bits; each ordinary product drops two
physical ~28-bit primes. Scale matching always uses actual multiplication by
encoded one and real Rescale, never metadata-only rewriting. Each encoder
closes at its input Q13/canonical scale, retaining its final three refreshes.

The native profile candidate uses d30, DA3, K16, two-prime logical rescaling,
compressed DFT diagonals and a two-prime ground modulus. The new default
message ratio256 corresponds to epsilon2^-8. **Early archived tests used
ratio32** and must be reproduced with `--bootstrap-ratio 32`. DFT grouping,
Hanki coefficients, dynamic special-P selection and complex vec2 pairing are
not declared equivalent to Cerium.

For the early Q52/P11/ratio32 candidate, the refined CPU schedule has **29
planned single-stream native Boots**, ~26.5k standard/opaque compute calls,
and 55,188 internal Linear plaintext products. Fewer Boots than Cerium do not
establish an equal-nonlinearity performance comparison. The previous
prototype used55 Boots; these are plan counts, not measured GPU execution.

## Measurements available so far (A800)

All tests use the assigned GPU UUID499236d3-b46c-5d25-4a22-530d47ed5112,
serially, without other benchmark jobs.

- **Original unoptimized prefix128:** one complete cold iteration, no warmup:
  online **3087.960906s**, Runtime wall3132.518049s; actual12,809,984 compute
  operations and1,920 native Boots. All4,194,304 physical output slots finite,
  max reference error6.97865 (not an accuracy pass). This cold sample must not
  be called a warm steady-state median. The first timed-out attempt remains
  archived separately and is not used to infer this latency.
- **Full FFN-up, strict preparation:** T128,768→3072,3→12 CT,1 warmup+3
  samples. Runtime online median **140.879ms**, synchronized callback median
  **140.634ms**, runtime wall median146.526ms;0 Boot. All393,216 output
  slots finite, max reference error3.77e-10. Actual compressed PT197,812,224B
  versus full equivalent24,121,442,304B, period512/no fallback. Preparation
  took146.074s and is excluded online. Versus previous4.617s online this is
  a32.77× same-shape system improvement; HE parameters/packing changed.
  Cerium's281.463ms program.run wall is about2× this online result, **not an
  equal-parameter implementation speedup**.
- **Fused-engine diagnostic only:** genuine 256×256 submatrix, 128 tokens,
  N65536, two-prime Rescale, working56, one warmup+three samples. Median
  **12.289ms**, max absolute error5.95e-8, exact period512 and zero fallback.
  This is not a full FFN or encoder measurement.
- **One-native-Boot diagnostic:** Q52/P11/ratio32/d30/DA3, one warmup+three
  samples. Synchronized online median **108.678ms**, including Runtime and
  post-Boot MulCP(1)+two-prime normalization, not an isolated kernel time.
  All physical slots finite, but max reference error~0.974: **not an accuracy
  pass**. Profile construction alone took~241s and is excluded online.
- The ratio256/d30/DA3/K16/h32 profile has now been genuinely built and
  uploaded, outputL20/scale56, setup243.874s; the probe itself ran zero Boots.
  An opt-in periodic setup path has passed separate small GPU arithmetic
  validation; strict remains the default and full/fast artifacts are separate.
- **Full compact encoder:** ratio256/periodic preparation,128 tokens,one
  complete block,1 warmup+3 samples. Online median **9.839523s**, Runtime
  wall median11.617528s,29 genuine native Boots/run (87 measured/116 total).
  Input/output3 CT at Q13/canonical scale; all98,304 output slots finite,
  but max reference error6.41065: **not an accuracy pass**. This is5.302×
  the previous52.169807s configuration; native Cerium remains6.059s, so we
  are still1.624× slower. Different HE parameters/approximations, not a pure
  ablation. See `aligned-encoder/report.json` and the top-level result README.

Artifacts: `test-results/bert-a800-20261115/`, notably `unoptimized/`,
`aligned-native-build/`, `native-boot-s56-probe-v2/`, `profile-q52/`, and
`compact-cyclic-q52-selected-v2/`. This directory is gitignored; archive it
separately. The 5.06GB raw-prefix plan stays on the remote machine. The second
cold measurement completed successfully; full evidence is under
`unoptimized/attempt2/completed/`. GPU0MiB/0% and no compute processes were
independently verified at06:50:20 UTC; no further GPU work is queued.

## Reproduction skeleton

```bash
python3 -m unittest discover -s bench/bert_dsl -p 'test_*.py'
python3 bench/bert_dsl/audit_cerium.py \
  --check test-results/bert-a800-20261115/cerium-audit.json

# CPU graph/schedule only, not a GPU measurement:
python3 bench/bert_dsl/generate_aligned.py --output-dir NEW_PLAN \
  --candidate q52-p11 --scope encoder --bootstrap-ratio 32 \
  --profile-report test-results/bert-a800-20261115/profile-q52/report.json \
  --profile-fixture test-results/bert-a800-20261115/profile-q52/fixture.json

# New, complete-field ratio256 probe:
python3 bench/bert_dsl/generate_aligned.py --output-dir NEW_FAST_PLAN \
  --candidate q52-p11 --scope encoder --periodic-encoding \
  --profile-report test-results/bert-a800-20261115/compact-native-build/fast-validation/profile-q52-r256.json
```

Build `aligned_native/CMakeLists.txt` against the matching immutable CPU/GPU,
Runtime and API archives. It links `compact_native/` and cuFFT; it does not
replace the established baseline runner. Use the previous managed allocator,
Host-only ordinary plaintext cache and4GiB native Boot workspace. The latest
fixture explicitly adds direct prepared-Linear rotation keys while ordinary
Runtime rotations can retain the binary policy; the actual key basis is
reported. This preserves hoisting without generating every ordinary rotation
at the full context level. For a prepared-only FFN plan the
standard plaintext bundle is deliberately absent; do not pass `--bundle`.
Do not start GPU work without coordinating the exclusive window.
