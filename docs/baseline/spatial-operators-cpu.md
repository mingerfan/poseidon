# Bounded Conv/AvgPool: real CPU candidate-path correctness

Extension: [grouped and dilated Conv](grouped-dilated-conv-cpu.md) now has six
additional rule/manual CPU cases and two encrypted counterexamples. The nine
original experiments below are preserved separately, not relabeled as the new run.

## Confirmed result

Nine spatial/composite manual programs passed Hecate tracing, Dacapo compilation,
HEVM/CST inspection, sandboxed real SEAL CPU execution and independent numerical
comparison. Two wrong-window programs compiled/executed but were rejected by the
comparison. These are manual golden results, not live LLM generation, Poseidon
GPU, general CNN support or completion of all goal requirements.

The supported input-graph operator set is now 13 kinds: add, multiply, subtract,
negate, square, power, linear, flatten, rotate, conv1d, conv2d, avg_pool1d,
avg_pool2d. Powers 2 and 4 are one kind; multiple sizes/weights are not extra
kinds. This exceeds twice the frozen six-kind input-graph/FX baseline. It does
NOT mean 13 new low-level Hecate primitives or doubled model-family coverage.

## Model semantics

- Unbatched, channel-first tensors: [C,L] for 1D; [C,H,W] for 2D.
- Convolution is cross-correlation in Torch's convention: do not reverse the
  kernel. Weight shapes are [Cout,Cin/G,K] and [Cout,Cin/G,Kh,Kw], where G is
  groups (default 1).
- Stride and zero padding are explicit lists. Optional groups and per-axis
  dilation now have the bounded semantics linked above; omitting them retains
  the original groups=1/dilation=1 behavior. Unsupported attributes are rejected.
- Average pooling uses floor output size, explicit kernel/stride/padding and
  explicit boolean count_include_pad. With false, padded positions do not count
  in the divisor; with true they do. MaxPool is not implemented or substituted.
- Four logical input elements remain the bound. Existing shapes [4], [2,2],
  [1,4] are joined by verified [1,2,2], [2,1,2], [1,1,4]. This expands logical
  tensor shape, not physical slot capacity or packing period.
- Spatial outputs have at most four scalar-neuron ciphertexts in channel-major,
  then row-major spatial order. Flatten only changes the logical view/order;
  it does not decrypt/repack those values into one ciphertext.
- Conv results can undergo square and average pooling before flatten. The
  existing Linear can consume flattened scalar-neuron results. No implicit
  conversion between packed-input and scalar-neuron layouts is added.

Example descriptor fields for a convolution node:

```json
{"id":"features", "op":"conv1d", "inputs":["x"],
 "weight":"w", "bias":"b", "stride":[1], "padding":[0]}
```

Pooling node:

```json
{"id":"pooled", "op":"avg_pool1d", "inputs":["features"],
 "kernel":[2], "stride":[1], "padding":[0], "count_include_pad":false}
```

Public constants may now have up to four bounded dimensions to hold kernels;
the 128-element and finite-value bounds remain. The Hecate constant registry
still consists only of scalar or period-four vectors after lowering.

## Implementation and independent checks

`spatial_ops.py` owns bounded geometry and compiler coefficient-matrix assembly.
Each matrix row describes one output window. Packed-input rows lower to a
plaintext weight multiplication and real rotate-and-add reduction; later
scalar-neuron rows lower to weighted ciphertext sums.

The reference separately walks input windows with scalar indexing and math.fsum;
it never evaluates the compiler matrix or DSL. Trusted Torch F.conv1d/conv2d and
F.avg_pool1d/avg_pool2d provide a third check. Hand calculations cover asymmetric
kernels, channel order, padding divisors and the Conv-square-pool composition.
Manual DSL sources are in `scripts/baseline/golden_cases/spatial`. They use a
different reduction order from the deterministic emitter and are checked against
the independent reference, not against decrypted intermediate values.

This does not validate upstream poly.HE_Conv/HE_Pool helper closures. Those
functions remain outside the Agent AST allowlist; the new local lowering uses
the existing verified Hecate arithmetic/rotation primitives.

## Actual failure and repair

The first deterministic nine-case run is preserved at
`/home/lhy/poseidon-work/results/fx-batch-gxw0b2jb`:
eight cases passed; Conv-square-pool failed in seal_runtime with
`result ciphertext is transparent`.

Cause: matrix expansion included zero coefficients outside a pooling window.
The scalar-neuron Linear emitted ciphertext * plaintext-zero; SEAL correctly
rejects the transparent result. The repair omits public zero terms exactly.
No SEAL check, modulus, scale, security parameter or numerical threshold changed.
Retest with the same model/weights passed at
`/home/lhy/poseidon-work/results/fx-batch-yo5fyunk`, max absolute error
1.055753189e-8. The original failure remains in its report denominator.

All-zero output rows remain explicitly rejected during FX translation: their
encrypted-zero output policy has not been verified. This is a real supported-
range restriction, not a claim that all possible user weights already work.

## Isolated manual-candidate evidence

Full batch: `/home/lhy/poseidon-work/results/spatial-golden-batch-z8dy657u`.
Its report points to each immutable candidate run and numerical result.

| Case | Max absolute error | Result |
|---|---:|---|
| conv1d-valid | 6.217162563e-9 | pass |
| conv1d-stride-pad | 6.539858086e-9 | pass |
| conv2d-valid | 6.240981620e-9 | pass |
| conv2d-channels | 1.178369094e-8 | pass |
| avg1d-pad-count | 7.194496998e-9 | pass |
| avg1d-pad-exclude | 2.845699509e-9 | pass |
| avg2d-valid | 3.185337383e-9 | pass |
| avg2d-wide | 6.357770784e-9 | pass |
| conv-square-pool | 1.744601055e-8 | pass |
| wrong conv output-window binding | 1.500000007 | expected rejection |
| wrong pool output-window binding | 0.3333333317 | expected rejection |

Nine correct cases x four input sets = 36 input runs and 68 scalar comparisons.
Negative cases add eight runs and 20 comparisons, not positive successes.
The suite includes zero, signed asymmetric, fixed-seed random and [-1,1] boundary
inputs. N=32768, SEAL tc128, fourteen 60-bit key moduli are unchanged; no bootstrap.
The threshold is still abs(actual-reference) <= 1e-5 + 1e-4*abs(reference).

Square is an explicit polynomial in the requested model, not a hidden ReLU
approximation. When approximations are later introduced, model-approximation
error and CKKS execution error must be reported separately.

## Reproduction and remaining gates

From `/mnt/d/Code Space/Poseidon`, using existing dependencies and no API calls:

```bash
python3 scripts/baseline/run_spatial_goldens.py
python3 scripts/baseline/run_model_batch.py --case scripts/baseline/cases/conv-square-pool.json
python3 scripts/baseline/run_model_batch.py --unit-tests
```

`run_model_batch.py --case-files file1.json file2.json ...` now runs an explicitly
selected set with one shared key set. All requested descriptors retain separate
results/failures. No installation/download occurs. Do not share whole result
directories because they contain FHE private keys.

Full regression with historical and new evidence: 250 discovered, 229 passed,
21 skipped under ordinary Python. Pinned Torch separately checks model semantics.
The semantic inventory records Conv/AvgPool subset status and the all-zero-output
gap; no source-name inventory is treated as proof of all parameter combinations.

Live Agent generation for the new graph families, doubled model-family coverage,
larger packing/broadcast, general widths, a verified encrypted-zero policy,
approximation error decomposition and Poseidon GPU remain open. No API calls,
installs, system changes, commits, pushes or PRs were made for this increment.
