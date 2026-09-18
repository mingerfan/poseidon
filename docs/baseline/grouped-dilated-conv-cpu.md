# Grouped and dilated Conv: bounded user graphs, real CPU evidence

## Outcome and limits

Six new user-defined Conv graphs passed both deterministic lowering and the
manual-candidate Hecate path, including actual Dacapo compilation, HEVM/CST gates,
CKKS encryption/evaluation/decryption and independent numerical comparison.
Two intentionally incorrect manual programs compiled and executed but failed
numerical comparison. All runs use existing upstream SEAL HEVM CPU, not GPU.
No API calls, installs, parameter changes or bootstrap occurred.

This extends parameters of the existing conv1d/conv2d operator kinds; it does not
increase the declared 13 operator kinds or the versioned 16-family/96-case suite.
The old suite and reports are unchanged. New live LLM generation is unverified.

## Input semantics

Conv nodes may now contain optional `groups` and `dilation`:

```json
{"id":"features", "op":"conv1d", "inputs":["x"],
 "weight":"w", "bias":"b", "stride":[1], "padding":[0],
 "dilation":[2], "groups":1}
```

- Inputs remain unbatched, channel-first and bounded to four logical elements.
- `groups` is an integer 1..4 and must divide both input and output channels.
  Boolean, float and null values are rejected. Available shapes/output limits
  still constrain which combinations are meaningful.
- `dilation` is a per-axis integer list, 1..4 per entry. Its length must equal
  spatial rank. JSON scalar dilation, null, booleans and extra axes are rejected.
- Omission means groups=1 and all dilation entries=1, preserving old descriptors.
- Weight shape is `[Cout,Cin/groups,K]` or `[Cout,Cin/groups,Kh,Kw]`.
- Stride 1..4 and zero padding 0..2 remain explicit per-axis lists.
- Every spatial output has 1..4 scalar-neuron ciphertexts. No general batch,
  reflect padding, transposed convolution or repacking is added.
- Pooling does not gain dilation/groups attributes. All-zero coefficient rows now use
  the separately verified [explicit encrypted-zero ABI](encrypted-zero-cpu.md);
  transparent-ciphertext checks remain enabled. This later extension does not
  retroactively change the grouped/dilated evidence recorded below.

For one spatial axis, a kernel tap t in output location o samples:

`input_position = o * stride - padding + t * dilation`.

The effective kernel span is `(K-1)*dilation+1`; floor output length is
`(L+2*padding-effective_span)//stride+1`. For each output channel, only input
channels in its contiguous group contribute. Kernel weights are not reversed:
this is PyTorch-style cross-correlation. With groups=Cin and Cout a multiple
of Cin, the same rule covers depthwise convolution and channel multipliers.

## Implementation and correctness checks

- `model_graph.py` validates node parameters, constructs trusted Torch calls,
  and passes parameters to the independent reference. It does not select by ID.
- `spatial_ops.py::lowering` builds the public coefficient rows for the emitter.
- `spatial_ops.py::reference` independently walks channel groups and sampled
  windows with scalar arithmetic; it never consumes the compiler matrix.
- `fx_to_hecate.py` extracts explicit FX arguments, then emits the existing
  weight multiplication and rotate/sum primitives. There is no new Hecate call.
- `test_grouped_spatial.py` records hand-computed outputs and exact expected
  coefficient masks, compares Torch/reference/rule/manual results, checks that
  arbitrary user IDs and changed weights retain their intended meaning, and
  confirms the provider request carries the graph but no reference answer.

Manual programs use rotate(2) then rotate(1), whereas the deterministic emitter
uses the opposite reduction order. They share the checked public constant ABI;
therefore explicit expected-mask assertions are important to avoid a common
incorrect coefficient expansion being mistaken for independent proof.

Static negative tests cover invalid group divisibility, weight channels,
dilation type/range/rank, effective kernels with no output, and unknown fields.
The encrypted negatives omit a dilated reduction contribution or use another
input-channel group. Both pass the AST/compile gates and are rejected only by
the numerical oracle, as intended.

## Actual experiments

Manual evidence: `/home/lhy/poseidon-work/results/grouped-spatial-goldens-ffbvf58_`.
Deterministic evidence: `/home/lhy/poseidon-work/results/fx-batch-3lsvho1l`.

| Case | Feature | Manual max absolute error | Rule max absolute error |
|---|---|---:|---:|
| conv1d-dilation2 | dilation 2 | 2.8296562e-9 | 9.1446479e-9 |
| conv1d-dilation3 | dilation 3 | 3.1749782e-9 | 2.6761999e-9 |
| conv2d-dilation2 | horizontal dilation | 5.0327997e-9 | 6.5179598e-9 |
| conv1d-groups2 | disjoint channel groups | 1.0753821e-8 | 4.9866473e-9 |
| conv2d-depthwise-multiplier | 2 groups, 4 output channels | 6.3735757e-9 | 2.8731330e-9 |
| conv2d-grouped-dilated-pad | groups + dilation + stride + padding | 1.9825156e-9 | 1.4574062e-9 |

Each program uses zero, signed fixed, seed-42 random and range-boundary inputs
in [-1,1]. The six positives have 24 input evaluations and 60 compared output
values per generator. The two negatives add 8 input evaluations; their maximum
errors are approximately 0.25000000063 and 0.12500180044. They are expected
rejections, not successful model outputs or provider generation failures.

SEAL tc128, N32768, key-context 14x60-bit moduli, input scale 2^40 and the
per-element threshold `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` are
unchanged. Evidence tests check parameters, sandbox probes, artifact hashes,
frozen inputs/reference/weights, expected failure layer and no GPU/API claims.
These fixtures are finite evidence, not a proof for every allowed parameter
combination. In particular the encrypted cases exercise groups 1/2 and dilation
1/2/3, not an exhaustive sweep of every group/dilation value accepted by geometry.

The original nine Conv/Pool/composition cases were freshly recompiled and
executed after this change: **9/9 passed**, recorded separately at
`/home/lhy/poseidon-work/results/fx-batch-csu_nzos`.
The pinned Torch frontend test run passed 61 tests and skipped 8 saved-evidence
checks (69 total at that point). The subsequent ordinary-WSL full run included
both new evidence paths: 284 tests, 259 passed and 25 skipped. The two new Torch
checks are among those host skips but passed in the pinned run. The final added
rule-evidence test passed in the full run. No skipped test is counted as passed.

## Reproduce

Run from `/mnt/d/Code Space/Poseidon`, reusing the existing isolated environment:

```bash
python3 scripts/baseline/run_model_batch.py --unit-tests
python3 scripts/baseline/run_grouped_spatial_goldens.py
python3 scripts/baseline/run_model_batch.py --case-files \
  scripts/baseline/cases/conv1d-dilation2.json \
  scripts/baseline/cases/conv1d-dilation3.json \
  scripts/baseline/cases/conv2d-dilation2.json \
  scripts/baseline/cases/conv1d-groups2.json \
  scripts/baseline/cases/conv2d-depthwise-multiplier.json \
  scripts/baseline/cases/conv2d-grouped-dilated-pad.json
```

Saved-evidence tests do not rerun encryption or invoke the model service:

```bash
POSEIDON_GROUPED_SPATIAL_RESULTS=/home/lhy/poseidon-work/results/grouped-spatial-goldens-ffbvf58_ \
POSEIDON_GROUPED_SPATIAL_RULE_RESULTS=/home/lhy/poseidon-work/results/fx-batch-3lsvho1l \
  python3 -m unittest discover -s scripts/baseline -p test_grouped_spatial.py -v
```

The existing Agent CLI accepts these data-only graphs, but no new live Agent
success rate follows from the rule/manual runs. Poseidon GPU opcode/profile
alignment and actual execution remain open, independently of this extension.
