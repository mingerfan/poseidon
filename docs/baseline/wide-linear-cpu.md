# User-defined Linear/MLP hidden widths 5..8

Date: 2026-09-07. Real CPU encrypted evidence; not new online Agent results.

## What changed

The data-only schema-2/3 graph validator and deterministic FX lowering now allow
Linear hidden widths 1..8 instead of 1..4. Users supply the actual matrices,
biases and connections; changing the model id does not select an implementation.
The shared limit is `model_graph.LINEAR_WIDTH_LIMIT`.

Input ABI remains four logical elements per ciphertext. The first Linear layer
computes genuine four-slot weighted reductions. Its hidden neurons are separate
ciphertexts containing broadcast scalars, so eight neurons do not require
changing the input slot period. Later Linear layers combine these ciphertexts
with public scalar weights. Scalar/length-one and exact-width vector broadcast
remain the only permitted public broadcast rules.

Final output is still a vector of 1..4 elements. This limit is now explicit in
graph validation rather than relying on the old per-layer width limit.
Width9, input[8], incompatible cipher layouts, wrong biases and bad broadcasts
are rejected. No security, runtime, key, compiler profile or generated-code
sandbox limits were relaxed. The existing finite graph/resource limits remain.
The cap is a supported interface bound, not a guarantee every depth/combination
below that cap will fit the compiler's parameter or resource budget.

## Models and independent oracle

Four cases describe `Linear(4,H) -> square -> add(public offset) -> Linear(H,2)`
for H=5,6,7,8. H=5/6 use scalar offset and H=7/8 use exact-width vectors.
Matrices contain signed, nonzero fixed values. Test inputs retain zero,
asymmetric fixed signed, fixed-seed random and declared range boundaries.

The independent plaintext reference uses explicit row dot products and
`math.fsum`, checked against CPU float64 PyTorch. The handwritten width8 DSL
uses rotate2 then rotate1 reductions, separately from the rule emitter's order.
Its constant-to-neuron mapping is explicitly checked. A wrong candidate uses
neuron6 in place of neuron7 in the first output's last weighted term; asymmetric
weights/offsets make this error visible even when shapes and compilation pass.

## Actual results

Rule evidence: `/home/lhy/poseidon-work/results/fx-batch-97kbilk1`.

| Hidden width | Result | Max absolute error |
|---|---|---:|
| 5 | Passed | 3.430952877e-8 |
| 6 | Passed | 6.481164785e-9 |
| 7 | Passed | 1.746338052e-8 |
| 8 | Passed | 1.684313944e-8 |

Four models x four input groups = 16 actual encrypted evaluations and 32
compared output values. Threshold stays `abs(error) <= 1e-5 + 1e-4*abs(reference)`.

Manual candidate evidence:
`/home/lhy/poseidon-work/results/wide-linear-goldens-02h0bwbh`.

| Candidate | Result | Max absolute error | Candidate evidence |
|---|---|---:|---|
| Correct manual width8 | Passed | 7.260639645e-9 | candidate-replay-ixz1my3w |
| Wrong neuron index | Rejected at numerical comparison | 0.527954094790 | candidate-replay-2ov2tnj4 |

Both manual candidates compiled and performed real encrypted execution before
comparison. Four input groups / eight output values each; the negative case is
not counted as successful model execution. Input/reference/artifact hashes were
verified. No Agent API calls or GPU execution occurred.

SEAL tc128, N=32768, 16,384 slots, the existing 14x60-bit key-context modulus
profile and scale 2^40 remain unchanged. No bootstrap was executed.

## Commands

Run from `/mnt/d/Code Space/Poseidon` in WSL:

```bash
# Local deterministic execution with explicit user graph files.
python3 scripts/baseline/run_model_batch.py --case-files \
  scripts/baseline/cases/custom-wide-mlp-5.json \
  scripts/baseline/cases/custom-wide-mlp-6.json \
  scripts/baseline/cases/custom-wide-mlp-7.json \
  scripts/baseline/cases/custom-wide-mlp-8.json

# Positive/negative manual candidates through the isolated pipeline.
python3 scripts/baseline/run_wide_linear_goldens.py

# Existing evidence audit, no encrypted rerun or API calls.
POSEIDON_WIDE_RULE_RESULTS=/home/lhy/poseidon-work/results/fx-batch-97kbilk1 \
POSEIDON_WIDE_GOLDEN_RESULTS=/home/lhy/poseidon-work/results/wide-linear-goldens-02h0bwbh \
python3 -m unittest discover -s scripts/baseline -p test_wide_linear.py -v
```

The usual `run_candidate.py --case <file> --prepare` can prepare a request for
one of these graphs without calling a model. Online generation still uses the
shared .env and requires the agreed live-call authority; no online width8
success is claimed here. Existing 48/96 benchmark definitions and historic
results are unchanged; these four width configurations are not counted as four
new model families.

## Remaining work

General input packing, larger final output, systematic deeper wide MLPs,
all-zero output rows and Poseidon GPU remain separate validation requirements.
The CPU artifact still includes ModswitchC; increasing model width does not
remove the known Poseidon adapter/GPU blocker.
