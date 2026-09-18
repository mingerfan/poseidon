# Schema-3: user multi-input graphs through the isolated candidate pipeline

**Historical baseline below.** New multi-input requests now use the clarified
v4 generation protocol with the same v3 AST semantics; historical v3 requests
remain valid. See [live failure diagnosis and protocol repair](deepseek-37-results-and-multi-input-v4.md)
for the completed DeepSeek batch and subsequent failure-only experiment.

## Outcome and limits

The public model loader, trusted Torch/FX translator, Agent request contract,
candidate AST validator, sandboxed Hecate tracing, HEVM artifact gate and real
SEAL CPU runtime now preserve two to four separate encrypted inputs. Four manual
candidates pass the complete isolated pipeline; a reversed-input subtraction
counterexample fails only after real decryption and numerical comparison.

No model API was called for this increment. The provider-neutral request format
is accepted by the existing DeepSeek adapter in offline tests, but live LLM
generation/repair for v3 remains unverified. This is not Poseidon GPU execution.
Existing schema-1 catalog and schema-2 single-input graphs remain supported.

## User input format

Start with `scripts/baseline/cases/custom-dual-linear.json`. Unlike a catalog
selector, it contains the user's graph, ordered inputs and actual public weights:

```json
{
  "schema": 3,
  "id": "my-two-input-model",
  "inputs": [
    {"name": "features", "shape": [2, 2]},
    {"name": "residual", "shape": [4]}
  ],
  "constants": {
    "weights": [[0.3, -0.7, 0.125, 0.5], [-0.4, 0.2, 0.75, -0.25]],
    "bias": [0.1, -0.2]
  },
  "nodes": [
    {"id": "flat", "op": "flatten", "inputs": ["features"]},
    {"id": "difference", "op": "subtract", "inputs": ["flat", "residual"]},
    {"id": "linear", "op": "linear", "inputs": ["difference"], "weight": "weights", "bias": "bias"},
    {"id": "activated", "op": "square", "inputs": ["linear"]}
  ],
  "output": "activated"
}
```

`id` is not a model lookup key. Tests change identifiers, input names, weights and
graph connections and check the actual resulting semantics. Inputs are an
ordered list, not a dictionary whose order could be changed by serialization.
The first input maps to DSL x, second to y, third to z, fourth to t. The request
records each user name, DSL name and logical shape explicitly; modifying this
mapping invalidates the request hash.

Every input still has four logical elements, with shape [4], [2,2] or [1,4].
Each is packed into its own ciphertext with period four. Different input shapes
can meet only after explicit compatible flattening; flatten is C-row-major and
does not perform a ciphertext reshape. This does NOT support arbitrary layout,
batch size, convolution or larger feature widths.

The existing nine graph operators and constant/node bounds remain in force.
All weights are public, fixed and provided as finite JSON numbers. No user
Python, pickle or object-deserializing model checkpoint is loaded. The test
input sets remain zero, signed asymmetric, fixed-seed random and [-1,1] boundary;
arbitrary user-supplied test datasets are a separate interface extension.

## Semantic path

1. `model_graph.py` validates schema-3 and builds one of four trusted static
   Torch signatures. User names bind to those arguments in declared order.
2. `run_model_batch.prepare_case` checks an independent scalar-arithmetic
   reference against Torch float64 and freezes the independent result. Runtime
   arrays have shape [4 test cases, input count, 4 packed values]; logical shapes
   are separately recorded in the immutable model and request.
3. `fx_to_hecate.py` emits an explicit multi-argument function and v3 layout.
   As before, the rule translator fixes constant names/layout; its complete
   answer is NOT included in the Agent request.
4. `candidate_contract.py` supplies the v3 grammar/rules and checks signature
   and ordered input bindings. The public request includes model semantics and
   public weights, not local reference outputs or secret keys.
5. `candidate_trace.py` constructs Hecate expressions from a validated AST in
   bubblewrap. It does not execute candidate Python. The tracing process has
   neither test/reference files nor FHE keys; runtime receives input arrays but
   not the reference.
6. The artifact gate checks expected HEVM arity. Runtime encrypts each argument
   separately, checks actual key files and ciphertext metadata, executes the
   unchanged SEAL HEVM backend, and compares decoded outputs locally.

## Real results

Batch: `/home/lhy/poseidon-work/results/schema3-golden-batch-ht5no6ir/report.json`.

| Public model | Manual candidate max absolute error | Result |
|---|---:|---|
| custom-dual-subtract | 1.165890893e-8 | pass |
| custom-dual-linear (flatten, subtract, user Linear, square) | 1.158750718e-8 | pass |
| custom-triple-merge | 1.315104492e-8 | pass |
| custom-quad-merge | 2.263376742e-8 | pass |
| custom-dual-subtract with reversed subtraction | 3.999999998 | expected numerical rejection |

The four correct cases cover 16 input tuples and 56 scalar output comparisons.
The counterexample adds four input tuples and 16 comparisons, counted separately.
All candidate runs passed the nine sandbox capability probes. Frozen model,
weights, inputs, reference, requests and compiled artifact hashes were checked.
No tolerance, compiler profile or security parameter was changed.

Separate deterministic FX evidence for custom-dual-linear:
`/home/lhy/poseidon-work/results/fx-batch-qfnvlk0m/report.json`, passed.
The manual golden uses native subtraction and a different reduction order from
the rule answer; neither computes its reference by interpreting the DSL.

Legacy isolated candidate self-test:
`/home/lhy/poseidon-work/results/candidate-replay-lx3j4xd_/report.json`.
Invalid JSON -> numerical wrong-reduction rejection -> correct program passed.
This is scripted feedback replay, not a learned LLM repair result.

Regression: 242 discovered tests, 223 passed and 19 skipped in ordinary Python
with all evidence directories supplied. Pinned Torch: 44 discovered, 42 passed,
two evidence-path tests skipped under the pure launcher; those two passed in
the ordinary-interpreter evidence run. No infrastructure or numerical failure
is hidden in these skips.

## Commands

From `/mnt/d/Code Space/Poseidon`, no API calls:

```bash
python3 scripts/baseline/run_model_batch.py --case scripts/baseline/cases/custom-dual-linear.json
python3 scripts/baseline/run_schema3_goldens.py
python3 scripts/baseline/run_candidate.py --case scripts/baseline/cases/custom-dual-linear.json --prepare
```

To request actual Agent generation, using the existing shared project-root
`.env` (this command was NOT executed for the evidence above):

```bash
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-dual-linear.json --live --provider deepseek
```

That command sends the public graph/weights/rules to the selected service and
can incur charges. A successful manual candidate does not guarantee its result.

## Remaining objective

Live v3 Agent generation and feedback repair, larger layout/broadcast, broader
Linear/MLP, Conv/Pool, approximation-versus-CKKS error separation, doubled
operator/model-family coverage and Poseidon GPU remain open. Four new descriptors
do not automatically count as four new model families. Changes are local and
uncommitted, on the existing branch; no install, sudo or remote write occurred.
