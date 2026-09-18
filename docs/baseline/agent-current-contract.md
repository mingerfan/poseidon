# Current Agent contract and remaining work

Snapshot: 2026-09-18. Source evidence: `model_catalog.py`, `model_graph.py`,
`packed_model.py`, `candidate_contract.py`, `native_function_rules.py`,
`run_candidate.py`, `candidate_sandbox.py`, `deepseek_provider.py` and
`dsl_semantic_inventory.py`. Implementation acceptance, manual execution evidence
and live Agent-generation evidence are deliberately different claims.

## Inputs

The public runner accepts a **data-only JSON model description**, not an arbitrary
uploaded Python program, ONNX model, natural-language request or pickle checkpoint.
The original project direction is restricted PyTorch; the implemented safe entry
constructs a trusted CPU-float64 PyTorch module from the validated description.
`fx_to_hecate.translate` also accepts trusted in-process modules, but that is the
deterministic reference translator, not a general untrusted model-file importer.

| JSON schema | Meaning | Bounds |
|---|---|---|
| 1 | Historical fixed catalog family/configuration | 8 families x 6 presets |
| 2 | User-defined static graph, one encrypted input | Four logical input elements; legacy output/layout limits |
| 3 | User-defined graph with independently encrypted inputs | 2..4 inputs, four logical elements each |
| 4 | One logical tensor partitioned across ciphertexts | Rank 1..4, 5..16 elements, 2..4 period-4 ciphertexts; output 1..4 elements |
| 5 | One variable-period packed encrypted tensor | Rank 1..4, total 1..256 elements, period 4/8/16/32/64/128/256 |

Schema5 supports arithmetic, explicit square/power2/4, last-axis Linear,
reshape/flatten, permute/transpose, fixed-statistics BatchNorm, concat, bounded
Conv1D/2D and AvgPool1D/2D. Conv supports checked groups/dilation/stride/padding;
not arbitrary spatial modes. Elementwise packed outputs may contain up to 256
logical elements. Scalar-neuron layouts (Linear/Conv/concat etc.) have at most
16 output ciphertexts. Rank, broadcast and layouts must match across merges.
Graphs have at most 64 nodes and 32 public constant entries; additional constant,
AST and operation budgets apply. Acceptance does not guarantee enough CKKS depth.

Example input (its supplied constants are public, x is encrypted):

```json
{
  "schema": 5,
  "id": "matrix-affine-example",
  "input_shape": [2, 3],
  "constants": {"gain": [0.5], "bias": [0.375]},
  "nodes": [
    {"id": "scaled", "op": "multiply", "inputs": ["x", "gain"]},
    {"id": "result", "op": "add", "inputs": ["scaled", "bias"]}
  ],
  "output": "result"
}
```

This is a small affine model, not Linear: Linear must perform cross-element
weighted reduction. Tests use fixed zeros, signed values, seeded random values,
and boundary values. Weights and input arrays are persisted without pickle.

## Processing and boundaries

```text
JSON model / fixed public weights
  -> schema, operators, type/shape/layout checks
  -> trusted PyTorch reference + Torch FX + deterministic rule baseline
  -> hash-bound public request: graph, constants, layout, DSL rules
  -> model provider -> candidate JSON containing Hecate source
  -> JSON/AST/type/construction checks
  -> bubblewrap worker -> real Hecate tracing -> Earth IR
  -> Dacapo lowering/optimization -> CKKS IR -> HEVM/CST
  -> artifact/security/key checks -> upstream SEAL CPU encrypted execution
  -> decrypt/decode -> independent reference comparison
  -> success, or restricted diagnostics -> regenerate (at most 3 repairs)
```

This is a constrained generator/verifier/repair system, not an LLM with arbitrary
shell/file access and not a multi-agent orchestration service. The only supported
provider is DeepSeek. Its key is loaded from one local backend `.env`.
Provider/network retries
and semantic repair rounds are different counters.

The LLM sees public model semantics, supplied constant registry, versioned rules,
fixed layout, response schema and sanitized feedback. It does not receive the
deterministic DSL answer, test input arrays, reference outputs or secret keys.
Numerical feedback contains summary errors, not a table of answers to memorize.
Generated source is interpreted through an AST allowlist into the trusted real
frontend; it is not executed with unrestricted Python eval/exec.

The deterministic translator is currently also used to construct constant/layout
metadata. Therefore the live Agent operates inside a compiler-supported model
subset. It cannot independently enable an operator rejected by that preparation
stage; successful synthesis is not evidence that an LLM surpasses that translator.

## Outputs

The model response has exactly `schema`, `request_id`, `hecate_source`.
It does not get to replace the fixed layout or model weights. An illustrative
source fragment is `@hc.func("c") def golden(x): ...`; supplied constants are
referenced by their assigned registry names, rather than invented at runtime.

The complete runner additionally saves `model.json`, `request.json`, fixed
`weights.npz` / `arrays.npz`, `rule-answer.py`, per-attempt response/source,
tracing evidence, Earth/CKKS IR, HEVM/CST, execution metadata, decrypted NumPy
arrays, comparison and failure-layer records, usage counts and integrity hashes.
Schema5 can also save decoded logical tensor outputs separately from physical
ciphertext/slot outputs. Temporary experiment keys are removed after terminal
runs; comparison evidence remains. Return layouts distinguish ciphertext lists
from the slots packed inside one ciphertext.

The frozen numerical rule is `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
Reports include MAE, maximum absolute error, relative error on nonzero reference
entries, cosine similarity when defined, and per-element results. This is finite
differential evidence, not equivalence for all possible inputs.

## DSL alignment: do not conflate three layers

1. **Mathematical operators.** Cipher/plain and cipher/cipher add, subtract,
   multiply, negate, rotations, explicit polynomials, reduction-based Linear and
   bounded tensor/spatial layouts have compiler/real SEAL CPU evidence, including
   separate live Agent cohorts. Compiler-owned scale/level/rescale/modswitch and
   relinearization are not free-form LLM statements.
2. **Native Hecate construction.** Opt-in contracts support typed decorated
   helper functions, forward/nested/repeated calls, multiple and Plain returns,
   positional unpacking, bounded object arrays, views/copies, shape transforms,
   public range loops and scalar/array augmented assignment. These contracts are
   versioned, not one unrestricted union of all Python constructs.
3. **Coverage evidence.** Eleven native-function call exercises have live Agent
   plus real encrypted evidence (10 first-pass, 11 after repair). Packed/native
   free synthesis has six successful programs but actually observed helpers in
   only two; it did not by itself cover arrays/loops/starred/augmented constructs.
   Manual goldens for those constructs must not be relabeled as paid Agent proof.
   The six targeted packed/native exercise definitions/checkers exist, but their
   full batch/paid coverage is not yet completed in this snapshot.

The separate public-construction contracts cover bounded construction-time
helpers, sequences/mappings, public loops/conditions, public numeric/string
operations and polynomial coefficient construction. Public condition evaluation
does not imply encrypted data-dependent branching. A feature accepted by one
contract must not be assumed legal under another; the request rules are authoritative.

## Not yet included or established

- Arbitrary PyTorch/ONNX/model.py loading, dynamic shapes and unbounded dimensions.
- One universal grammar combining every historical construction contract.
- Arbitrary Python imports, filesystem/network/process access, recursion,
  unrestricted NumPy, ciphertext-dependent if/while or mutable slot indexing.
- Unrestricted broadcasting/packing, batch sizes, depth, or arbitrary multi-input
  large-tensor combinations. A supported operator is still bounded by its ABI.
- Verified execution of all upstream `HE_Conv`, `HE_Linear`, `HE_Concat`,
  `HE_ReLU`, `HE_SiLU` and other poly wrappers. Our lowered Conv/Linear support is
  not proof of those helpers' original layout/closure semantics.
- Automatic ReLU/SiLU/MaxPool replacement. Explicit approximation fixtures are
  distinct from preserving the original model exactly.
- True bootstrap in this Agent execution chain; unsupported Upscale/backend
  operations remain explicit failures. No decrypt/re-encrypt substitute.
- End-to-end generated-program execution on Poseidon GPU. Existing GPU primitive
  or schedule tests do not establish the HEVM/DSL GPU chain.
- Transparent legacy evidence archive relocation.

## Recommended next engineering milestones

1. Complete the six targeted packed/native real Agent exercises, checking actual
   construction influence and real trace records, not merely source spellings.
2. Publish a single generated input/operator/contract capability matrix and freeze
   a testable external interface; do not silently merge incompatible versions.
3. On a newly provisioned Ubuntu x86_64 checkout, repeat
   doctor -> offline Linear -> small live Agent run. Provisioning is separate.
4. Extend larger/multiple tensor inputs and unseen graph compositions with manual
   goldens, rule baselines, negative cases and per-construct live coverage.
5. Treat high-level upstream poly helpers, bootstrap and Poseidon GPU integration
   as separately scoped work. Optimize only after correctness.

Portability, supported-set correctness and repeatable repair are engineering
progress. Research claims need controlled comparison against the deterministic
translator and evidence of capability beyond already supported one-to-one mappings.
