# First real DeepSeek-generated Linear: passed (2026-09-06)

## Outcome and evidence

The first returned candidate passed parsing, restricted AST/type/layout checks,
real Hecate tracing, Dacapo compilation, actual encrypted SEAL HEVM CPU execution
and independent PyTorch float64 comparison. No repair request was needed.

Evidence directory (including private FHE keys; do not share the whole directory):
`/home/lhy/poseidon-work/results/agent-deepseek-yr8gvw80`

Safe entry points for inspection:

- `report.json`: stages, provider metadata, parameters, hashes and numerical data.
- `attempt-00/candidate.py`: the actual model-generated Hecate function.
- `attempt-00/output/candidate_trace.mlir`: traced frontend IR.
- `attempt-00/output/lowered.earth.mlir` and `lowered.ckks.mlir`: compiler IR.
- `attempt-00/output/lowered._hecate_golden.hevm` and `_hecate_golden.cst`.
- `attempt-00/output/decrypted.npy`: actual decrypted outputs.

Report flags: `provider=deepseek_api`, `agent_calls=1`,
`artifact_replay_only=false`, `llm_generation_validated=true`,
`loop.repairs_used=0`, `loop.first_attempt_passed=true`.
The transport received a response with model `deepseek-v4-pro` and request ID
`6b395185-db1b-4d7c-bfa8-d0a36d357788`. No offline provider or rule-answer fallback
was used. Usage was 936 prompt tokens and 6649 completion tokens (including
6393 reasoning tokens), total 7585. Currency cost is not recorded.

## Model and generated semantics

Input descriptor: `scripts/baseline/cases/linear-example.json`, catalog case
`linear-1`: public fixed `Linear(4, 2, bias=True)`, encrypted four-element input.

Weights and biases supplied as public constants:

```
W = [[ 0.5,  -0.75, -0.125, 0.25],
     [-0.75,  0.125, -0.25,  0.25]]
b = [0.1875, 0.03125]
y = W x + b
```

The function multiplies the encrypted input by each weight row, sums the four
slot positions using left rotations by 1, 2 and a composition for 3, then adds
the corresponding bias. It returns two ciphertexts, selecting slot zero from
each. This is a true cross-slot reduction, not elementwise multiplication in
place of Linear. Input packing has period four; layout and constant registry
were supplied by the deterministic harness, not independently chosen by the LLM.
Source operator counts: 2 multiplies, 6 rotations and 8 additions.

The four test inputs were zero, fixed signed input, RNG(seed=42) uniform input,
and range-boundary input. All eight scalar outputs passed the unchanged rule:

`abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)`

| Metric | Result |
|---|---:|
| MAE | 7.605718580980247e-9 |
| Maximum absolute error | 1.2659063436393225e-8 |
| Maximum nonzero-reference relative error | 2.45423328326666e-7 |
| Cosine similarity | 0.9999999999999998 |
| Compared scalar outputs | 8/8 passed |

## Execution configuration and boundaries

- Existing upstream SEAL 4.0.0 HEVM CPU runtime, not Poseidon GPU.
- N=32768, slots=16384, SEAL tc128 parameter validation passed.
- Fourteen 60-bit key moduli, thirteen initial data moduli.
- Input log2(scale)=40; observed output log2(scale)=80 with two data moduli.
- Rotation keys for steps 1 and 2; no bootstrap executed.
- Real frontend construction uses a validated AST interpreter, not arbitrary
  execution of model-generated Python.
- Compiler used `--eva --waterline=40 --verify-each --mlir-disable-threading`.
- All nine sandbox probes passed. Hidden reference is compared locally and is
  not passed to the runtime or service as a target answer.

The report's `execution.wall_seconds=2.2098011699999915` is runtime-worker
evidence only, not API/compilation/end-to-end latency or a GPU benchmark.

This establishes one successful real generated case, not a 48-case success rate,
unseen-family generalization, a tested real LLM repair loop, arbitrary PyTorch
model import, or a formal proof for every possible input. The active backend is
explicitly `upstream_SEAL_HEVM_CPU`; `poseidon_gpu_validated=false`.

## Reproduce

Configure the ignored project-root `.env` locally, then run. Credential loading
now uses the shared file for all providers; the historical `.env.local` name
is no longer read. This configuration update does not change the recorded run:

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/linear-example.json \
  --deepseek
```

This makes a fresh paid request and writes a fresh results directory. Response
programs and CKKS noise can vary; exact previous numbers need not repeat. No key
value is included in this document, source-control changes or command line.

Next validation: explicit polynomial and square-activation MLP, followed by
fan-out/residual structure holdouts and the full 48-case matrix. Keep Agent and
deterministic-translator statistics separate; do not optimize before expanding
correctness evidence.
