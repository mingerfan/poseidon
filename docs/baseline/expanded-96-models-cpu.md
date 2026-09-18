# 16 model families / 96 cases: deterministic CPU milestone

Date: 2026-09-07. Evidence: `/home/lhy/poseidon-work/results/fx-batch-0z3yioki`.

## What is verified

96/96 descriptors passed PyTorch/reference validation, deterministic FX lowering,
Hecate tracing, Dacapo compilation, artifact checks, actual SEAL HEVM CKKS
execution, decryption, and elementwise comparison. There were **zero Agent/API
calls** and **no Poseidon GPU execution** in this batch.

Four input batches per case give 384 encrypted evaluations and 1,088 compared
output values. The maximum absolute error across the cohort is
`3.1183864379613624e-8`. The frozen threshold remains
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
Per-case reports retain MAE, relative error, cosine similarity and individual
values. Security was not reduced: SEAL tc128, N=32768, 16,384 slots,
14 x 60-bit key-context moduli / 13 initial data-context moduli, input scale 2^40.
No bootstrap or decrypt-and-reencrypt replacement was executed.

## Taxonomy and exact model identity

`scripts/baseline/expanded_model_suite.py` defines
`restricted-model-suite-v1`. Its ordered descriptor-set SHA256 is
`0856364bfe771bb226d90963847f9f6987dda9ad9a1245f228d4aee713f65714`.

The original 8 x 6 schema-1 cases are unchanged. The added 8 x 6 cases are full
schema-2/3 user graphs with explicit public weights, not new factory selectors.
Model-family labels live outside those descriptors and are not passed to the
translator to select an implementation. Six configurations are six cases, not
six model families; the taxonomy does not claim eight novel neural architectures.

| Family | Passed | Maximum absolute error |
|---|---:|---:|
| affine | 6/6 | 1.017893514e-8 |
| polynomial | 6/6 | 1.428114893e-8 |
| linear | 6/6 | 1.171818820e-8 |
| mlp2 | 6/6 | 2.267264332e-8 |
| mlp3 | 6/6 | 2.540127858e-8 |
| fanout | 6/6 | 3.118386438e-8 |
| residual | 6/6 | 2.179746794e-8 |
| flatten_linear | 6/6 | 1.273917559e-8 |
| conv1d | 6/6 | 1.440979081e-8 |
| conv2d | 6/6 | 9.669966078e-9 |
| avg_pool1d | 6/6 | 1.014928139e-8 |
| avg_pool2d | 6/6 | 1.115679693e-8 |
| conv_poly_pool | 6/6 | 1.251791959e-8 |
| dual_affine | 6/6 | 1.200354405e-8 |
| dual_bilinear | 6/6 | 1.058549566e-8 |
| dual_linear | 6/6 | 2.825263223e-8 |

The model graph operator set is now 13 kinds (baseline 6): add, multiply,
subtract, negate, square, power, linear, flatten, rotate, conv1d, conv2d,
avg_pool1d, avg_pool2d. Aliases, overloads and shapes are not counted as operators.
Native subtraction/negation and all six signed rotations have separate manual
golden evidence; not every distinct DSL syntax is exercised by every rule batch.

## Commands

Run from `/mnt/d/Code Space/Poseidon` in WSL.

```bash
# Existing evidence audit; no model API calls and no encrypted rerun.
PYTHONDONTWRITEBYTECODE=1 \
POSEIDON_EXPANDED_RULE_RESULTS=/home/lhy/poseidon-work/results/fx-batch-0z3yioki \
python3 -m unittest discover -s scripts/baseline -p test_expanded_suite.py -v

# Inspect the exact Agent cohort without reading .env, entering Nix or using API.
python3 scripts/baseline/run_agent_batch.py --plan --extended

# Fresh deterministic encrypted run (new evidence directory, zero model calls).
python3 scripts/baseline/run_model_batch.py --extended

# Fresh paid online Agent experiment, not a way to inspect the above results.
python3 scripts/baseline/run_agent_batch.py --live --extended --provider deepseek \
  --model deepseek-v4-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 900 --jobs 2
```

Agent batch reporting accepts independent family metadata, preserves the full
96-case denominator, and supports hash-linked failure-subset reruns with
`--extended --failed-from <report.json>`. Its audit checks cohort hashes and
family labels against the selected deterministic baseline, along with existing
weights/input/reference, compiler/runtime and tolerance checks. The default
Agent batch remains the original 48 cases. Credentials remain only in the shared
project-root `.env` (or explicit per-variable process overrides).

## Remaining limits

- This document records deterministic results, not online Agent success on 96 cases.
- Four logical elements per encrypted input, up to four separately encrypted
  inputs, widths/output counts <=4: shape diversity is not general packing.
- Conv/AvgPool are the bounded local lowerings documented in
  [spatial-operators-cpu.md](spatial-operators-cpu.md), not blanket permission to
  call upstream high-level poly helpers.
- All-zero Linear rows are rejected pending verified encrypted-zero semantics.
- Explicit square is the model, not an approximation to an original ReLU model.
  Separate approximation-error validation remains required before such rewriting.
- Poseidon GPU opcode, level/scale and execution compatibility remain a separate
  gate. This CPU result does not close that requirement.

The stored benchmark manifest is a definition, not a mutable success certificate;
its initial validation booleans remain false. Use the completed execution report
and the evidence test above to determine this batch's outcome. Subsequent local
reporting/Agent-entry changes do not rewrite the original experiment or its
source hashes.

## Actual Poseidon adapter gate on this cohort

The existing native CPU-only `poseidon_mgpu_dacapo_hevm_dump` was run against
all 96 frozen HEVM/CST pairs, with one device configured and no false claims
that GPU keys or communication were available. Evidence is
`/home/lhy/poseidon-work/results/poseidon-cohort-preflight-4n1228ta`.
Every tool invocation returned code 1 and a durable `not_ready` JSON report:
**96/96 contain unsupported opcode 4 (`ModswitchC`); 0/96 built a schedule.**
This is an observed adapter blocker before GPU execution, not a CUDA installation
error, and not a claim that ModswitchC is the only remaining GPU incompatibility.

```bash
python3 scripts/baseline/run_poseidon_cohort_preflight.py \
  /home/lhy/poseidon-work/results/fx-batch-0z3yioki
POSEIDON_COHORT_PREFLIGHT_RESULTS=/home/lhy/poseidon-work/results/poseidon-cohort-preflight-4n1228ta \
python3 -m unittest discover -s scripts/baseline -p test_poseidon_cohort_preflight.py -v
```

The planned 96-case online Agent command has **not run**: execution approval
rejected the larger paid workload pending explicit confirmation of this cohort
and spending scope. No new online success rate exists. Do not combine the old
48-case Agent result with these 96 deterministic successes.
