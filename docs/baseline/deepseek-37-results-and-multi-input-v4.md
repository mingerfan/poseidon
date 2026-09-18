# DeepSeek 37-case results and multi-input protocol repair

## Completed experiment

Source: `/home/lhy/poseidon-work/results/agent-batch-9unkigdy/report.json`.
Status `completed_with_failures`; elapsed 5901.876 seconds (98.36 minutes).
All 37 selected cases were attempted: 19 passed, 18 failed, none pending.

| Family | Passed / attempted |
|---|---|
| Conv2D (remaining configurations) | 2 / 2 |
| AvgPool1D (remaining configurations) | 5 / 5 |
| AvgPool2D | 6 / 6 |
| Conv-polynomial-pool | 6 / 6 |
| Dual affine | 0 / 6 |
| Dual bilinear | 0 / 6 |
| Dual Linear | 0 / 6 |

First JSON parse: 34/37; Python parse: 32/37; static check, compile,
execution and numerical correctness: each 19/37. Final correctness was also
19/37: no initially failing case was recovered in this experiment.
There were 79 API attempts, zero transport retries, 1,407,163 recorded tokens
(114,437 prompt + 1,292,726 completion), and five calls with unknown usage.
Recorded tokens are not a price estimate and exclude unknown usage.

The 19 passing cases executed 76 encrypted input tuples and compared 164 values.
Maximum absolute error was 1.1359059696809694e-8, with frozen
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.
All 37 terminal key directories were cleaned by the existing retention policy;
model/program/compiler/numerical evidence remains.

Combined with previously preserved successes, the 96-case lineage reached
78 distinct passes before the new retry. This is a cross-provider/configuration
cumulative result, NOT a single homogeneous 78/96 DeepSeek experiment.
All encrypted results here use upstream SEAL HEVM CPU, not Poseidon GPU.

## Diagnosis

Confirmed: 13 final failures were static candidate checks. All received
multi-input candidates (including candidates before later TLS failures) failed
at function name, parameter signature or decorator syntax before compilation.
An actual affine example cycled through:

1. `@c(c,c)` with `def extended_dual_affine(left, right):`;
2. undecorated `def golden(left, right):`;
3. undecorated `def golden(x, y):`;
4. `@c(c,c)` with `def golden(x, y):`.

The expected header is:

```python
@hc.func("c,c")
def golden(x, y):
    # Agent must generate the actual model computation here.
```

Inference: the v3 wording "decorator is c,c" and generic error messages were
insufficiently explicit. This is an Agent-interface failure hypothesis, not
proof that every rejected candidate's computation was numerically correct.
The AST checks were not relaxed and historical answers were not auto-rewritten.

Five final provider failures were `transport_tls_failed`, after HTTP 200 during
body reception. Historical diagnostics do not identify the SSL subtype.
These are not evidence of certificate verification failures or model errors.
TLS was deliberately excluded from retries; zero retries does not mean the
configured retry feature was absent. New diagnostics retain only allowlisted
SSL classes/reasons, never exception text, keys or response bodies.
Unknown TLS and certificate failures remain nonretryable.

## Repair and validation

- New request protocol `hecate-function-synthesis-v4` contains exact 2-, 3- and
  4-input headers, one-string decorator syntax, canonical argument order, and
  explicit bans on annotations/defaults/docstrings. It still maps to the SAME
  `hecate-function-v3` AST/type/layout contract.
- v3 rules remain immutable and old requests still validate. New request IDs
  hash the clarified rules; old evidence and hashes were not rewritten.
- Signature/decorator diagnostics now show the exact expected header.
- No rule-translator answer, private reference or test vectors were added to
  the prompt. No security, numerical, compiler or backend parameter changed.
- Related host suite: 117 tests, 110 passed and 7 environment/evidence skips.
  Seven selected tests passed inside pinned Nix/Torch, including multi-input
  model/reference/FX checks and protocol checks.
- Real manual dual-input Linear regression:
  `candidate-replay-0br7h7ov`, v4 request, four input tuples, two separately
  encrypted inputs, max absolute error 8.975901466534708e-9; zero API calls.
  678,980,010 bytes of generated keys cleaned after completion.
- Three unauthenticated HTTPS HEAD probes completed certificate verification
  and returned 401 as expected without a key. This is only short-connection
  evidence, not a long-response stability gate or a credential test.

The broad host discovery also exposed a stale test assuming the default route
was DeepSeek; the credential-wiring test now selects DeepSeek explicitly and
its 11 tests pass. The offline Nix-plan check cannot acquire the serialized
launcher while the live batch owns it; it is not claimed passed during the run.
No environment reinstall was attempted for this unrelated test contention.

## New paid failure-only experiment

`/home/lhy/poseidon-work/results/agent-batch-stx1g12n/report.json`
contains exactly the 18 failed descriptors selected through the original
hash-linked report. DeepSeek official API, deepseek-v4-flash, high, 384000 output
ceiling, 900s/request, jobs=2, non-streaming, at most three extra transient
transport retries and three semantic repair rounds, unchanged from the source.

The first three dual-affine cases passed on their first model response after
the protocol repair. This supports the interface hypothesis, but is not a
completed 18-case result or a controlled whole-cohort provider comparison.
Read the current batch report for subsequent results; do not start a duplicate.

```sh
env -u DEEPSEEK_API_KEY PYTHONDONTWRITEBYTECODE=1 \
python3 scripts/baseline/run_agent_batch.py --live --extended \
  --provider deepseek --model deepseek-v4-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 900 --provider-retries 3 --jobs 2 \
  --failed-from /home/lhy/poseidon-work/results/agent-batch-9unkigdy/report.json
```

## Remaining goal boundaries

Finish and diagnose live multi-input generation before performance work.
This closes a gap between an existing deterministic/manual multi-input baseline
and live Agent usability. It does not establish complete Hecate language
coverage, arbitrary shapes/control flow, bootstrap support, unseen-family
generalization or a research contribution over the rule translator.
Poseidon GPU opcode/parameter alignment remains a separate uncompleted gate;
no GPU security/profile changes are made by this protocol repair.
