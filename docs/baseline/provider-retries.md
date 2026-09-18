# Bounded provider retries (2026-09-08)

> Historical GLM experiment record. The OpenCode launcher and restart gate
> described below were removed on 2026-09-18; the reproduction command is retired.
> DeepSeek retains bounded transport retries via `--provider-retries 3`
> (default zero), distinct from DSL repairs. DeepSeek batches pause on model
> identity mismatch; other provider failures are reported per case. Current
> launch commands are in [the Agent README](../../scripts/README.md).

The first GLM batch `agent-batch-uu27dnj_` paused after 4/45 cases:
3 encrypted numerical passes and one provider failure (`extended-conv1d-3`),
leaving 41 unstarted. The failure is `transport_invalid_stream/missing_done`:
HTTP 200, 122.245 seconds, 1,319,289 response bytes, 6,488 SSE events, no `[DONE]`.
No incomplete answer was accepted. All 3 passing cases retain their evidence.
Across the historical 96-case lineage, 54 distinct cases have passed;
42 remain (one failed and 41 unstarted in this cohort).

The backend for these passes is upstream Dacapo/SEAL HEVM CPU, not Poseidon GPU.
This is finite-input numerical validation, not a proof for all inputs.

## Policy

- OpenCode entry point defaults to `--provider-retries 3`; generic runners and
  the provider Config default to zero for backward-compatible, opt-in use.
- Each logical generation can make one initial request and three extra attempts,
  waiting 5, 15 and 30 seconds. Every attempt retains the identical public request,
  model settings and existing semantic-repair history; partial answers are discarded.
- Retry timeouts, disconnects, incomplete reads, connection/DNS failures,
  missing-DONE streams and HTTP 408/500/502/503/504.
- Do not retry malformed streams, model mismatch, refusal, token truncation,
  invalid usage, authentication/permission/payment errors, TLS/certificate errors,
  or HTTP 429 whose quota versus temporary-rate-limit cause is not established.
- Each attempt remains bounded by 1200 seconds. Up to 4 logical generations
  (initial + 3 DSL repairs) can therefore consume at most 16 API attempts per case.
  API/backoff allowance is 19,400 seconds per case at these settings, plus the
  existing 1,800-second compile/execution allowance. Outer deadlines include this
  budget; they do not silently cut off an authorized retry early.
- The provider checkpoints a sanitized attempt ledger before dispatch and after
  each outcome. `agent_calls` counts attempted live posts, not confirmed billing.
  `generation_attempts` and `transport_retries` are separate; semantic repairs
  are generations minus one. Missing usage remains unknown, never zero cost.
- Nonretryable failures or exhaustion still pause the batch to avoid repeatedly
  charging subsequent cases through a broken service. This is not infinite retry.

## Evidence and retention

Original batch reports, requests, generated programs, diagnostics and numerical
outputs are preserved. Successful cases are excluded through the existing
hash-linked remaining-case selector. Terminal-case key cleanup continues; retries
inside one case reuse its existing key set and do not create extra case directories.

`agent-batch-uu27dnj_/interrupted-report.json` is the sealed 42-case source;
its SHA-256 is `f6124a888d787a09e7f78d663b39ae51095f2d1d90ab65d727d0f8c021063363`.
Its legacy reason label mentions provider switching; no provider was switched
when sealing it. Future seals use a generic remaining-case restart reason.

GLM continuation validates ancestors back to the approved 45-case cohort.
The historical six-trial DeepSeek connection probe remains **failed**; user
acceptance of a bounded retry experiment does not establish stable GLM connectivity.

## Reproduction

From `/mnt/d/Code Space/Poseidon`, with the ignored shared `.env` configured:

```sh
python3 scripts/baseline/run_opencode.py --batch --live \
  --remaining-from /home/lhy/poseidon-work/results/agent-batch-uu27dnj_/interrupted-report.json \
  --model glm-5.3-flash --reasoning-effort high --max-tokens 131072 \
  --api-timeout 1200 --provider-retries 3 --stream --jobs 1 \
  --connection-report /home/lhy/poseidon-work/results/opencode-connection-66nckjsl/report.json \
  --accept-connection-risk
```

This command starts a new paid batch; do not launch another copy while it runs.
Keys are never placed in argv, reports, generated programs or compiler processes.

Offline checks cover recovery, exhaustion, unchanged requests/repair history,
nonretryable errors, separated counters, deadline propagation, legacy reports,
lineage tampering and key-retention boundaries. Fixtures do not count as API/FHE
successes. Do not claim the remaining 42 cases passed before their reports exist.
