# OpenCode transport diagnostics and restart gate

> Historical diagnostics only. As of 2026-09-18, the OpenCode provider, probe,
> launcher and connection gate have been removed. Commands below are historical,
> not runnable instructions. Use [the DeepSeek startup guide](../../scripts/README.md).

The user requested stopping bulk tests until stable connection conditions are
established. The batch `agent-batch-5p0tp6am` was terminated and sealed into a new
`interrupted-report.json`; its original report and case artifacts remain intact.
It ended with 21 recorded outcomes: 5 passes and 16 failures, with 31 unfinished.
Including previous batches, 49 distinct cases passed and 47 remain eligible.

## Safe diagnostic contract

- Credentials stay in the shared ignored `.env`; the selected key travels only
  to the fixed HTTPS endpoint as authentication, not to model prompts or logs.
- Preserve certificate/hostname verification. No proxy discovery, redirects,
  implicit service fallback or automatic paid retries.
- Distinguish remote disconnect, incomplete response, malformed status line,
  TLS/certificate/DNS/socket errors and malformed/incomplete SSE streams.
- Record connection (DNS/TCP/TLS combined), send, response-header and body timings.
  A failed stage may have no completed-stage duration; total elapsed time and
  the active stage still identify where it failed.
- Read at most 8193 error-body bytes with a short socket deadline. Return only
  fixed error-code/parameter/hint enums, never free-form provider text. Unknown
  errors remain unknown. Do not claim that this recovers every upstream detail.
- Parent independently allowlists worker diagnostics. Exception text, arbitrary
  headers, error bodies, credentials and reasoning text never enter diagnostics.
- SSE assembly requires an explicit DONE event, preserves usage/finish metadata,
  and rejects tool calls, refusal and partial streams. The normal provider
  response validation still checks model, completion status and usage.

## Connection experiments

Use `probe_opencode_connection.py` on a sealed report. It selects three distinct
failed cases with no previously received candidate and sends their existing
synthetic/public request payloads only. Parameters remain Flash/high/384000/900s.
All trials are sequential; the first failure stops the experiment. No batch is
launched automatically. A transport pass does not count as DSL/FHE correctness.

The minimum connection smoke gate is six consecutive valid complete responses
across those three requests, under one unchanged mode. This is limited empirical
evidence, not a claim that future requests cannot fail. Before bulk restart,
use the same tested connection mode and concurrency, preserve historical
successes and rerun only the hash-linked failed/unfinished selection.

Confirmed buffered experiment `opencode-connection-q0j1qg7o`:

- one paid request, stopped at first failure;
- `flatten_linear-2`, single concurrency, `stream=false`;
- connection established in 1.625 seconds, send completed;
- `transport_remote_disconnected` while waiting for response headers;
- worker elapsed 159.988 seconds, well before local 900-second timeout;
- no valid HTTP response headers or candidate received; connection gate failed.

This narrows the failure to response waiting on the established connection. It
does not identify which gateway/upstream component closed it. Streaming control
experiment: `opencode-connection-66nckjsl`; read its report for actual outcome.
Ordinary bulk starts must not proceed while the connection gate remains false.
The explicitly approved exception below records risk acceptance separately.

`run_opencode.py` and the underlying batch runner now require an explicit
`--connection-report` before any new OpenCode bulk call. It must show six valid
single-call trials across three cases, unchanged model/wire configuration,
deadline, concurrency=1 and unchanged transport/probe source hashes. Failed or
running reports are refused before credentials are loaded. The `--stream` flag
is forwarded end-to-end; previous reports default to non-streaming in config
diffs. Probe status does not certify generated DSL or FHE correctness.

Bulk execution also stops starting queued cases at its first OpenCode provider
failure. Remaining rows stay pending, not fabricated failures. A resulting
`paused_provider_failure` report may be sealed after workers exit, retaining
the original case evidence and all earlier selection links.

The first buffered probe predates source-hash fields in the diagnostic runner.
Later probe reports include runner/provider/worker hashes and Python version;
historical reports are not edited to imply they originally recorded these.

Final streaming result: five complete HTTP200 responses with valid usage and
finish=stop (254, 628, 665, 179, 608 seconds), then `transport_timeout` at 900.103
seconds for `extended-conv1d-0`. The gate remains false. The final hard parent
timeout returned no stage diagnostics, so slow generation versus a stalled
connection cannot be distinguished from this record. None are FHE test passes.

## Explicitly approved 1200-second restart

The user subsequently approved raising the per-request deadline to 1200 seconds
and starting the remaining 47 cases despite the failed probe. Entry points and
the provider hard ceiling now support 1200 seconds. Outer case/batch deadlines
are derived from this value; the single-request deadline is not a whole-case
deadline when generation repair is needed.

`--accept-connection-risk` is an explicit, non-default exception requiring the
original failed probe and matching sealed source lineage. It accepts only the
approved Flash/high/384000, streaming, single-worker, 1200-second, 47-case scope
after five complete responses and a final 900-second timeout. The report records
`connection_gate_passed=false`, `user_accepted_connection_risk=true`, source
hashes and old/new deadlines. It does not modify the historical probe, disable
TLS validation, relax numeric checks, or disable pause-on-provider-failure.
Without this flag, the original six-success gate still applies.

Read-only topology checks also found WSL `mirrored`, `opencode.ai` resolving to
198.18.0.59, and Windows processes `clash-verge`, `clash-verge-service` and
`verge-mihomo`. This is consistent with a local Fake-IP proxy route; it is not
evidence identifying the exact component responsible for buffered disconnects.
The client does not discover HTTP proxy environment variables, but that does not
bypass OS-level transparent routing. No proxy/WSL/system setting was changed.
