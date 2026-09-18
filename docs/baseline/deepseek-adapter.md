# DeepSeek adapter: offline verification stage

Date: 2026-09-06. Source checkout: `D:\Code Space\Poseidon`.

Update: the real `--deepseek` CLI is now connected to the existing compiler,
encrypted execution and feedback loop. See [Agent quickstart](agent-quickstart.md).
The sections below document the adapter-stage tests; they are not live API results.

## Scope and evidence boundary

The existing `generate(request, feedback_history) -> raw JSON string` interface
now has a DeepSeek adapter. No API key was read, no paid request was made, and no
SDK, compiler, CUDA or system dependency was installed for this change.

This is **offline adapter validation**, not autonomous Agent generation or
feedback repair evidence. Fixtures exercise the HTTP-format boundary and local
validator; they do not simulate FHE and do not count as encrypted executions.
The existing deterministic SEAL CPU evidence remains separate from the still
unvalidated Poseidon single-GPU path.

## Files and flow

- `scripts/baseline/deepseek_provider.py`: immutable configuration, public-request
  hash/field checks, bounded conversation construction, response/usage parsing,
  per-session call limits and safe diagnostics.
- `scripts/baseline/deepseek_http_worker.py`: stdlib HTTPS worker with a fixed
  DeepSeek endpoint, normal CA/hostname validation and no automatic redirect or
  retry. Its parent applies a total process timeout, including DNS/TLS/read time.
- `scripts/baseline/candidate_contract.py`: the shared feedback loop includes a
  provider's safe metrics when available; existing replay behavior is preserved.
- `scripts/baseline/test_deepseek_provider.py` and `test_deepseek_transport.py`:
  offline fixtures and regression tests, using fake keys only.

Trusted prepared public request -> DeepSeek adapter -> untrusted response string
-> existing JSON/AST validator -> existing isolated compiler/runtime and host
numerical comparison -> bounded public diagnostics -> next generated response.

The adapter retains previous response text so a repair request includes the
program being repaired. It does not forward or store `reasoning_content`.
Compiler diagnostics stay in user-message data, never system instructions.
Nested feedback numerical fields permit aggregate errors, not reference/actual
vectors. Text diagnostics are supplied by the trusted harness: the field checks
are **not a general-purpose secret-redaction or information-flow proof**.

The v0 public constant registry and packing layout still come from the rule
translator. This remains fixed-layout constrained synthesis, not independent
Agent packing selection or a general arbitrary-Python model importer.

## Configuration and limits

`Config()` proposes `deepseek-v4-pro`, thinking enabled, effort `high`, maximum
8192 generated tokens per request, at most four requests (initial + three
repairs), and 120 seconds wall time per HTTPS worker. `deepseek-v4-flash` is also
accepted. These are initial experiment limits, not validated optimal settings.

Model IDs, thinking settings, JSON output and response usage fields follow the
[official Chat Completions reference](https://api-docs.deepseek.com/api/create-chat-completion/)
checked on 2026-09-06. The
[official changelog](https://api-docs.deepseek.com/updates/) says the model IDs
can point to updated releases. Record returned model/fingerprint and evaluation
date; this is not equivalent to having an immutable vendor model snapshot.

The adapter requests JSON output and also explicitly instructs JSON in the
prompt. A completed response's content still goes through the existing validator:
valid JSON does not establish valid DSL or model equivalence. Truncation,
refusal, unexpected tools, malformed envelopes, oversized responses, missing or
inconsistent usage, model mismatch and non-200 HTTP statuses fail closed.

HTTP/API failures stop the session without implicit retry (a timed-out request
might already have incurred a charge). Invalid candidate JSON/AST or numerical
failure is instead handled by the existing maximum-three-repair feedback loop.
The provider's request count is an additional bound, not a replacement for the
loop's repair limit.

Public request <= 128 KiB; candidate content <= 128 KiB; complete HTTP response
<= 1 MiB; accumulated request body <= 768 KiB. Conversation limits reject excess
rather than silently dropping earlier feedback. Existing source/operation limits
remain stricter where appropriate.

## No accidental live calls

`DeepSeekProvider()` has no transport and cannot send requests. The built-in
`HTTPSTransport` additionally defaults to disabled and requires an explicitly
provided key plus an approved immutable public-request hash. It does not search
environment variables, config files, credential stores or the workspace for keys.
The worker receives the explicit key via stdin, never argv or environment.
HTTP error bodies and native worker tracebacks are not emitted into logs.

The transport also checks the complete outgoing envelope against an immutable
`approved_config`: model, thinking/effort, token cap, JSON mode, non-streaming,
exact system message, message roles, repair count and feedback field allowlist.
It rejects added tools or arbitrary extra message fields even when the embedded
public request hash is unchanged. Custom provider settings must have matching
explicit transport approval. This was added after independent read-only review
and a failing offline reproduction.

`run_candidate.py --deepseek` now selects the real provider and existing
compiler/runtime evaluator. Prepare, self-test and replay remain offline modes.
Do not patch a replay result's labels to present it as a live Agent experiment.

The transport is an internal trusted-caller API, not a security boundary against
someone deliberately writing Python to bypass approval. The candidate never
receives access to it. No endpoint/proxy override is included in this first
version; an approved proxy, if needed later, requires separate bounded work.

## Accounting

- `request_attempts`: transport attempts, including offline fixtures.
- `agent_calls`: attempted sends through a live transport; not proof of server
  receipt, successful inference or a settled bill. Disabled/approval-rejected
  requests are checked before incrementing either counter.
- `calls`: safe per-call status, error code and accepted-response token usage;
  no key, candidate source, raw error body or reasoning text.
- `cost_usd: null`: actual cost is not implemented or verified. **Token/call
  caps are not a monetary budget guarantee.** Unknown usage on failed calls must
  not be treated as zero cost.
- `agent_success_rate: null`: the shared loop does not infer aggregate research
  metrics from a single session or from fixture responses.

Do not send research model structure/weights merely because CKKS labels weights
public. External disclosure still needs the user's authorization. Reference
outputs, test inputs, encryption keys and the rule-generated answer are not part
of the initial provider request.

## Reproducible offline tests

From the actual Windows source checkout:

```powershell
python -B -m unittest discover -s scripts/baseline -p 'test_deepseek*.py' -v
```

Or from WSL, without model credentials/network:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- timeout -k 3s 60s env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/baseline -p 'test_deepseek*.py' -v
```

The worker test uses a real isolated Python process only with invalid input,
which fails before HTTPS construction. HTTPS success/error fixtures run in
memory; TLS settings, fixed host/path, no redirect and bounded reads are checked
without contacting the service. The total timeout is also tested at the process
invocation boundary; no live DNS/TLS/account availability is claimed.

For the original baseline suite, use the evidence environment variables and
commands in `fx-rule-translator.md` plus `POSEIDON_CANDIDATE_RESULTS` from
`candidate-feedback-loop.md`. Checking old artifact hashes and decrypted arrays
is not a fresh run of all 48 encrypted models.

## Next gate

### Adapter-stage verification (before CLI integration)

- Windows: all 25 DeepSeek offline tests passed, including full-envelope
  rejection, worker boundary tests and fixture feedback-loop integration.
- Final WSL host regression: 129 discovered, 120 passed and 9 explicitly skipped
  because they require the pinned Torch environment.
- Those same 9 Torch tests were separately run via
  `run_model_batch.py --unit-tests` in the existing isolated environment: all
  passed. Skips were not counted as passes in the host result.
- Existing compiler/SEAL/candidate/48-model evidence hashes and numeric outputs
  were checked read-only. No full encrypted model batch was rerun this turn.
- `git diff --check` passed; existing CMake CRLF advisories remain unrelated.
- No service call, real key read, package installation, backend change, branch
  switch, commit, push or PR was performed.

The first test run failed because the adapter was absent. Additional tests
reproduced disabled-call miscounting, lost error categories and incomplete
transport envelope approval; the final regression includes all fixes. The
read-only reviewer identified the envelope issue; its follow-up session was no
longer available, so the final fix was reverified locally, not represented as a
second independent approval.

### Before live execution

The user has selected direct progress on the Agent. Set DEEPSEEK_API_KEY locally
and use the live CLI in the quickstart; no additional per-case approval dialog
is imposed. The runner records token usage (unknown billing on interruption
remains unknown), model identity and all stage outcomes. It uses the real
Dacapo/SEAL CPU path with fixed reference/weights/tests/tolerances/parameters.
Poseidon GPU remains a separate milestone. Do not put credentials in chat or Git.

All changes are local and uncommitted. A later scoped commit could contain the
adapter, worker, tests, optional provider-metrics hook and this document; it
should not sweep in earlier environment/compiler/Poseidon changes.
