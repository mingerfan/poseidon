# First live request diagnosis (2026-09-06)

Original evidence is preserved at
`/home/lhy/poseidon-work/results/agent-deepseek-i72pwt6x/report.json`.
The run ended `provider_failed`, with one transport request attempt and zero
candidate attempts. The recorded error was `transport_worker_failed`.
There was no generated DSL, compilation or encrypted execution in this run.
The attempt counter does not establish receipt or billing by DeepSeek.

## Evidence and fix

- No-key curl GET `/models` returned HTTP 401.
- No-key Python GET `/models` inside the pinned Nix/venv and a clean isolated
  child environment also returned HTTP 401, with worker exit 0 and no stderr.
  This confirms connectivity and TLS, not credential/model access.
- Transport code silently capped socket timeout at 20 seconds although the
  CLI's configured parent deadline defaults to 120 seconds. The worker now uses
  the configured timeout; the parent still enforces the hard total deadline.
- Worker exceptions now produce fixed allowlisted codes for DNS, certificate,
  TLS, socket timeout, trust store, connection, protocol and payload failures.
  Native messages/tracebacks and HTTP error bodies remain suppressed.
- The CLI prints final status and safe provider error code, not just the result
  directory. No automatic retry of transport failures has been added.

The original error's exact cause remains unconfirmed: its report discarded the
exception type. In particular, neither timeout nor an invalid key is proven.

## Reproduce without credentials or generation

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/deepseek_network_probe.py
```

The probe does not retain `DEEPSEEK_API_KEY` across the pure shell. It sends only
an unauthenticated GET and never requests model generation. It does not change
network, certificate, package or system configuration.

After these local fixes, rerun the original Agent command from the terminal
where the user exported their API key. Do not extract keys from another shell's
process environment. Keep new evidence separate from the failed run. A real
successful API response, compilation and decrypt/compare are still required
before claiming LLM-generated correctness. The active execution backend is the
existing SEAL HEVM CPU runtime, not a validated Poseidon GPU path.
