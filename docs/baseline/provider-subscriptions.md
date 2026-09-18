# DeepSeek provider configuration

Updated 2026-09-18. The Agent now supports only the official DeepSeek API.
OpenCode Go and CommandCode Goat adapters, routes, launchers and credential
variables have been removed. Historical experiment reports are retained without
changing their provider identities or numerical results.

## Credential

Configure only this variable in the backend checkout's project-root `.env`:

```dotenv
DEEPSEEK_API_KEY=your-local-key
```

The root `.env.example` contains no real credential. Never commit `.env`.
A nonempty `DEEPSEEK_API_KEY` process environment variable overrides the file.
Other provider keys are not accepted and there is no credential/provider fallback.

## Invocation

See [Agent startup and accepted inputs](../../scripts/README.md).
The provider argument may be omitted or set to `--provider deepseek`.
The fixed HTTPS endpoint is `https://api.deepseek.com/chat/completions`.
The default model remains `deepseek-flash`, with `high` reasoning and
10 API workers for batches. Transport retries remain opt-in through
`--provider-retries 3`; this is separate from at most three DSL repair rounds.

Each real request requires the explicit `--live` (alias `--deepseek`) switch.
Removing subscription support does not authorize new paid calls, change DSL
semantics, rerun old experiments or establish Poseidon GPU execution.
