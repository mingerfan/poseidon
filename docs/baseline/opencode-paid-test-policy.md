# Subsequent paid tests: OpenCode only

> Historical record only. Superseded on 2026-09-18: OpenCode Go and CommandCode
> Goat support has been removed. The policy and commands below are not active
> startup instructions. Use [the DeepSeek startup guide](../../scripts/README.md).
> Original experiment facts are retained unchanged.

The user selected OpenCode for all subsequent paid API experiments on 2026-09-08.
Use `scripts/baseline/run_opencode.py`, which explicitly selects `opencode-go`
for both single cases and the extended 96-case batch. It never falls back to
the direct DeepSeek service. DeepSeek V4 Flash is the model, not the provider.

Credentials remain in the shared, ignored project-root `.env` under
`OPENCODE_GO_API_KEY`; no new credential file is required. Existing credential
loading gives a nonempty matching process environment variable precedence.
Missing OpenCode credentials fail rather than use another provider's key.

From `/mnt/d/Code Space/Poseidon`, preview without calling any API:

```sh
python3 -B scripts/baseline/run_opencode.py --batch
```

Explicit live single case:

```sh
python3 -B scripts/baseline/run_opencode.py --case scripts/baseline/cases/linear-example.json --live
```

`--batch --live` starts a NEW 96-case experiment, not a continuation, and must
not be used to silently rerun successful paid cases. Defaults are Flash/high,
384000 maximum output tokens, 900-second API timeout, two workers. These are
requested limits, not proof that a subscription accepts them or offers unlimited
generation. Provider compatibility needs a real response before a large run.

At the changeover, the existing DeepSeek batch `agent-batch-jxowqsx2` was still
running. Its frozen runner/provider modules and evidence were not changed.
An attempt to terminate its verified process groups was rejected by the safety
review because interrupting an already-running paid batch needs explicit user
confirmation. No workaround was attempted. This new entry point does NOT stop
that batch; pending work must not be described as migrated to OpenCode.

After explicit authorization, the old batch was stopped and the generic runner
defaults were changed to `opencode-go`. Explicit provider arguments still support
historical reproduction. No running experiment was edited in place.

## Interrupted-batch handoff

`interrupted_batch.py --seal <batch>/report.json` checks that workers have stopped,
validates saved terminal case metrics against their original evidence, and writes
a new `interrupted-report.json` without altering `report.json`. The snapshot links
the original by SHA-256; pending cases remain pending, not numerical failures.

`run_opencode.py --batch --remaining-from <snapshot> --live` creates a fresh batch
containing exactly failed and unfinished descriptors. Completed successes are
excluded. Nested interrupted replacements retain the same hash-linked selection
history. Changing provider is recorded explicitly in the new report.

The DeepSeek batch `agent-batch-jxowqsx2` stopped with 42 passes, 7 provider failures,
and 47 unfinished cases. Its 54-case OpenCode replacement `agent-batch-nuh87wl1`
was paused after an initial HTTP400 to check compatibility. Before shutdown it
saved two encrypted/numerical passes (residual-4 and residual-5), one HTTP400
(residual-3), and one source-parse failure after four calls (residual-2).
Thus continuation selects 52, not 54 or 96. The HTTP400 cause is unconfirmed;
successful calls with the same settings rule out universal incompatibility.
In-flight requests during either stop may still have been billed; completed-row
usage totals do not measure every interrupted server-side request.

Active continuation at handoff: `agent-batch-5p0tp6am`, 52 selected cases, provider
`opencode-go`, Flash/high/384000/900s/two workers. Verified the 52 descriptors are
disjoint from the 44 preserved successes and their union is the original 96.
The first completed case, residual-2, passed on its first API call, including
encrypted execution and numerical comparison. This is an interim observation,
not the final batch result. Read its current report for later progress.

The configured Flash model ID and endpoint match the
[official Go endpoint table](https://opencode.ai/docs/go/#endpoints).
Current numerical execution remains upstream SEAL HEVM CPU, not Poseidon GPU.
