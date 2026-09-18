# DeepSeek remaining-case restart, 2026-09-08

User explicitly requested stopping OpenCode, restoring the former DeepSeek API
test configuration and concurrency, and finishing the remaining cases. This
supersedes the earlier OpenCode-only provider preference for this restart.

## Stop and preserve

- Stopped the verified `run_opencode.py` process tree rooted at WSL PID 470;
  all ten identified processes exited. No distribution shutdown or unrelated
  process termination was used.
- Source batch: `/home/lhy/poseidon-work/results/agent-batch-evsizeeq`.
- Five terminal cases passed and remain preserved; 37 cases remain.
- Sealed `interrupted-report.json` without replacing the original checkpoint.
  The running case `extended-conv2d-3` is included in the remaining selection;
  its interrupted API call may have incurred usage not available locally.
- The historical 96-case lineage now has 59 distinct passing cases before this
  restart. Already passing cases are not sent to DeepSeek again.

## Recovered settings and deliberate difference

Historical evidence: `agent-batch-b1cv_1rd/report.json` (48/48 passed) and
`agent-batch-jxowqsx2/report.json` (96-case run interrupted for provider switching).

| Setting | Restored value |
|---|---|
| Provider | `deepseek`, official `api.deepseek.com/chat/completions` |
| Model | `deepseek-v4-flash` |
| Reasoning | `high` |
| Output token ceiling | 384000 |
| Per-request deadline | 900 seconds |
| Concurrency | 2 |
| Streaming | false, matching the historical non-streaming configuration |
| DSL repair rounds | at most 3 |
| Transient transport retries | at most 3 extra attempts, retained from latest user request |

The retry policy is the one deliberate addition to the historical settings;
request, candidate, compiler and security validation remain unchanged. This is
at most 16 API attempts per case when every repair and retry is used. Backoff
and outer deadlines are bounded. A successful return still has to pass the
unchanged encrypted numerical checks; retries cannot guarantee correctness.

## Started batch and reproduction

New batch: `/home/lhy/poseidon-work/results/agent-batch-9unkigdy`.
Its report records the provider/configuration delta and the source snapshot hash.
No claim of completion is made by this launch record; read its current report.

Run from `/mnt/d/Code Space/Poseidon` (do not launch a duplicate while active):

```sh
env -u DEEPSEEK_API_KEY PYTHONDONTWRITEBYTECODE=1 \
python3 scripts/baseline/run_agent_batch.py --live --extended \
  --provider deepseek --model deepseek-v4-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 900 --provider-retries 3 --jobs 2 \
  --remaining-from /home/lhy/poseidon-work/results/agent-batch-evsizeeq/interrupted-report.json \
  --allow-config-change
```

The outer process reads `DEEPSEEK_API_KEY` from the existing shared ignored
project `.env`; the command unsets a possible inherited key to avoid stale
provider credentials. The key is not logged or placed in command arguments.
No OpenCode-specific entry point or failed OpenCode probe is used for DeepSeek.

Backend: Dacapo/SEAL HEVM CPU encrypted execution, not Poseidon GPU. Terminal
case key cleanup remains enabled; interrupted evidence is preserved separately.
No compiler rebuild, dependency install, branch change, commit or remote write
is part of this restart.
