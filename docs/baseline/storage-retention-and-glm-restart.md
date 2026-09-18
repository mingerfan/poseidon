# Result retention and OpenCode GLM restart (2026-09-08)

> Historical record. OpenCode/GLM launch support was removed on 2026-09-18.
> Provider setup and restart commands below are no longer supported.
> Recorded cleanup and experiment results are preserved unchanged.

## Cleanup completed

Ubuntu was previously relocated to `D:\WSL\Ubuntu-22.04` with matching pre-boot
SHA256. Linux paths remain unchanged. This phase cleans inside ext4, not Windows
application directories and not the compiler installation.

Inventory found 282 generated `private-keys` directories consuming 89,177,577,003
bytes. Of these, 276 had a recognized terminal report and the SEAL 4.0.0/tc128
parameter metadata produced by `scripts/baseline/seal_keys/main.cpp`.

`result_retention.py --apply` removed only `gal.seal`, `relin.seal`, `pub.seal`,
`sec.seal`, and `parm.seal` in those 276 directories: **87,393,878,715 bytes**.
Six directories with missing reports were preserved. The helper rejects links,
hardlinked files, unknown entries, nonterminal runs and paths outside results.
No recursive deletion is used. Each run has new `key-cleanup.json` and
`key-cleanup-outcome.json` audit sidecars. All 276 original report hashes matched
their pre-cleanup values afterward.

Preserved: model descriptions, weights, test inputs, references, provider
responses, candidate source, Earth/CKKS IR, HEVM/CST, decrypted vectors, numerical
comparisons, diagnostics and batch-selection lineage. The deleted random keys
are not recoverable; rerunning `seal_golden_keys` with the saved rotation steps
generates fresh keys with the same parameter policy, not identical random keys.

Results decreased from about 84 GiB to 1.8 GiB. ext4 available space increased
from about 109 GiB to 190 GiB. Windows VHDX allocation does not necessarily shrink
immediately; new workloads can reuse the free filesystem blocks. No VHD compact,
system-package removal, LLVM rebuild, or Nix garbage collection was performed.

The candidate runner now removes these keys after the final report and outputs
are saved. Incomplete/unknown key setup is left untouched with a diagnostic.

## Verification

- 79 focused offline tests passed (retention, provider, routing, gate, selection,
  deadline, streaming diagnostics, credential handling and pause behavior).
- Non-API self-test `candidate-replay-w9mw1ziz`: deliberately malformed JSON,
  wrong numerical reduction, then correct Linear. The final candidate passed
  real isolated tracing/compilation/SEAL execution and 297,283,048 bytes of keys
  were automatically removed. This scripted test is not an Agent inference.
- Provider stream failures now retain an allowlisted subreason such as
  `missing_done`, without retaining raw error bodies or reasoning text.

## New model and remaining cohort

User explicitly requested OpenCode GLM 5.3 Flash. Verified model ID:
`glm-5.3-flash`, from the public provider model list:
https://opencode.ai/zen/go/v1/models

Endpoint remains `https://opencode.ai/zen/go/v1/chat/completions`. Credentials
continue to come from the shared ignored project `.env`, key
`OPENCODE_GO_API_KEY`; no new credential file, fallback service or direct Z.ai
API is used.

Provider-maintained model metadata confirms effort `low/high/max` and the base
model output limit of 131,072 tokens:

- https://raw.githubusercontent.com/anomalyco/models.dev/dev/providers/opencode-go/models/glm-5.3-flash.toml
- https://raw.githubusercontent.com/anomalyco/models.dev/dev/models/zhipuai/glm-5.3-flash.toml

New settings: GLM-5.3-Flash, high, output ceiling 131072, per-request deadline
1200 seconds, streaming with usage, concurrency 1. The model-specific ceiling is
not a claim of unlimited generation. Complete responses, valid usage and frozen
numerical tolerances remain mandatory; TLS verification is not relaxed.

Previous batch `agent-batch-l3tlh4wq` paused after 2 passes and one provider
`transport_invalid_stream`, with 44 pending. Its new `interrupted-report.json`
preserves the original report; source SHA256 is
`51c1df16abb7f87d27db2bdb843233fa4e6ebdd46932b5fa03ed4decd68de7c6`.
The selected restart is **45 cases**, preserving **51 existing distinct passes**.
The old DeepSeek probe is not evidence of GLM connectivity. The explicit
`--accept-connection-risk` path records the user's approved model change without
marking that probe as passed. First provider failure still pauses queued cases.

New paid batch: `/home/lhy/poseidon-work/results/agent-batch-uu27dnj_`.
Read its `report.json` for current outcomes; starting it is not completion.
Execution backend remains upstream SEAL HEVM CPU, not Poseidon GPU end-to-end.

The first paid launch was rejected by execution review. Read-only inspection
then established the exact synthetic schema-2/3 graph and public-weight scope,
request/feedback allowlists and fixed endpoint. The identical launch command was
resubmitted with that evidence and accepted; no alternate route bypassed review.
