# DeepSeek TLS diagnosis and three-case route experiment

## Confirmed failure layer

The completed batch `agent-batch-stx1g12n` has 15 passes and three provider
failures: `extended-dual_bilinear-3`, `extended-dual_linear-0`, and
`extended-dual_linear-1`. All three received HTTP 200, then the TLS implementation
reported `DECRYPTION_FAILED_OR_BAD_RECORD_MAC` while reading the response body.
No complete candidate was received. This is transport integrity failure, not
CKKS decryption error, a compiler error, or a demonstrated wrong DSL program.
The failed responses are never used as candidates.

Current WSL is mirrored. DeepSeek DNS resolves through a Fake-IP route, with
the default route via 198.18.0.2. Windows runs Clash Verge/Mihomo with TUN enabled,
gvisor stack, fake-ip DNS, and mixed-port **7897**. The old suggested port 6478 is
not the current listener. The existing worker ignores proxy environment variables;
that does not bypass the OS transparent route.

The actual API worker uses Nix Python 3.10.14/OpenSSL 3.0.13; host WSL Python is
3.10.12/OpenSSL 3.0.2. Neither runtime was replaced.

## Credential-free route probes

`scripts/baseline/probe_deepseek_tls.py` performs three paired rounds, comparing
the default transparent route with explicit loopback HTTP CONNECT through 7897.
Each child has a hard 20-second deadline and sends only an unauthenticated
`GET /models` to the fixed DeepSeek host. HTTP 401 is the expected probe success,
not an authenticated API success. It neither reads keys nor generates tokens.

Evidence under `/home/lhy/poseidon-work/results/`:

- `deepseek-tls-probe-2zemdomq`: host Python, 6/6 complete.
- `deepseek-tls-probe-j0l8m4f9`: pinned Nix Python, 6/6 complete.

All negotiated TLS 1.3 / TLS_AES_128_GCM_SHA256, with hostname/certificate
verification enabled and identical observed certificate SHA-256.
Short responses did **not** reproduce the long-response failure. These probes
do not establish whether a proxy/TUN, remote gateway, server, TLS library, or
another component caused the earlier integrity error.

## Explicit paid experiment

User authorized rerunning the three remaining cases. Batch
`agent-batch-3s484qfx` selects exactly those three via the hash-validated
`agent-batch-stx1g12n/report.json` lineage. Its only configuration difference
is `loopback_proxy_port: 0 -> 7897`.

Unchanged: official DeepSeek, deepseek-v4-flash, high reasoning, 384000 output
token ceiling, 900-second per-call timeout, concurrency 2, non-streaming,
up to three eligible transport retries and three semantic repair rounds.
The model's actual output length/usage is recorded separately from its ceiling.

The new explicit `--loopback-proxy-port` option:

- Defaults to zero, preserving existing behavior; never discovers a proxy.
- Accepts only an integer loopback port; is currently limited to DeepSeek.
- Sends a credential-free CONNECT request to 127.0.0.1, then uses the existing
  verified TLS context with SNI/hostname validation for api.deepseek.com.
- Sends API authorization only inside that destination TLS connection.
- Records the route in batch and candidate provider metrics and requires an
  explicit configuration-change flag when it differs from source lineage.
- Does not fall back to another route/provider, disable certificate/integrity
  checking, modify global proxy settings, force HTTP/1.1, hardcode a remote IP,
  or add integrity/certificate failures to automatic retry policy.

Reproduction of this **paid** selection (do not rerun after completion without
new authorization):

```bash
python3 scripts/baseline/run_agent_batch.py --live --extended \
  --provider deepseek --model deepseek-v4-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 900 --provider-retries 3 --jobs 2 \
  --loopback-proxy-port 7897 --allow-config-change \
  --failed-from /home/lhy/poseidon-work/results/agent-batch-stx1g12n/report.json
```

Read the batch report for terminal results. A few successful transfers support
this connection condition empirically, not a guarantee that TLS can never fail.

### Completed result

The batch completed in **689.439 seconds**: **3/3 first-attempt passes**, three
paid calls, zero transport retries and zero semantic repairs. All three complete
HTTP 200 bodies arrived intact; end-to-end HTTP times were 216.666, 236.017 and
439.849 seconds. No partial response or failed integrity check was accepted.

| Case | MAE | Maximum absolute error | Total API tokens |
|---|---:|---:|---:|
| extended-dual_bilinear-3 | 2.23947e-9 | 1.06653e-8 | 27,282 |
| extended-dual_linear-0 | 2.69239e-9 | 6.22797e-9 | 28,263 |
| extended-dual_linear-1 | 3.56586e-9 | 8.04947e-9 | 51,782 |

Total usage: 4,729 prompt + 102,598 completion = **107,327 tokens**.
Real CPU FHE validation covered 12 input tuples and 32 compared output values.
The existing terminal-key retention policy removed 2,036,940,030 bytes of
regenerable key material (about 1.90 GiB); original random keys cannot be
recovered, but fresh keys can be generated. Inputs, weights, DSL, IR, artifacts,
decrypted arrays and reports were retained.

Final batch report SHA-256:
`b9099c39555402ab7285b5c558315ccbc8e798e1ea1d546d18f302cbeada4fa0`.
This supports using the explicit loopback condition for these three cases.
The precise source of the earlier bad-MAC corruption remains **unconfirmed**;
the experiment changes a route, not just one isolated internal component.

## Continuing the correctness goal

`audit_agent_lineage.py` performs no API calls and no new FHE execution. After
the batch is terminal it follows the bounded, hash-linked failure-only lineage,
rejects repeated already-passing cases, verifies saved case identities/metrics,
checks frozen input/weight/request/artifact hashes, reloads decrypted NumPy arrays
without pickle, and compares them with saved independent references at exactly
atol=1e-5 and rtol=1e-4. It uses the already installed pinned NumPy environment.

```bash
python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-3s484qfx/report.json
```

The cumulative ledger spans providers, request versions, and connection settings.
It must not be presented as the success rate of a single unchanged experiment.
It separately reports the 16 families, unresolved cases, retained private-key
directories, and the explicit false Poseidon-GPU/full-goal completion markers.

The completed audit is `agent-lineage-audit-73raxsas/report.json`, SHA-256
`05f3f8b8e3e2d1bfff1b6d7595e711ac68818189a606d42dad5980f3bbb272aa`.
It confirms **96/96 cumulative passes, 16 families x 6**, 384 encrypted input
tuples and 1,088 compared values. Maximum absolute error across the cumulative
set is **7.546270709424263e-7**; all elementwise frozen tolerance checks pass.
No successful-case private-key directories remain. The audit itself generates
no models, calls no service, and does not repeat encryption.

The semantic inventory now links this explicit cumulative evidence. It still
marks complete semantic verification and complete goal fulfillment false.
In particular, paid cohort evidence for two inputs is not evidence for live
three/four-input generation, and local Conv lowering does not validate every
upstream HE_Conv helper.

The next GPU gate remains compiler-profile/physical-Q mapping, not package
installation: `profiled_SEAL_CPU.json` uses N32768, rescalingFactor60 and
13 active data primes; `GpuWord` is uint32 and the GPU parameter/uploader
implementation rejects values that do not fit. Existing HEAAN GPU profile is
N131072/rescalingFactor51/29 levels, not a drop-in profile for this machine or
these artifacts. Do not silently reinterpret the existing 60-bit-prime artifacts
or unlock unsupported ModswitchC/UpscaleC.

Fresh real GPU regression after migration:
`poseidon-gpu-drop-schedule-cxaz8q4s`, passed on the unchanged
N16384/Q12x30/P2x30/scale2^40/tc128 primitive profile. Four independently encrypted
inputs x ten target physical-Q states = 40 transitions; exact CPU/GPU retained
coefficient agreement, unchanged-source checks, four invalid-source-count
rejections, and decoded max absolute error **4.6519194182016495e-7**.
Its stored-evidence test also passes. This executes a manual static schedule,
not HEVM, and does not replace the compiler profile.

All changes remain local. No system configuration, dependency installation,
branch/index/sparse-checkout change, commit, push or PR is part of this work.

Final validation: host regression discovered 338 tests (266 passed, 72 skipped,
zero failures/errors); pinned-environment lineage tests passed 6/6, including
actual re-comparison of all 96 saved cases. Fresh GPU evidence audit passed 1/1.
The new CLI-route test initially treated the existing shell-quoted options string
as a list; parsing it with shlex.split fixed the fixture. No production check was
relaxed. The argparse max-repairs=4 diagnostic in the full suite is an expected
negative test. git diff --check passed; LF/CRLF warnings are not patch failures.
