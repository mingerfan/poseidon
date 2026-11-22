# Same-material nine-forward GPU diagnostic

One process/context/main keys, original3input ciphertexts encrypted once, one
complete nativeBoot build/install including hiddenQ2/P2 material; shared Linear,
three separately owning QK andAV adapters. Schedule warmABC thenA,A,B,A,C,A.
This is comparison of independent single-block plans, NOT a12layer chain.

| Plan | Online diagnostic | Wall diagnostic | Dense reference complex max / RMS |
|---|---:|---:|---:|
|A control|6.3020926485s median,4samples|6.515732047s median|7.763057864 /1.468583780|
|B QKpack|5.525627980s single sample|5.708872140s single|41.935757393 /8.559734144|
|C QK+AVpack/A2|5.502451627s single sample|5.731478513s single|5.193467797 /1.186716734|

B/C have only1diagnostic sample each, not medians or speedup evidence. No strict-clean,
formal steady or HE-accuracy acceptance; no CUPTI. All endpoints98304complexfinite.

## Numerical replay and interpretation

All10A/A decoded-f64 comparisons are BITWISE EQUAL,max/RMS0. B/B andC/C also equal.
This is decoded equality, **not ciphertext-bit equality**. All36pairs include
AA10/AB10/AC10/BB1/BC4/CC1; whole-output differences:
- AB max43.7280792233,RMS8.56876842243.
- AC max5.47799359621,RMS1.25556030444.
- BC max42.8482229981,RMS8.52377428971.

Within this one fixed material set, each plan repeats identically yet different
plans have different HE results. Earlier cross-process combined-error worsening
cannot be uniquely assigned to A2; here B has the larger reference error andC is
smaller. Neither observation proves any variant generally more accurate; **full-model
HE accuracy still fails**. Material setup is corroborated by source control flow
and runtime counts, not core RNG-draw instrumentation. No raw secret/key export.

Actual9forwards:207nativeBoot/243Linear/54QK/54AV, allmode counts checked. Original
input is reused, never last output/decryption feedback. Downloads/decrypt/save and
36pair comparisons lie outside Runtime online/wall.

## Resource/setup scopes and release

Native620.207785s/main616.565741s; setup552.707011s excludes5.161154s real
library/CPU-context audit. Detailed peradapter/keyupload/callback/endpoint scopes
are inSETUP-AND-RESOURCE-SCOPES.json; callback waits are not isolated kernels.

RMMlive28789171972B/peak30810826500B vs pool50331648000B. All6attention adapters
persist7758546952B; sharedLinear3629568764B. AV has3actual bundles/2927362048maskB.
Shared CPU keys do not eliminate these GPU copies.121monitor samples/88unknown
rows, no authenticatedNVMLownership, minfree32719MiB; liveRMMreservation/creditunknown.

Window23:15:18–23:26:01UTC2026-11-15,642.326s; work/hard23:33:18/23:35:18unchanged.
Nativeexit0, independentauthenticatedpidfdexit, used0/actualfree81152/apps0,
noowned/native/watchdog/queue,6simultaneousread-onlyEXNB locks reclaimed unchanged.
No retries, leftover-budget continuation or newGPU phase belongs to this task.

## Evidence and portable backup

Root:`test-results/bert-a800-20261115/multi-plan-a800/20261115T220156Z/`:
DIAGNOSTIC-REPORT.md,NUMERICAL-AND-TIMING.json,SETUP-AND-RESOURCE-SCOPES.json,
BOUNDED-MULTI-PLAN-OUTCOME.json,independent-release andpost-run-review.json.
RawSHA4144bd7b7044c7ad0eeb25417f19ae4833850e2fc672e837dd98d03455fc6802.
- Primary25450865B,727files(726manifest entries),SHA256
  9f15f1505059b357089ac504ac51de96e1c8b551d3201e4ef583df4b92d3201a.
- Supplement8144B,6files,SHA256
  ae3fd187e7cc75e5de8937376c8faeb88bf62ec2b0a86b9a5104e889d93c2689.
Portable copies+detached SHA files are inthis directory's history/. Source114 and
all682admitted bindings/five mature imports verified unchanged; normal freshGCC11/
CUDA11.8/SM80build. CPU86methods/162targetinvocations and final safety118normal+
118-O passed; not a cross-binary or production accuracy/performance certificate.
