# v6 complete single-block A800 result

| Same-runner/same-window | Online median | Candidate/Cerium×100% | Runtime wall median | Main wall |
|---|---:|---:|---:|---:|
| control-B (algebra/shared/QK) |8.236990537s|135.946370%|9.716042210s|545.102443103s|
| v6 (+polynomial/+AV) |**6.303876115s**|**104.041527%**|6.490315884s|565.974913868s|
| Cerium reference, separate run |6.059s|100%|not this metric|not this metric|

Both1warmup+3 full measured samples; fixedB→v6 order, no CI/randomized order.
Online time fell23.468698%, **not process wall**. Both23nativeBoot/27Linear/6QK
per complete run; v6 also6AV, one1037303808B shared MaskBundle for all6 IDs.
All128keys/12heads/dense/nonlinear/LN/residual/nativeBoot remain. Prepared internal
actual counters reconcile; opaque callbacks are not total scalar work.

All98304complex output slots finite, L12/Q13/scale55.370070863338654. V6 max
reference error37.939775934 and RMS8.37–8.54: **accuracy has not passed**.
Processes use different random keys/profile material; neither error changes nor
combined poly+AV speedup can be assigned to a single flag. Cerium differs in HE,
approximations and graph strategy. This is NOT twelve-layer timing.

Raw online samples seconds:
- B:8.229828257,8.248761752,8.236990537.
- v6:6.538113274,6.300505808,6.303876115.

Tracked RMM peaks B23217971996B/v624635935828B; pool reserve/capacity81788928000B
(78000MiB) is separate. HostRSS58623060/63614720KiB. NativeBoot profile setup
244.759553096/243.696735372s; keygen128.804557676/159.880385241s; Linear
119.633741310/120.097695730s, v6 AV preparation5.909149858s. Not additive wall scopes.

Execution15:05:16–15:24:13UTC2026-11-15, hard deadline15:40:16. Independent15:24:56
cleanup proves0MiB/0%, actualNVMLfree81152MiB, no compute/owned native/watchdog,
both locks released, no queued task. Future admission must read **memory.free**:
81920total−0used is NOT actualfree (768MiB reserved). Nonzero utilization alone is
not rejection; later supersized-pool windows still require actualfree>=80048MiB.

## Immutable evidence

Root: `test-results/bert-a800-20261115/v6-a800/20261115T144308Z/`.
- `FINAL-REPORT.md`, `machine-summary.json`, `collected/run-{control-B,candidate-v6}/`.
- Same runner SHA256 `c34c180d4076be76ac9962b98c69002be2100cb2c83b756569d23cda46a395a0`.
- `final-evidence.tar.gz`:209272645B, SHA256
  `8ba24d3526cd3923abc599f36c8fcc05806f9d90c9c6420c78d2930e3c07a9af`.
- 171source freeze and264remote file checks, final manifest308files/archive309regular files;
  all local and remote mirrors independently verified. Scripts/inputs/results were not retroedited.
- `v6-reports-small.tar.gz` in this directory is a portable small report backup,
  not the fullsource/binary/weights/decoded evidence archive.
