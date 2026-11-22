# QK/AV input packing: actual GPU diagnostic, NOT FORMAL

| Complete one-block case | Online median | Runtime wall median | Time/Cerium6.059×100% |
|---|---:|---:|---:|
| same-new-runner controlv6 |6.315600019s|6.502398136s|104.235023%|
| QKpack |5.545887620s|5.726887514s|91.531402%|
| **QKpack+AVpartial/A2** |**5.391346155s**|**5.565882934s**|**88.980791%**|

All1warmup+3complete measured intervals, fixedA→B→C order. Observed median changes
−12.187479%/−14.634459% relative to this control; **not strict-clean, equal-load
causal speedups or accuracy-preserving results**. UnequalCeriumHE/approximation/Graph.
All samples permanently shared-or-unresolved diagnostic, instrumented_diagnostic=true
(routing label, not CUPTI), formal_steady_sample=false, strict_clean_certified=false.

Raw online seconds:
- A:6.311752708,6.315600019,6.508159312.
- B:5.587461780,5.538463459,5.545887620.
- C:5.374876777,5.391346155,5.592793485.

## Actual complete work and numerical warning

Eachrun23nativeBoot/27Linear/6QK/6AV. Both packed engines really execute64CC/pair,
not merely a changed report; all128key offsets/head/features remain. Perfullrun:
- QK768→384CC/relin,128CP/pair unchanged; R2774→390, drops1548→780,
  rotations5502→3186, adds5370→3078, hoist12 unchanged.
- AV768→384CC/relin, CP2298→1914, Rescale1542→1158 (R2774→390,R4768 unchanged),
  drops4620→3852, rotations1788→1776, adds2286→1914, hoist114 unchanged.
- AV6IDs share onebundle/five reused:baseline1037303808B,partial852754432B.
  OwningQKQ9/P11 andAVQ11/P11; these persistent scopes exclude scratch/pool.

Final98304complex slots finite, but max reference error is A4.83–5.57, B5.23–5.49,
C50.68–61.54; RMS~1.18–1.20/1.22/12.72–13.10. **Full-model HE precision not passed;
combined endpoint is materially worse in this observation.** Different processes
use different real keys/Boot material, so this does not uniquely attribute the
error to A2. A later independent same-material diagnostic completed A/A/B/A/C/A,
AA decoded-f64 equal0 but B's error larger andC's smaller for that one material set;
see[its separate report](../multi_plan_validation/GPU_RESULTS.md). Not executed by
this window, and not general HE-accuracy evidence. DescriptorL12/Q13/scale55.37 is not new independent finalCTQ sampling.

All four productionN65536pair probes pass full32768slot independent oracles;
partialAV additionally matches fullAV on16384live lanes and its nonzero gaps match
the explicitly definedpartialfunction. Diagnostic pair medians:QK0.297526603→
0.167475460s, AV0.146629006→0.120891714s. Maximum oracleerrors~4.88e-10/5.99e-10/
8.80e-9/1.07e-8. 72prepared IDs/mode in AVprobes are not72executed calls.

## Setup, memory and release

Fullcase main A584.283026/B582.047352/C583.391942s; nativeprocess587.451564/
585.180273/586.702697s. Keygen178.8–181.0s, Bootprofile246.4–249.3s,
Linear115.6–116.1s each; not online or additive kernel accounting. RMMtracked peaks
24635935828/24681024596/24531078228B vs pool50331648000B(48000MiB); hostHWM
63613900/63941008/63936952KiB. Native requested/effectivelyclipped cache budget
and sampledNVML occupancy are distinct. In-flight RMMcredit remainedunknown.

Window21:01:21–21:37:21UTC,2160.385122s; all7processesexit0, harddeadline met.
Initial/pre/post3s free>=50048MiB/apps empty; inflightfree floor2048MiB, observed
fullcase minimum32717MiB.414samples/241nonempty rows all remained UNKNOWN and
unauthenticated. No PID whitelist, claimedowner/foreign inference or unknownsignals.
Independentpidfd exit/release:used0/free81152/apps0/ownprocesses0/queue0, all5locks
read-onlyEX|NB reacquired unchanged. No retry/12layer/QP/fullmodelcache experiment.

## Independent immutable evidence

Root:`test-results/bert-a800-20261115/packing-unresolved-a800/20261115T195255Z/`:
DIAGNOSTIC-REPORT.md,BOUNDED-DIAGNOSTIC-OUTCOME.json,raw/official/decoded percase,
independent-release andpost-run-integrity; bothlocal/remote verified. Oldfailed
93secondwindow208-filemanifest remains unchanged and is not resumed/relabelled.
- runnerSHA a5c45f3410fd5cfa112d3b94749016af19103408884c39342006b865bbccbccb.
- finaldiagnosticreport archive15148715B SHA256
  870587708c464c637feb5c7c77bcdb43cf82896234a035037fba2c6a006b9e3f;
  portablecopy:`history/20261115T195255Z-diagnostic-reports.tar.gz`.
- immutableassets125773284B SHA256
  ec650c284d198005ea93e281ef7f6edd6c740064022542d256cfc19618030999.

Formal-summary ingress rejects raw,official andvalidated entries in this scope.
Do not insert them into the formal historical table. Measured full12v676.417356609s
is a different unchanged c34 run; never infer a packed12 time by multiplying here.
