# Next bounded packing validation

First attempt `packing-a800/20261115T181839Z/` stopped after baselineQK only:
19:05:50–19:07:23UTC; no candidate/full-block run. Native baseline exit0 and full
oracle passed, but NVMLPID9693 could not be attributed to containerPID83461.
Do not call that proven foreign compute or an algorithm error. Old launch/receipts
are immutable and cannot be resumed. Independent19:11:44release passed. CPU-only
namespace/ownership diagnosis is separate; any remeasurement needs a new freeze,
lease and admission, not a PID whitelist or relaxed old script.

Prerequisites: receive full12-v6's actual native/watchdog exit + fresh resource
release; independently review AV/A2 integration; freeze the final coherent source.
No GPU queue or automatic retry. Unknown NVML PID ownership must be distinguished
from proven foreign ownership, and an authenticated namespace mapping is required
for strict attribution. A separately declared margin/shared-or-unresolved diagnostic
may retain uncertainty but cannot be labeled strict-clean or disable memory checks.
Current150Python tests and39AV+36QK no-device
checks are CPU evidence only. Do not deploy localGCC15 ELF to theGCC11/CUDA11.8 card.

## Scope and isolation

- One private A800GCC11/CUDA11.8/SM80 build, existing20261114 mature libs read-only;
  no core rebuild or QP uploader replacement. New runner and both probes share the
  same frozen module sources. Explicit arch80, no `native` GPU detection during build.
- Golden1 assets copied byte-for-byte (dense28339200B/f196…/all16parameter hashes,
  original client input3591…); no seed-only data regeneration, no HE/expected output
  as input. Ratio256 profile exactf845…, T128/H768/12×64/F3072/oneencoder.
- Generate from frozen source only: control-v6 (all prior optimizations, no new
  packing), qk-only pack, and combined QK+AV pack. Baseline manifestsv1, packed
  manifestsv2; AVpartial has distinct callee and actual referenced capability.
- CPU StandardVerifier, compact audit, every input/W/manifest/call/scale/key union,
  original7/10artifact regression; default QK/AV math remains separate from new modes.

## Resource window

A new explicit bounded lease with independent owned-PGID watchdog/cleanup and no
ACK gate; UTC start/work/hard times published before12-second automatic release.
All native setup, runs and3-second exit settles count against the lease. No
extension, shrinking workload or blind rerun. Stop subsequent stages on any native
error, bad counters, nonfinite output or insufficient remaining budget.

Right-size a NEW single-block window to **48000MiB managed pool** (not the old
78000MiB allocation). Measured oldv6 tracked peak24635935828B motivates this
candidate capacity; it does NOT certify future peak or fragmentation. Matched
controls use the same cap. Actual NVML `memory.free>=50048MiB` (pool+2048outside)
is required. Never use total-used: observed reserved768MiB. Utilization alone does
not reject. Formal A/B uses no visible foreign compute; if a shared diagnostic is
chosen explicitly, size separately and label it, not strict clean A/B.

OMP/BLAS4, taskset32-35, binary ordinary rotations + actual prepared direct keys,
host-only ordinary PT cache, EAGER loading,4GiB Boot workspace remain. No async
allocator, CUDA Graph, profiler capture or altered HE/approximation. At exit settle
exactly3seconds then one fresh resource check; no busy polling/resource queue,
no signals to other users. Preserve raw telemetry and later foreign load as facts.

## GPU functional gates before full model

1. New-source baseline QK and packed-input QK, each1warmup+3pair runs, complete
   encrypt→GPU→decrypt vs independent all32768slot oracle. Post-CC paired2 can be
   omitted: it is not the selected frontend path and has no claimed GPU benefit.
2. Baseline AV and PackedInput2Partial AV, full32768definedpartial oracle plus
   separate16384live completeAV comparison. Gap magnitude is NOT reference error.
   Use `--prepare-both-algorithms` to exercise genuine mixed-key reupload/views and
   mode-specific72-ID one-bundle sharing if supported by the frozen probe. Only
   executed IDs/runs count;72prepared IDs do not imply72executions.
3. Actual scale/parms/component counts and every mode-specific evaluator counter,
   finite output, direct rather than decomposed rotations must match. Do not use a
   zero-error old gap0 oracle for the new partial operator.

## Complete one-block comparison

Same new runner, original input/W, each1warmup+3complete measured iterations:
control-v6 → QKpack → combinedQK/AVpack (fixed order disclosed, no CI/single-flag
causality beyond the actual matched configurations). Setup/profile/keygen/main
and whole-native wall remain separate; online is synchronized execution+finalize,
Runtime wall includes per-run init/setup. No instrumented samples in formal table.
A bounded lower-cost alternative is a validated same-material diagnostic library,
but its per-step times stay **diagnostic**, not these formal steady samples.

Every run:23actualnativeBoot/27Linear/6QK/6AV, last3CT L12/Q13/scale55.370070863338654.
Partial numerator may have nonzero gap; only the real masked reciprocal/context CC
makes complete hidden semantics equal. Do not decrypt-refresh or fake gap/scale.

| Actual prepared work expected PER FULL RUN | controlv6 | QKpack only | combined |
|---|---:|---:|---:|
| QK CC/relin |768|384|384|
| QK CP / R2 / drops |768/774/1548|768/390/780|768/390/780|
| QK rot / add / hoist |5502/5370/12|3186/3078/12|3186/3078/12|
| AV CC/relin |768|768|384|
| AV CP |2298|2298|1914|
| AV R2 / R4 / drops |774/768/4620|774/768/4620|390/768/3852|
| AV rot / add / hoist |1788/2286/114|1788/2286/114|1776/1914/114|

These are expectations, not observations. Ordinary88CC/209CP/291Rescale/244rot and
Linear55188CP terms are CPU inventories unless the API exposes actual counters.
Total staticCC1624→1240→856 must not be described as opaque39work items. Serial
QK depth stays4primes; AV scoreR4 remains real. No inference from counts to seconds.

Each single-mode AV adapter must have one shared bundle for6IDs, five reused flags:
baseline1037303808B, partial852754432B (coefficient storage scope, not live peak).
Report actual owning Q11/P11 and QKQ9/P11 keys, persistent vs scratch, trackedRMM
live/peak vs pool reserve/occupancy and hostRSS. A per-plan diagnostic may legitimately
have multiple adapters/bundles; count total actual storage rather than saying one
global bundle or treating shared CPU material as zero GPU copies.

## Result and cleanup evidence

Archive independent source/bin/plan/spec/fixture/profile/input/W hashes, command/env,
raw samples/counters/failures/decoded final endpoints; external archive SHA receipts,
independent release verification (process identities, locks, actual NVMLfree/apps).
Report candidate/Cerium6.059s×100% for singleblock only, unequalHE/approximation/Graph.
Never divide by6.059×12. Full-model precision remains unpassed even with finite
slots and correct standalone probes. New modes change CKKS noise propagation;
masked seed coefficient2^-14 atPT55.43 has only~41.43-bit amplitude budget.
