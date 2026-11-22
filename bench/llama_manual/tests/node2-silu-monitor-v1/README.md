# node2 public SiLU monitor v1 — frozen structural failure

**No numerical, model, or performance promotion. No retry or repair was performed.**

## Outcome

- True registered child: namespace PID/PGID `95580`, start ticks `1373031700`; NVML host PID `747769`. Supervisor was PID/PGID `95575`, not the CUDA PID.
- Child `returncode=1`, `abort_reason=null`, `requested_signals=[]`; guard classification `structural_or_resource_failure`. No cleanup error or possibly-live-owned-process flag was emitted.
- `stdout.log:33`: `PUBLIC_SILU_PAIR_FAILED projection actual N/Q/id/scale/components/device mismatch`.
- Observed **16 physical Boot counter events / 8 projection-pair events / 8 refresh rows / 0 completed cases**, not the intended 32/16/16/2. No DONE record. `SUMMARY.json` exists but is **0 bytes**.
- The event prefix is first-case paired OFF four calls, ON four calls, unary A four calls, unary B four calls. Local source places the following closed final shape checks at `bench/llama_manual/tests/public_paired_silu.cpp:179-186`, before OFF/ON exact-RNS and final numerical validation. The generic exception does **not** identify the failing output or individual metadata field; neither is inferred.
- Eight observed refresh rows independently meet the unchanged `1e-3` complex-error gates: maximum identity error **7.83342521243629e-05**, maximum real-sine-oracle error **7.794571379446548e-05**. Their refresh metadata is Q6/scale2^40 to Q27/scale2^50 and immutable=true. This is only an observed prefix, not whole-control qualification.
- Final source-oracle maxima, OFF/ON metadata/RNS equality, final original-input/output immutability, all-slot finiteness and Q10/scale2^40 final metadata are **unestablished**, not passes. The closed shape predicate explicitly failed.

## Timing and sampled resources

- `SOURCE_SETUP stage=completed seconds=1.696275776` is its own source-reported setup marker.
- Public runtime/profile setup: **84.933210287 s**. Guard wall: **94.33101697638631 s**.
- No completed-case diagnostic wall was printed. Do not subtract walls to invent one. No clean online/GPU-busy/performance timing exists.
- Guard telemetry: **45 noncontinuous samples**, **29–34°C**, maximum sampled memory **33227 MiB**, only host PID `747769`, **0 foreign samples**. First sample at 0.1231037043 s; maximum gap 2.1832620483 s; final sample-to-terminal tail 2.0006306674 s. Clocks were not sampled. No-foreign samples are not continuous exclusivity proof.
- **One postflight**, `2026-11-16T21:11:33.345493+08:00`: expected supervisor and registered namespace child absent; exact target `GPU-499236d3-b46c-5d25-4a22-530d47ed5112`, **29°C / 0 MiB / no compute PID**. This is a single observation, not proof of cooperative lease release.

## Collection and provenance

One existing-master, noninteractive SSH succeeded, with terminal present at the first tiny check. Raw allowed public files were captured once and transport SHA256-checked; no other remote files were collected. One bounded read-only Python observer and two postflight `nvidia-smi` queries ran; **no GPU/CPU application launch**, remote writes, locks/auth changes, retries, reconnects, cleanup/signals, or peer contact. Main's earlier launch-SSH success is separate from the application's failure. Detached outer exit status was not independently waitpid-observed; the guard's terminal and final printed record agree.

Authorization SHA256: `ea218a2d67a0e04f5f439286b3784a2c915a7bbbb0e6682bbd68549929bb0d13`. Captured command, authorized environment, eight existing lease identities and supervisor identity match the local Main authorization/launch records. `PROVENANCE.json` records the executable/backend/CPU library pins, source inventory pin, framework base `319d811959949e1ea9250d2d48dec8362e6d7729`, local source hashes, and read-only reference parser hashes. Monitor-side remote binary/source rehash was not authorized/performed; it relies on the guard's pre-admission checks and frozen-root assumption.

The old common CPU helper's `physical_Boots=40` is **CPU/planned schema metadata**, not actual GPU work. The intended 64 EvalMod calls are derived-only and are not reported as completed for this prefix. The oracle is the source G/G/F/F/PQ approximation, **not true/HF SiLU**. The separately reported source-versus-true-SiLU gap `0.48781247052462845` is nongating. Refresh identity uses full complex preinput; real-sine uses `factor*sin(real(preinput)/factor)` with complex output error, **not complex sine**. Public Gate ±25 / Up ±8 does not qualify full32 model range.

## Artifacts

- `MONITOR-REPORT.json`: compact aggregate outcome and limitations.
- `PARSED.json`, `FAIL.json`, `CASES.csv`, `REFRESHES.csv`: derived parser results; no case rows were fabricated.
- `command.json`, `pid-map.json`, `stdout.log`, `gpu.jsonl`, `terminal.json`, `SUMMARY.json`, `guard-launch.log`: frozen remote captures.
- `postflight.json`, `CAPTURE-MANIFEST.json`, `COLLECTION.json`, `capture.stdout.json`, `capture.stderr.log`: collection/postflight evidence.
- `parse.py`, `test_parse.py`: local adaptation of the old read-only parser; fixed numerical gates retained, new guard schema/provenance and orphan refresh validation added.
- Final **27/27 pure local synthetic tests passed**, including the retained 18 reference tests and nine node2 cases. `parser-tests-final-v2.stderr.log` is the final validation log. The initial test log preserves a local adapter-only pending-terminal comparison failure, fixed without changing numerical gates.
- `FINAL-INVENTORY.json`: SHA256 and sizes for retained local artifacts (excluding itself).

All new writes remain in this directory. No private weights, keys, tensors, CT/PT directories or arrays were read or collected. Do not rerun `collect.py`: this authorization covered a single observation/collection only.
