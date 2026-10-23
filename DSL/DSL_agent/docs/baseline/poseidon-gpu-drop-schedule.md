# Explicit physical-Q drop through the real Poseidon GPU static executor

## Confirmed increment

`MgpuOpKind::DropModulus` / `drop_modulus` now exists in the optional static IR.
Its two mandatory attributes are **physical prime counts**, not compiler levels:

```text
source_q_count = 12
target_q_count = 2
```

Both must be positive `int` values, and target must be strictly less than source.
The operation preserves scale, NTT form and retained RNS coefficients. It is not
Rescale and is not a zero-count identity alias for HEVM opcode 4.

The verifier checks arity, ciphertext kind, device/SSA rules and attributes.
Preflight checks the attribute contract without CUDA. At execution the backend
also checks the bound ciphertext's actual Q count, P count and context parameter
id before calling the already verified `GpuEvaluator::drop_modulus`. Bad physical
bindings do not publish an output. Existing full-single-device restrictions stay.

Integration covers kind strings/JSON, summaries, placement/copy insertion,
static scheduler value tracking, sequential execution and GPU dispatch. The enum
was appended, preserving previous numeric values. The latency key is separate:
`drop_modulus_single`, with the existing generic fallback when no measurement
exists; no speedup or measured cost claim is made.

**HEVM opcode 4/5 remains rejected.** The adapter does not yet infer physical
counts from compiler logical levels. This increment supplies the verified static
runtime operation needed for an explicit compiler-profile mapping; it does not
pretend the existing 60-bit-prime SEAL profile is GPU compatible.

## Real execution evidence

Directory: `/home/lhy/poseidon-work/results/poseidon-gpu-drop-schedule-jyj4j4r6`.
The report includes sources, binary/library hashes, linkage checks and captured
execution output. Backend is `poseidon_gpu_static_schedule`, not SEAL or HEVM.

Actual path:

```text
manual static schedule -> JSON encode/decode -> preflight
-> SequentialScheduleExecutor -> PoseidonGpuExecutionBackend
-> upload -> Add -> DropModulus -> download -> decrypt/decode -> compare
```

- The same tc128-checked N16384/Q12x30/P2x30/scale2^40 primitive profile is used.
- Four independently encrypted input vectors: zero, signed fixed pattern,
  fixed-seed bounded pseudorandom, and alternating +/-0.25 boundaries.
- For each, target physical Q counts 11..2 are executed: 40 transitions total.
- Reference is independent `2*x`. GPU output coefficients/metadata also match
  CPU Add/drop from the same ciphertext exactly. Source and Add-output downloads
  confirm no mutation of the earlier values.
- All decoded slots pass `abs(actual-reference) <= 1e-5+1e-4*abs(reference)`.
  Largest absolute error across all 40 transitions: **5.104296109962256e-7**.
- Four declared-source-Q mismatches are rejected by runtime, without publishing
  the output or performing the later downloads.

These are four input vectors under one short graph, not 40 distinct models.
`gpu_executed=true`, `fhe_executed=true`, `hevm_executed=false`, `agent_calls=0`.

## Static tests and preserved behavior

New CPU tests cover valid/invalid count ranges, missing attributes, arity, type,
device and duplicate-output errors, JSON round trip, summary counts and static
pipeline behavior. SingleDevice and GreedyReady pass. Existing HEFT/PEFT remain
unimplemented and explicitly rejected: an initial test incorrectly expected
those enum names to imply implementation, and was corrected after inspecting
`schedule_static`; no scheduler feature or validation was disabled.

Fourteen targeted CTests pass: new physical-drop contract and real GPU schedule,
previous GPU primitive, IR, schedule JSON, runtime, static pipeline, placement,
copy insertion, static scheduler, topology, GPU schedule/aggregate preflight and
Dacapo adapter. The older security-disabled GPU smoke was built, not run.

The concurrent full Python regression is **not all green**: 284 tests ran with
25 skips and one `OfflineNixPlanTests.setUpClass` error. That class invokes the
serialized Nix launcher and its 30-second lock wait failed while the live batch
held `launcher.lock` (confirmed with `lslocks`). No lock was released and no test
was disabled. Rerun that class/full suite after the paid batch finishes; do not
describe this run as a full-suite pass. The standalone new GPU-evidence and
semantic-inventory tests pass without acquiring that lock.

## Concurrent build and Agent experiment

An online 96-case DeepSeek experiment is independently running in
`/home/lhy/poseidon-work/results/agent-batch-jxowqsx2`. That directory is the source
for its status; this document is not its completed result. Agent generation,
candidate checking and SEAL execution scripts are unchanged during the batch.
The semantic inventory is only imported by its offline inventory tests, not by
the live generation path; recording GPU evidence there does not change prompts.

The Nix portable launcher owns mutable shared `tmpbin`, so invoking it twice is
not safe. GPU builds can instead use `--native-tools`: a temporary bubblewrap
root, read-only cached Nix tool tree/source/system mounts, writable regenerable
work directory, empty inherited environment and disabled network. It invokes
only the already installed CMake binary, not the Nix launcher or database.
It does not create a host `/nix` directory or modify the Agent namespace.
The initial root-bind attempt failed with `Can't mkdir /nix`; using a temporary
namespace root fixed this without host system changes. Configuration evidence:
`poseidon-gpu-build-vhbso3kb`; build evidence: `poseidon-gpu-build-0wbr75s9`.

## Commands

From `/mnt/d/Code Space/Poseidon` in WSL:

```bash
timeout -k 3s 220s python3 scripts/baseline/build_poseidon_gpu.py --native-tools
timeout -k 3s 1850s python3 scripts/baseline/build_poseidon_gpu.py --native-tools --build
timeout -k 3s 170s python3 scripts/baseline/run_gpu_drop_schedule.py

POSEIDON_GPU_DROP_SCHEDULE_RESULTS=/home/lhy/poseidon-work/results/poseidon-gpu-drop-schedule-jyj4j4r6 \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s scripts/baseline -p test_gpu_drop_schedule.py -v
```

The next gate remains an explicit HEVM/compiler-profile mapping and an actual
compiled Linear artifact on GPU, including its required multiply, keys and
rescale semantics. No bootstrap substitution, new CPU HEVM backend, full ResNet,
multi-GPU execution, git commit, push or PR is part of this increment.
