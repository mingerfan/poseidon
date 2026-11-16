# Controlled same-process multi-plan diagnostic

`bert_multi_plan_runner` is a separate bench executable. It includes the existing
`../native_runner.cpp` helpers in one translation unit with its legacy `main`
renamed; it does **not** modify the single-plan runner, backend, core, PRNG, or
bootstrap implementation.

This compares 2–3 compatible candidate plans using **one real context, one main
key-generation setup, one encryption of the three original Host input tiles,
and one complete native bootstrap profile installation**. This includes the
random material of bootstrap subcontexts, not merely a shared seed. Linear is
prepared once and shared. QK/AV adapters are prepared separately for each plan,
using the same main CPU key objects, but retaining independent GPU uploads and
constant bundles; equal numeric IDs across plans do not share an engine cache. Every scheduled run starts
from the same immutable encrypted original input; final ciphertexts are never
fed back. There is no secret-key-assisted refresh or repeated fixed-seed noise.

This is **not old/new binary pairing**, not a chained 12-layer inference, and not
an established performance/accuracy result. Even `phase: "measured"` remains an
instrumented diagnostic: all runs get endpoint download/decryption and all-pairs
slot-difference summaries, outside Runtime online/wall intervals. Numerical
acceptance is not enforced; nonfinite values and reference errors are reported.

## Build

See [the portable build guide](../BUILDING.md). From the repository root:

```sh
bash bench/bert_dsl/multi_plan/build-local.sh build -- -DCMAKE_CUDA_ARCHITECTURES=80
# Or build both runners together:
cmake -S bench/bert_dsl -B build-bert -DBERT_BUILD_MULTI_PLAN=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-bert --parallel 2
```

The standalone wrapper defaults to `build-bert-multi-plan/`, uses the current
coherent bench source, and respects the normal CMake toolchain/environment.
SOURCE mode builds dependencies; PREBUILT mode requires explicit matching
libraries/config headers. It no longer selects historical snapshots, GCC15,
CUDA13.4, old reader archives, or MPI/NCCL automatically.
`BERT_MULTI_PLAN_BENCH_ROOT` remains an explicit override for a coherent source
snapshot; do not mix helper and adapter revisions. Imported readers must support
`ApiCall`, not merely link successfully. Cross-toolchain ABI compatibility is
not established. Historical frozen-source qualification/results describe their
original artifacts, not a fresh build from this updated entry point.

## CLI and schedule

```sh
BIN=build-bert-multi-plan/bin/bert_multi_plan_runner
$BIN --help
$BIN --check-only schedule.json check-report.json
# Only this explicit command enables device work (not run during CPU validation):
$BIN --run-gpu schedule.json gpu-report.json \
  --native-boot-workspace-bytes 4294967296 \
  --host-cache-budget-bytes 8589934592 \
  --decoded-dir decoded-runs
```

No args, `--help`, and `--check-only` make no CUDA **device** calls. CUDA-linked
static fatbinary registration is present and is not a device execution claim.
The preloaded guard is a symbol-interposition test aid, not a sandbox.

```json
{
  "plans": [
    {"name": "A", "plan": "A/model.runtime-plan.json", "spec": "A/operator-spec.json", "fixture": "A/fixture.json", "bundle": "A/bundle"},
    {"name": "B", "plan": "B/model.runtime-plan.json", "spec": "B/operator-spec.json", "fixture": "B/fixture.json", "bundle": "B/bundle"}
  ],
  "schedule": [
    {"name": "A", "phase": "warmup"},
    {"name": "A", "phase": "diagnostic"},
    {"name": "B", "phase": "diagnostic"},
    {"name": "A", "phase": "measured"}
  ]
}
```

Paths are relative to the schedule file, not the process working directory.
Omit `bundle` only when the plan has no bundle reference. Limit: 1–64 scheduled
runs, exactly one rank and one visible device, exactly one native Boot profile.
The mixed-mode `attention-modes/v6-v7-v8.schedule.json` uses v6 baseline, v7
QK-packed, and v8 QK-packed+AV-partial candidates. It warms each plan then
schedules A/A/B/A/C/A diagnostics. It is ready only for a **separate future**
GPU validation; no GPU replay or performance/accuracy result is established.

## Fail-closed audit and ownership

- Every plan gets the standard `RuntimePlanReader`, `OperatorSpecReader`, and
  `PlanVerifier` with digest checks enabled. Raw plan/spec/fixture and prepared
  manifest SHA256 are recorded per candidate. Source snapshots are checked
  again around GPU setup.
- Context including explicit Q/P primes, native Boot configuration/contracts,
  target, Rescale execution configuration, three initial Host IDs/descriptors,
  and ordered final physical descriptors must agree.
- The entire `fixture.model.config` is also a shared semantic contract, including
  LayerNorm epsilon/variance center, reciprocal iterations, layers, attention
  geometry and intermediate width. `model.seed` is also compared as a nonnegative
  JSON integer, in addition to actual input and all parameter hashes. Required fields have conservative type and
  dimension checks; `layers` must be a positive JSON integer, and parameter-hash
  keys must be exactly the canonical strings `0..layers-1`. Optional `head_dim`
  must equal `hidden/heads`. Model-top-level experiment flags are not part of
  this config contract and may differ. Known QK/AV mode flags and required
  capabilities are nevertheless checked against the authoritative frozen
  adapter parser and the actual referenced Runtime calls. Config and its hash
  are recorded per candidate.
- The actual original input is unpacked to canonical little-endian f64 and
  rehashed. Packed values are compared explicitly, with a packed-byte hash.
  Every layer's full set of 16 parameter hashes must agree, not just its seed.
  The actual dense file and its individual matrix/bias ranges are rehashed.
  LN hashes are compared as trusted fixture declarations: raw LN arrays are
  not supplied here and cannot be independently rehashed by this helper.
- Complete Linear matrix/call contracts must match across plans, with exact
  call IDs, contiguous output0-first matrix sequences and unchanged operands.
  No complex Linear remapping is attempted. Attention uses separate per-plan
  engine namespaces: QK v1 baseline versus v2 `pack_inputs`, and AV baseline
  versus v2 `PackedInput2Partial`, may reuse the same physical ID/call ID across
  different plans. Only these known modes are admitted; post-product QK pairing
  is outside this bounded runner. AV partial must use
  `bert.compact_av_packed_partial`, not the full-output callee. Within each plan,
  duplicate IDs, contradictory modes/callees, false capabilities, unused rows,
  and physical descriptor mismatches are rejected. The frozen QK parser/key
  planner and AV parser/key planner/context/headroom validator are reused;
  QK's real-prime scale reference also validates intermediate scales. Actual
  engine rotation requirements are canonicalized and checked against manifests
  and each fixture union. Different attention contracts are conservatively
  limited to `layers=1`; identical-contract multi-layer libraries are not newly
  rejected. Linear retains a shared `callee + call_id` contract namespace.
- CPU validation constructs at most one real CPU context for authoritative
  attention checks; GPU mode reuses that context rather than constructing a
  second one. This CPU work is in library-load/audit time, not free setup work.
- Verified plan capabilities are concatenated and key requirements sorted and
  deduplicated for **one real API union preflight**. Standard `Runtime.run`
  still verifies each plan and calls preflight; the derived API checks its
  audited identity/requirement allowlist without repeating materialization.
  Main CPU Galois keys contain the policy basis plus the real prepared-direct
  union. Opaque prepared engines upload their own real cropped keys; these are
  not fabricated metadata requirements of the ordinary API.
- All plans retain separate real **Host-only CPU-encoded** constant maps, even
  when ValueIds collide across plans. The active map is moved on each switch;
  there is no ID-only cross-plan sharing. Total retained coefficient bytes are
  charged, not only the current plan. Original Device uploads remain real and
  in their declared Runtime phases. The reported host bytes exclude allocator
  and object overhead, not an end-to-end memory admission. Waterline,
  `PreparePlaintext`, Device constants and `reuse_input` are rejected in this
  bounded implementation. Default Host cache admission uses half currently
  available host memory after prepared setup, or the explicit byte budget.
- `begin_run()` clears the shared Linear and active plan's attention sequence
  state every run; existing output0 rebuilding remains intact. Active dispatch
  and Host caches switch only after Runtime returns and the API drains. All
  retained adapter persistent bytes and key-upload counters are summed,
  including separate AV bundles for identical manifests in different plans;
  shared CPU key material does not imply zero-cost GPU copies. Runtime failure
  cleanup drains before releasing
  Runtime values; the outer guard drains before retained owners are destroyed
  and snapshots actual counters, including partial engine evaluator counters.

## CPU validation

```sh
python3 bench/bert_dsl/multi_plan/tests/test_check_only.py \
  --runner test-results/bert-a800-20261115/multi-plan-build/attention-modes/build/bin/bert_multi_plan_runner \
  --guard test-results/bert-a800-20261115/multi-plan-build/attention-modes/build/tests/no_device_guard.so
python3 bench/bert_dsl/multi_plan/tests/test_attention_modes.py \
  --runner test-results/bert-a800-20261115/multi-plan-build/attention-modes/build/bin/bert_multi_plan_runner \
  --guard test-results/bert-a800-20261115/multi-plan-build/attention-modes/build/tests/no_device_guard.so \
  --output-dir test-results/bert-a800-20261115/multi-plan-build/attention-modes/tests-modes/recheck
```

The permanent suite creates minimal schema mocks without ignored golden assets.
It checks compatible two/three-plan libraries, data/context/parameter/weight and
Linear contract mismatches, namespace collisions, exact sequence operands,
shared Rescale settings, attention internal fields/AV headroom, complete model
semantics/layer-hash sets, rotation auditing, standard-verifier rejection,
waterline rejection, help/noargs, and zero device-call guard markers. These
mocks are structural tests, **not GPU-valid HE fixtures**.

The original **37 methods / 42 guarded mock invocations** are preserved and
rerun for this upgrade. New mode tests are in `tests/test_attention_modes.py`.
Current authoritative results/counts and source/binary provenance are in
`attention-modes/validation.json`; real v6/v7/v8 and three original integration
combinations are checked without device calls. Earlier 37-test evidence remains
read-only under `attention-modes/prior-37-tests/`; the initial 23-test evidence
also remains under the original `multi-plan-build/history/pre-model-config/`.
The on-entry archived root binary already differed from the SHA in historical
37-test validation; both are preserved without relabeling that binary as the
historically validated one (see `prior-37-archive-provenance.json`). New coherent
builds have independent source/binary/test binding. See current test logs,
`tests-37/no-device-gate.json`, mode-test guard evidence,
`real-fixture-gates.json`, and `*.check.json` in the new output directory. No `--run-gpu`, device execution, SSH, or GPU speedup test
was performed. GPU lifecycle, peak memory, actual counts and numerical outputs
remain unvalidated until an independently authorized GPU run.
