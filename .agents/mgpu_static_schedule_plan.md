# Multi-GPU Static Schedule Plan

## 1. Overall Plan

| Layer | Goal | Decision |
|---|---|---|
| V1 | Single-node 8-GPU ciphertext-level parallelism | Each ciphertext/plaintext is a full single-device object at every compute op. |
| Scheduling | Static | Dacapo emits an instruction sequence, then a placement/copy pass annotates devices and inserts explicit copies. |
| Execution | Interpreter | The interpreter executes the static schedule and validates invariants; it does not dynamically schedule. |
| Operators | Existing GPU operators | Every compute op calls the existing single-GPU `GpuEvaluator` on one target GPU. |
| Communication | CUDA P2P first | V1 uses a CUDA peer-copy backend; NCCL/MPI are planned for later cluster support. |
| Compatibility | Non-invasive | Existing single-GPU APIs and tests must keep working unchanged. |

## 2. Folder Plan

| Path | Purpose |
|---|---|
| `src/poseidon/mgpu/` | New multi-GPU compiler, IR, interpreter, runtime, and communication root. |
| `src/poseidon/mgpu/compiler/` | Dacapo adapter, placement pass, copy insertion pass, schedule verifier. |
| `src/poseidon/mgpu/runtime/` | Device contexts, object store, and schedule interpreter. |
| `src/poseidon/mgpu/comm/` | Communication abstraction and CUDA P2P implementation. |
| `src/poseidon/mgpu/ir/` | Internal schedule IR and readable debug dump/parser utilities. |
| `src/poseidon/mgpu/third_party/dacapo/` | Dacapo git submodule from `git@github.com:corelab-src/dacapo.git`. |
| `src/poseidon/tests/mgpu/` | Multi-GPU runtime tests; cross-device tests must skip on single-GPU machines. |

## 3. Modules

| Module | Responsibility |
|---|---|
| `GpuDeviceContext` | Own per-device `GpuParameterData`, `GpuEvaluator`, keys, and stream handle placeholder. |
| `GpuObjectStore` | Map logical value IDs to value kind, device ownership, and optional opaque object handles. |
| `IoBindingScheduleHandler` | Bind static upload/download value IDs to external object handles while forwarding compute ops to a fallback handler. |
| `PoseidonGpuScheduleHandler` | Optional CUDA/RMM-gated executor that uploads CPU Poseidon objects, calls existing single-GPU `GpuEvaluator`, and downloads scheduled results. |
| `GpuComm` | Define object-level copy/clone operations between devices and return destination object handles. |
| `MaterializedGpuComm` | Bridge logical copy requests to materialized full-object buffer copy requests. |
| `PoseidonGpuObjectCopyMaterializer` | Optional CUDA/RMM-gated bridge from Poseidon GPU objects to full-object copy buffers. |
| `CudaPeerComm` | Implement same-device copy, CUDA peer copy, and host-staging fallback as a materialized object-copy backend. |
| Communication topology planner | CPU-only classification of copy ops as same-device, intra-node CUDA peer, or inter-node transport for future cluster expansion. |
| Schedule IR | Represent upload, copy, compute, bootstrap fallback, and download operations independent of Dacapo format. |
| Dacapo adapter | Translate internal JSON debug input and Dacapo HEVM binary output into internal IR. |
| Placement pass | Assign each op/value to a GPU. |
| Copy insertion pass | Insert explicit copy operations when input placement differs from compute placement. |
| Verifier | Check device availability, input placement, object form, keys, scale, level, and NTT form before execution. |
| Interpreter | Execute the verified schedule in order. |
| Dumper | Emit MLIR-like readable text for debugging. |
| Dacapo HEVM dump tool | Load `.hevm + .cst`, run placement/copy insertion, and print schedule/I/O summaries for debugging real artifacts. |

## 4. Interface Plan

Use an internal schedule format first, and translate Dacapo output into it later.

```cpp
using ValueId = std::uint64_t;

enum class MgpuOpKind {
    UploadPlain,
    UploadCipher,
    CopyPlain,
    CopyCipher,
    Add,
    AddPlain,
    Sub,
    MultiplyPlain,
    Multiply,
    Relinearize,
    Rescale,
    Rotate,
    BootstrapFallback,
    Download
};

struct MgpuValueRef {
    ValueId id;
};

struct MgpuOp {
    MgpuOpKind kind;
    int device_id;
    std::vector<MgpuValueRef> inputs;
    std::vector<MgpuValueRef> outputs;
    std::string debug_name;
    std::unordered_map<std::string, std::int64_t> integer_attributes;
};

struct MgpuSchedule {
    std::vector<MgpuOp> ops;
};
```

Execution pipeline:

1. Dacapo adapter, internal JSON debug input, or handcrafted tests produce internal IR.
2. Placement pass assigns devices.
3. Copy insertion pass inserts `CopyPlain`/`CopyCipher`.
4. Verifier checks schedule invariants.
5. Dumper optionally prints a readable text form.
6. Runtime IO binding maps scheduled upload/download value IDs to external CPU/GPU object handles.
7. Interpreter executes the static schedule.

Static placement options:

- `default_device` is the fallback device for uploads, single-device compute,
  and downloads when no more specific option is set.
- `upload_device`, when set, places all unassigned upload ops on that device.
- `compute_devices`, when non-empty and `policy == RoundRobinCompute`, defines
  the exact round-robin compute device order. When empty, round-robin starts at
  `default_device` and wraps over `[0, device_count)`.
- `download_device`, when set, places unassigned downloads on that device.
- Placement assigns only devices. If upload, compute, or download devices differ,
  the copy insertion pass must insert explicit `CopyPlain`/`CopyCipher` ops.
  Runtime and GPU handlers must not infer or insert these copies.

Current Dacapo bridge:

- `DacapoInputFormat::Json` accepts Poseidon's internal schedule JSON for debug and tests.
- `DacapoInputFormat::HevmBinary` parses Dacapo `.hevm` output without linking MLIR/Dacapo.
- HEVM registers are translated to Poseidon SSA-like `ValueId`s; do not reuse HEVM register IDs directly as values.
- Static integer parameters such as `rotate_step`, `encode_level`, `encode_scale`, and bootstrap `target_level` live in `MgpuOp::integer_attributes`.
- Unsupported HEVM opcodes must fail clearly rather than being guessed.
- HEVM `NegateC` is supported as `MgpuOpKind::Negate`; HEVM `ModswitchC`
  and `UpscaleC` remain unsupported until Poseidon GPU has a verified
  dynamic modulus/scale-management mapping.

Optional GPU object materialization:

- `POSEIDON_BUILD_MGPU_GPU_OBJECTS=ON` builds the CUDA/RMM-gated Poseidon GPU object materializer.
- Default `poseidon_mgpu` must remain buildable without enabling this option.
- Materializers must reject `fields_.size() != 1` and per-poly multi-shard layouts; V1 copies only full single-field, single-shard objects.

Optional Poseidon GPU schedule execution:

- `POSEIDON_BUILD_MGPU_GPU_RUNTIME=ON` builds the CUDA/RMM-gated schedule handler that bridges mgpu IR to the existing single-GPU `GpuEvaluator`.
- The handler may execute only full single-device objects (`fields_.size() == 1`) and must reject copy ops; copy ops remain owned by the mgpu communication layer.
- It is a runtime executor, not a scheduler: device IDs and `integer_attributes` must already be present in the static schedule.

Runtime IO binding:

- `IoBindingScheduleHandler` is format-agnostic and stores opaque object handles by `ValueId`.
- Upload bindings are provided by the caller after placement/copy insertion has produced the final schedule.
- Downloads record the scheduled source value handle; typed CPU/GPU conversion remains the responsibility of a higher-level Poseidon GPU handler.
- Compute ops are forwarded to a fallback handler so the interpreter remains static and does not infer missing placements or payloads.

Dacapo artifact debugging:

- `POSEIDON_BUILD_MGPU_TOOLS=ON` builds `poseidon_mgpu_dacapo_hevm_dump`.
- The dump tool accepts `--hevm`, `--constants`, `--devices`,
  `--default-device`, `--upload-device`, `--compute-devices`,
  `--download-device`, `--round-robin-compute`, `--summary-json`, and
  `--no-schedule`.
- Use the dump tool before running a real ResNet20 artifact to confirm op
  counts, HEVM I/O metadata, explicit copy counts, and device distribution.
- The dump tool is CPU-side only; it must not link Dacapo/MLIR, CUDA runtime,
  RMM, or GPU evaluator code.
- `poseidon_mgpu_external_hevm_artifact_tests` is a skipped-by-default CTest
  for real `.hevm + .cst` artifacts. Set `POSEIDON_MGPU_EXTERNAL_HEVM` and
  `POSEIDON_MGPU_EXTERNAL_CST` together, plus optional placement variables
  `POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT`,
  `POSEIDON_MGPU_EXTERNAL_DEFAULT_DEVICE`,
  `POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE`,
  `POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES`,
  `POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE`,
  `POSEIDON_MGPU_EXTERNAL_ROUND_ROBIN_COMPUTE`, and
  `POSEIDON_MGPU_EXTERNAL_DEBUG_DUMP`.
- Use the CPU-only Poseidon GPU preflight before execution on real artifacts.
  It reports required communication, RelinKeys, GaloisKeys, invalid devices,
  and unsupported executor ops without linking CUDA/RMM. The dump tool exposes
  this through `--poseidon-gpu-preflight`, `--preflight-comm-available`,
  `--preflight-relin-keys`, and `--preflight-galois-keys`.
- Use `poseidon_mgpu_dacapo_hevm_dump --communication-plan` to classify real
  artifact copies as same-device, intra-node CUDA peer, or inter-node routes.
  For 4x8 previews, add `--nodes 4 --devices-per-node 8`. This remains a
  CPU-only diagnostic and must not pull NCCL/MPI into the normal build.
- Use `poseidon_mgpu_dacapo_hevm_dump --opcode-summary` on real artifacts to
  see the full HEVM opcode distribution before schedule translation. This is
  especially important for detecting unsupported `ModswitchC` and `UpscaleC`
  opcodes in ResNet20 artifacts.

ResNet20 artifact runbook:

- Use only the Nix-isolated Dacapo environment for Dacapo dependencies:
  `nix-shell src/poseidon/mgpu/nix/dacapo-shell.nix`.
- Dacapo's upstream helper functions in `config.sh` expect
  `$DACAPO_ROOT/build/bin/hecate-opt`; keep that build path when using
  `hc-trace`, `hopts`, or `hbt`, or invoke the optimizer directly.
- Generate the ResNet20 HEAAN GPU artifact with:
  `hc-trace ResNet` and `hbt dacapo 40 ResNet HEAAN GPU`.
- The expected artifacts are:
  `$DACAPO_ROOT/examples/traced/_hecate_ResNet.cst` and
  `$DACAPO_ROOT/examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm`.
  Verify both files exist before running Poseidon diagnostics.
- First Poseidon gate is CPU-side only:
  `poseidon_mgpu_dacapo_hevm_dump --opcode-summary --communication-plan
  --poseidon-gpu-preflight --summary-json --no-schedule` with explicit
  `--hevm`, `--constants`, and placement flags.
- Repeatable artifact validation uses the skipped-by-default
  `poseidon_mgpu_external_hevm_artifact_tests` with
  `POSEIDON_MGPU_EXTERNAL_HEVM`, `POSEIDON_MGPU_EXTERNAL_CST`, and explicit
  placement environment variables.
- Treat unsupported opcode diagnostics as adapter work, not scheduler work.
  Do not guess `ModswitchC` or `UpscaleC` semantics.
- Treat bootstrap fallback diagnostics as an execution-backend gap. Loading,
  placement, copy insertion, and communication planning may still be valid.
- For 4x8 previews, run the dump tool with
  `--communication-plan --nodes 4 --devices-per-node 8`; any `inter_node`
  routes are diagnostic until a cluster transport backend exists.

## 5. Test Plan

| Test | Requirement |
|---|---|
| Existing tests | Continue passing without `POSEIDON_BUILD_MGPU`. |
| IR tests | Construct schedules and verify kind/string/dump behavior. |
| Verifier tests | Missing input, missing copy, invalid device, unavailable key, and form mismatch must fail clearly. |
| Single-GPU interpreter tests | Upload, run one op, download, and compare against the existing single-GPU path. |
| IO binding tests | Missing upload bindings, kind mismatches, fallback compute output, and metadata-only downloads must fail clearly. |
| Copy tests | Same-device copy always runs; cross-device copy runs only when at least two GPUs are visible. |
| Materialized copy tests | Object-handle copy dispatch validates one-buffer full-object copy requests. |
| GPU runtime smoke tests | Optional CUDA/RMM tests cover upload/download, same-device Add, and a cross-device CopyCipher+Add schedule when at least two GPUs are visible. |
| Static graph tests | Handwritten ResNet-like small graph verifies copy insertion and execution order. |
| Placement configuration tests | Upload, compute-device list, and download placement must all trigger explicit copies when devices differ. |
| Artifact dump tool smoke | Build the optional tool and run it on mock `.hevm + .cst` artifacts with upload/compute/download device options. |
| External artifact test | Skips by default; when env vars point at real ResNet20 `.hevm + .cst`, load artifacts, run placement/copy insertion, build HEVM I/O plan, and print schedule summaries. |
| Poseidon GPU preflight tests | CPU-only checks report required comm/keys and unsupported GPU executor operations before attempting GPU execution. |
| Communication topology tests | CPU-only tests classify scheduled copy ops for single-node and uniform cluster topologies without linking NCCL/MPI. |

## 6. Phases

| Phase | Goal | Gate |
|---|---|---|
| 0 | Optional mgpu build skeleton and `.agents` planning files | Existing default build remains unchanged. |
| 1 | Internal schedule IR, dumper, basic verifier | Unit tests pass on CPU/single GPU. |
| 2 | Single-GPU interpreter and object store | Single-GPU schedule matches existing GPU result. |
| 3 | `GpuComm` and CUDA peer-copy backend | Same-device tests pass; multi-GPU tests skip or pass. |
| 4 | Static placement and copy insertion | Handwritten small graph verifies inserted copies. |
| 5 | Add Dacapo submodule and JSON/HEVM adapters | Adapter tests use small captured/mock Dacapo input. |
| 6 | ResNet20 static schedule path | Dacapo runbook generates real `.hevm + .cst`; dump tool reports opcode summary, schedule summary, communication plan, and CPU-only Poseidon GPU preflight; skipped-by-default external artifact CTest validates the same artifact; unsupported opcode and bootstrap diagnostics are resolved or explicitly recorded before GPU execution; single-node 8-GPU run validates on cluster hardware. |
| 7 | Cluster communication planning | CPU-only topology planning classifies inter-node copies first; NCCL/MPI interface is introduced only after single-node path is stable. |

On a single-GPU development machine, complete Phases 0-4 with same-device copy
and skip cross-device tests. Do not block implementation on multi-GPU hardware
until the first real cross-device validation phase.

## 7. Risks and Rules

| Risk | Rule |
|---|---|
| Accidentally changing single-GPU behavior | Do not modify existing public GPU APIs unless a test proves compatibility. |
| Confusing data-structure support with operator support | V1 full-object only; no multi-shard execution. |
| Hidden dynamic scheduling | Placement decisions belong to compiler/pass output, not interpreter runtime. |
| Async dependency bugs | Keep V1 execution conservative; introduce stream overlap only after correctness. |
| Dacapo dependency churn | Use Nix isolation; do not touch system package state. |
| Large key replication memory pressure | Replicate keys per GPU for V1, then profile before optimizing. |
| Bootstrap complexity | Treat bootstrap as a barrier/fallback op until native support is planned. |
| Premature cluster dependency lock-in | Do not add NCCL/MPI as required dependencies before the single-node path and topology interface are stable. |
