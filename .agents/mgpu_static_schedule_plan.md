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
| `GpuComm` | Define object-level copy/clone operations between devices and return destination object handles. |
| `MaterializedGpuComm` | Bridge logical copy requests to materialized full-object buffer copy requests. |
| `PoseidonGpuObjectCopyMaterializer` | Optional CUDA/RMM-gated bridge from Poseidon GPU objects to full-object copy buffers. |
| `CudaPeerComm` | Implement same-device copy, CUDA peer copy, and host-staging fallback. |
| Schedule IR | Represent upload, copy, compute, bootstrap fallback, and download operations independent of Dacapo format. |
| Dacapo adapter | Translate internal JSON debug input and Dacapo HEVM binary output into internal IR. |
| Placement pass | Assign each op/value to a GPU. |
| Copy insertion pass | Insert explicit copy operations when input placement differs from compute placement. |
| Verifier | Check device availability, input placement, object form, keys, scale, level, and NTT form before execution. |
| Interpreter | Execute the verified schedule in order. |
| Dumper | Emit MLIR-like readable text for debugging. |

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

Current Dacapo bridge:

- `DacapoInputFormat::Json` accepts Poseidon's internal schedule JSON for debug and tests.
- `DacapoInputFormat::HevmBinary` parses Dacapo `.hevm` output without linking MLIR/Dacapo.
- HEVM registers are translated to Poseidon SSA-like `ValueId`s; do not reuse HEVM register IDs directly as values.
- Static integer parameters such as `rotate_step`, `encode_level`, `encode_scale`, and bootstrap `target_level` live in `MgpuOp::integer_attributes`.
- Unsupported HEVM opcodes must fail clearly rather than being guessed.

Optional GPU object materialization:

- `POSEIDON_BUILD_MGPU_GPU_OBJECTS=ON` builds the CUDA/RMM-gated Poseidon GPU object materializer.
- Default `poseidon_mgpu` must remain buildable without enabling this option.
- Materializers must reject `fields_.size() != 1`; V1 does not execute multi-shard objects.

Runtime IO binding:

- `IoBindingScheduleHandler` is format-agnostic and stores opaque object handles by `ValueId`.
- Upload bindings are provided by the caller after placement/copy insertion has produced the final schedule.
- Downloads record the scheduled source value handle; typed CPU/GPU conversion remains the responsibility of a higher-level Poseidon GPU handler.
- Compute ops are forwarded to a fallback handler so the interpreter remains static and does not infer missing placements or payloads.

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
| Static graph tests | Handwritten ResNet-like small graph verifies copy insertion and execution order. |

## 6. Phases

| Phase | Goal | Gate |
|---|---|---|
| 0 | Optional mgpu build skeleton and `.agents` planning files | Existing default build remains unchanged. |
| 1 | Internal schedule IR, dumper, basic verifier | Unit tests pass on CPU/single GPU. |
| 2 | Single-GPU interpreter and object store | Single-GPU schedule matches existing GPU result. |
| 3 | `GpuComm` and CUDA peer-copy backend | Same-device tests pass; multi-GPU tests skip or pass. |
| 4 | Static placement and copy insertion | Handwritten small graph verifies inserted copies. |
| 5 | Add Dacapo submodule and JSON/HEVM adapters | Adapter tests use small captured/mock Dacapo input. |
| 6 | ResNet20 static schedule path | Single-GPU fallback works; multi-GPU run validates on cluster. |
| 7 | Cluster communication planning | NCCL/MPI interface is introduced only after single-node path is stable. |

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
