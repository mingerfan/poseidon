# Poseidon / dacapo 内存管理 IR 设计背景：现有系统调研

本文供后续设计者了解 Poseidon、其内置 dacapo 编译器、CKKS Runtime，以及 CPU/GPU 后端中与内存释放、存储复用、Host↔GPU 搬运和预取相关的现有实现。本文只记录系统现状、已有文档中的职责约定和实现边界，不提出新增 IR 或新系统方案。

## 1. 调研范围、版本和阅读约定

调研日期：2026-10-06。调研对象是本地源码快照，而不是原版 DaCapo 论文或其他分支。

| 仓库层级 | 本地位置，相对于 Poseidon 根目录 | HEAD |
| --- | --- | --- |
| Poseidon | `.` | `62dbe22559667199487edb8ecdcb06624f66e6d7` |
| CKKS Runtime submodule | `third_party/ckks-runtime` | `23c8b94202a14d3e322f87c50434253d56acb9f7` |
| dacapo 嵌套 submodule | `third_party/ckks-runtime/third_party/dacapo` | `edcebf5de955dd2a5cd1b87726b92355276e4390` |

调研开始时三个 checkout 均无本地改动。本文依据源码、测试源码、仓库设计文档和已提交 RuntimePlan 产物；没有重新编译或运行 GPU/MPI 测试，也没有测量实际内存峰值。

为避免重复很长的文件路径，后文代码引用使用三个前缀：

- `P/`：Poseidon 根目录。
- `R/`：`P/third_party/ckks-runtime/`。
- `D/`：`R/third_party/dacapo/`。

例如 `R/runtime/runtime.hpp:91` 表示 Poseidon 中的 `third_party/ckks-runtime/runtime/runtime.hpp` 第 91 行附近。行号对应上表快照。

本文区分三类依据：**代码已实现**、**已有文档描述但尚未实现**、**根据现有布局/产物做的静态计算**。负面结论，例如“没有 Prefetch 指令”，限定在本文所列的编译、协议和执行路径，不把第三方依赖中的同名通用功能算作 Poseidon 功能。

### 1.1 文档与代码存在时间差

`R/README.md` 开头仍称 Encode、bundle、Host compute 和 OperatorSpec 完整验证尚未实现，且把 Poseidon 后端列为后续工作。当前源码已经有这些功能及 CPU/GPU API。`R/docs/overview-design/implementation-status.md` 的部分 GPU/多卡状态也落后于当前实现。

类似地，已有架构文档中写“GPU 原生 Boot 暂不可用”属于早期背景；当前 `PoseidonGpuApi` 已有需要显式配置资源的 native Boot 分支。本文对执行行为以代码为依据，不把旧状态说明当作当前功能列表。

来源：`P/README.md`；`R/README.md`；`D/README.md`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1204`、`:1375`。

## 2. 系统结构与职责边界

### 2.1 当前执行链

```text
Python tracing / 已有 Earth MLIR
          │
          ▼
dacapo / hecate-opt，独立 MLIR 工程
  Earth：scale、level、Boot 插入等优化
          │ EarthToCKKS + UpscaleToMulcp
          ▼
  CKKS SSA：数学结果 + CKKS 元信息
          │ 可选 physical-level materialization
          │ placement
          │ communication materialization
          ▼
  带 dist 属性和 dist.transfer 的 CKKS / Dist MLIR
          │ EmitRuntimePlan
          ▼
RuntimePlan V1 JSON + OperatorSpec + 可选 plaintext bundle
          │
          ▼
RuntimePlanJsonReader → PlanVerifier → SequentialRuntime<Api>
          │                                  │
          │                                  ├─ Vec / Mock / MPI 明文测试后端
          │                                  ├─ PoseidonCpuApi
          │                                  └─ PoseidonGpuApi
          ▼
CPU Ciphertext / Plaintext       GPU 对象 / GpuEvaluator
CPU MemoryPool                  DeviceVector → RMM resource
                                CUDA local copy / MPI + NCCL
```

当前 fork 的可执行输出合同是 RuntimePlan。原 HEVM 字节码生成器、解释器和 Python runner 已移除。Poseidon 旧 `mgpu` 调度/解释路径也不是当前 RuntimePlan 执行主线。

Poseidon 是更广泛的 FHE 库，包含 CKKS/BFV/BGV、编码、加解密、密钥、参数上下文和 evaluator。本次编译器集成与计划协议针对 CKKS。原有 `PoseidonFactory` 有 software/hardware 分支；Runtime GPU API 直接建立 `GpuParameterData` 和 `GpuEvaluator`，不是把 GPU 映射成 factory 的 hardware evaluator。

来源：`D/README.md`；`D/tools/optimizer.cpp:304`、`:487`；`R/runtime/runtime.hpp:117`；`P/src/poseidon/factory/poseidon_factory.cpp:1`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1060`。

### 2.2 已有架构约定

现有架构文档把职责划分为：

| 层 | 已有职责约定 | 当前代码对应 |
| --- | --- | --- |
| 编译器 | 确定计算 Place、显式跨 Place 搬运、生成可执行计划 | placement、communication、export 三个 Pass |
| Runtime | 加载验证、按计划执行、管理值句柄与等待、绑定输入、返回输出 | `SequentialRuntime<Api>`、`ValueStore<Api>` |
| Api | 具体对象、计算、通信、底层格式转换、设备映射、同步和错误处理 | `PoseidonCpuApi` / `PoseidonGpuApi` |
| GPU evaluator / handler / kernel | FHE 运算、内部临时空间、kernel 提交 | `GpuEvaluator` 及各 handler |
| 内存资源 | CPU/GPU 物理块的分配与归还、池内复用 | CPU MemoryPool / GPU RMM |

架构文档还约定：一个 ValueId 只属于一个 Place；通信输出用新的 ValueId 表示目标位置副本；普通计算在数学层面不夹带通信；Runtime 不重新做 placement、不动态选择值的驻留位置、不补计划缺失的传输。API 可以选择保持数学值及端点不变的底层通信实现。

**当前 placement 没有显存容量模型或内存生命周期规划。** 架构文档所列的完整拓扑、内存容量与未来内存规划职责，不能等同于当前 scheduler 已经具备这些功能。

来源：`R/docs/overview-design/architecture.md:23`、`:217`、`:248`；`D/lib/Dialect/CKKS/Transforms/AssignPlacement.cpp:873`。

## 3. 与内存有关的现有术语和元信息

### 3.1 Place、rank 和设备编号

Runtime 使用：

```cpp
enum class PlaceKind { Host, Device };
struct Place {
    PlaceKind kind;
    int rank;
    int index;
};
```

- Host 的 JSON 是 `{"kind":"host","rank":0}`；C++ `index` 为 0。
- Device 的 JSON 是 `{"kind":"device","rank":0,"index":1}`。
- 分布式 GPU 后端中，`rank` 是 MPI 进程编号，不保证一个 rank 就是一台物理节点。
- `index` 是进程内逻辑 GPU 编号。API 构造时传入 `cuda_device_ids`，将其映射到当前进程看到的 CUDA device id；测试包含逻辑编号与物理编号反向映射。
- 编译器的 `dist.device = -1` 表示 Host，非负数表示 Device；export 时转换成上述 Runtime JSON。
- NCCL rank 由各 MPI rank 的 device count 前缀和加本地逻辑编号得到，不直接作为 Place。
- `rank_to_node` 是通信 profile / GPU 进程拓扑中的节点关系信息，不是 ValueDesc 的组成部分。

来源：`R/runtime/plan.hpp:15`；`R/runtime/json_plan_reader.cpp:16`；`D/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp:57`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1318`；`P/src/poseidon/runtime_api/communication/nccl_mpi_transport.h:13`。

### 3.2 CKKS 逻辑值、物理存储和数据表示

三层对象有所区别：

| 层 | 表示 | 已有内容 |
| --- | --- | --- |
| CKKS MLIR | `tensor<…x!ckks.poly<C * S * L>>` | components、scale_log2、level，及外部 tensor shape |
| Runtime | `ValueDesc` + ValueId | kind、Place、context、level、scale_log2、ntt、components |
| 后端 | Host/GPU 数据对象 | 实际内存块、`parms_id`、真实 scale、RNS 形状、layout、CUDA event 等 |

`ValueDesc` 本身没有内存地址、分配容量、storage id、alias 信息、buffer slot、访问权限、最后使用位置、stream/event id、tensor shape、P-limb 数量或预计字节数。Runtime 保存的是后端定义的值句柄，不展开它内部的 RNS buffer。

`context` 在 RuntimePlan 中是字符串标识；实际多项式次数 N、模数链及其参数由 OperatorSpec 和 API 的 `PoseidonContext` 提供。`parms_id` 是 Poseidon 后端标识具体参数层的 ID，不直接出现在 CKKS PolyType 中。

当前 GPU Runtime 验证要求普通 Device 值为完整、单设备、Q-only 对象：`p_count == 0`，一个 field，每个 component 恰好一个完整 shard，物理大小与逻辑形状一致。底层 key / Boot 工作区中的 QP 对象不属于这一普通值合同。

来源：`D/include/hecate/Dialect/CKKS/IR/CKKSOps.td:19`；`R/runtime/plan.hpp:91`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:2570`。

### 3.3 level 方向、分量数和内存大小变化

Earth 的 level 主要描述优化阶段的消耗层数；`PolyTypeConverter` 把 Earth level 转成 `base_level - earth_level`。CKKS / Runtime 的方向则是剩余参数层：普通 Q-only 数据的 `q_count = level + 1`。

GPU lazy rescale 的物理层展开另由 `MaterializePhysicalLevels` 完成。默认一个逻辑层映射为 4 个物理层，公式为：

```text
physical_level = OperatorSpec.upper_bound
                 - (logical_init_level - logical_level) × factor
```

它改写参数和结果类型、函数类型、ModSwitch downFactor 等，并验证 Rescale 丢弃模数的 bit 数与 scale 降幅匹配。该步骤在 placement 前执行。因此后续 placement 和 RuntimePlan 的 level 可以已经是物理 RNS 层数。

普通 CKKS 运算对存储形状的影响包括：

| 操作 | 当前元信息规则与存储关系 |
| --- | --- |
| AddCC、Negate、Rotate | 通常保留 level 和 components；Rotate 在 Runtime 合同中要求输入为 2 components |
| MulCC | 输出 components 为两输入分量数之和减 1；2×2 产生 3 components，scale 相加 |
| MulCP | components 不变，scale 相加 |
| Relinearize | Runtime 合同为 3 → 2 components，level/scale 不变 |
| Rescale | 减少 level / Q limbs，并改变 scale |
| ModSwitch | 减少 level / Q limbs，保持 scale |
| Boot | 根据 profile 改变 level、scale、components；内部有远多于一个输出密文的工作区 |
| Transfer / Replicate | 保持 kind 和 CKKS 元信息，改变 Place 并生成新 ValueId |

当前 GPU API 的 Runtime Rescale 分支，按目标 level 逐次调用单层 `evaluator->rescale`，生成中间密文，最后把 scale 设成计划声明值。底层 evaluator 另有 `rescale_x2`、`rescale_many`、`rescale_dynamic`，并不等于 Runtime 的这条分派路径已经使用它们。

来源：`D/lib/Conversion/CKKSCommon/PolyTypeConverter.cpp:44`；`D/lib/Dialect/CKKS/Transforms/MaterializePhysicalLevels.cpp:165`；`P/src/poseidon/crt_context.cpp:67`；`R/runtime/verifier.cpp:64`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1543`。

## 4. dacapo 已有 IR 和编译模块

### 4.1 Earth 与 CKKS 的现有语义

Earth 是 CKKS 参数优化层。它的类型区分 `!earth.ci<scale * level>` 与 `!earth.pl<scale * level>`；已有 scale、noise、latency、Boot placement 分析接口。`Earth_Op` 基类带 MLIR `Pure` trait。

当前 CKKS 是 result-style SSA：算子读取操作数并定义结果，没有 `dst` 操作数，也没有通过 `tensor.empty` 指定输出存储。`!ckks.poly<components * scale_log2 * level>` 中 1 component 表示 plaintext，至少 2 components 表示 ciphertext。

当前 CKKS TableGen 算子有：`encode`、`rotatec`、`negatec`、`relinearize`、`rescalec`、`modswitchc`、`upscalec`、`bootstrapc`、`addcc`、`addcp`、`mulcc`、`mulcp`。`upscalec` 是中间操作，导出前通过 `convert-upscale-to-mulcp` 消除。Runtime 同时定义 SubCC/SubCP，但此 CKKS TableGen 没有对应独立 sub 算子。

Earth→CKKS 中，密文×密文首先生成 3-component `ckks.mulcc`，再显式生成 `ckks.relinearize`。常量降成 `ckks.encode`，payload 保留为真实 MLIR attribute，不再只是旧 `.cst` 中的整数索引。

需要区分“设计文档中的纯数学语义”和 MLIR trait：当前 CKKS `CKKS_Op` 基类没有统一添加 `Pure` trait，也没有针对物理内存定义 Read/Write/Allocate/Free effect 或 BufferizableOpInterface。PolyType 的 `MemRefElementTypeInterface` 仅是元素类型接口，不表示已有 bufferization 或内存规划。

来源：`D/include/hecate/Dialect/Earth/IR/EarthOps.td:77`；`D/include/hecate/Dialect/CKKS/IR/CKKSOps.td:19`；`D/lib/Conversion/EarthToCKKS/EarthToCKKS.cpp:144`；`D/lib/Conversion/CKKSToCKKS/UpscaleToMulcp.cpp`。

### 4.2 与本次目标相关的 Pass 顺序

`hecate-opt` 的 dacapo 管线在 Earth 优化后执行 EarthToCKKS、UpscaleToMulcp、canonicalization，再通过 `addRuntimePlanExport` 追加目标相关步骤：

| Pass | 当前输入/作用/输出 |
| --- | --- |
| `materialize-ckks-physical-levels` | 可选；把逻辑 level 映射成目标物理 level，读取 lazy OperatorSpec |
| `assign-ckks-placement` | 读取固定 device counts、OperatorSpec V2 延迟及可选通信 profile；给算子分配 Place 和静态估计时间 |
| `materialize-ckks-communication` | 将跨 Place operand 替换为显式 `dist.transfer` 结果 |
| `emit-runtime-plan` | 输出 V1 JSON 和可选 bundle |

placement 与 communication 在提供 `runtime-plan-device-counts` 时追加。没有 placement 时，export 默认按 Host rank 0 输出。

这里没有 alloc/dealloc、Release、Evict、Prefetch、buffer assignment、跨 ValueId 的存储复用或通用 last-use 分析 Pass。

来源：`D/tools/optimizer.cpp:304`、`:487`；`D/include/hecate/Dialect/CKKS/Transforms/Passes.td:8`；`R/integrations/dacapo/generate_model_artifacts.py:187`。

### 4.3 AssignPlacement：已有计算图、时间和通信成本信息

`PlacementScheduler` 只接收单 block 函数，并要求线性、单结果、无 region 的 CKKS 操作。它使用确定性的 HEFT list scheduling：建立 predecessor/successor 图、计算优先级，在各 Place 已占用的时间区间中寻找依赖满足后的最早空档，再选择最早完成的候选 Place。

已有行为：

- `device-counts` 全零时用各 rank 的 Host 做 CPU candidates；GPU 拓扑用各逻辑 Device；零和正数混用不受支持。
- external arguments 固定来自 Host rank 0。
- Encode 固定来自 Host rank 0，不作为普通 compute 节点排程。
- decrypt_reencrypt Boot 使用 Host candidates；native Boot 属于 Device 计算路径。
- 用原始顺序赋 `dist.logical_id`，使计算结果 ID 不随 placement 重排改变。
- 给 compute op 写 `dist.rank`、`dist.device`、`dist.schedule_start`、`dist.schedule_finish`。
- 按估计 start/finish、Place 和原始索引排序，将计算 op 重排到 block 中。
- 对 `(Value, destination Place)` 记住已估计的 copy arrival time，同一目标上后续消费者复用该副本的成本信息。

延迟来自 OperatorSpec V2 的逐 level 表；Boot 使用选定 profile。通信成本可使用 rank 内/间两个固定整数，或 payload-aware profile。后者已有的 payload 公式为：

```text
components × (level + 1) × tensor_element_count
           × poly_degree × coefficient_bytes
```

通信 profile 包含启动延迟、最大速率、饱和尺寸和可选 payload/rate 点位插值；V2 还支持 `rank_to_node` 及 first-match 有序端点规则。

这些数据用于估计 placement 成本。scheduler 没有统计当前驻留字节数、GPU 容量、存储空闲区间、内存峰值、释放点或预取时间窗；`copies` 保存的是值副本的预计到达时间，不是运行时缓存和驱逐状态。估计时间属性也不变成 Runtime 的定时启动指令。

来源：`D/lib/Dialect/CKKS/Transforms/AssignPlacement.cpp:873`、`:908`、`:958`、`:1143`、`:1253`；`R/integrations/dacapo/communication-profile.md`。

### 4.4 Dist IR：目前只有一对一完整值 Transfer

当前 `DistOps.td` 只定义 `dist.transfer`。操作数和结果是相同类型的 ranked tensor，属性为：

```text
transfer_id
source_rank, source_device
destination_rank, destination_device
initialization
```

Transfer 结果还被附加 `dist.rank` / `dist.device`，表示目标 Place。验证器检查输入是 CKKS polynomial、ID/rank 非负、device 为 -1 或非负、源目标不同。

这一层没有 `dist.replicate`、传输完成 token、显式 await、拷贝 stream、部分区间、源是否失效、源释放、evict 或 prefetch 操作。`dist.transfer` 的 summary 明确称其为 copy。

来源：`D/include/hecate/Dialect/Dist/IR/DistOps.td:15`；`D/lib/Dialect/Dist/IR/DistDialect.cpp:23`。

### 4.5 MaterializeCommunication：副本复用和插入位置

该 Pass 要求已经有 `dist.device_counts`，且函数只有一个 block；如果已存在 `dist.transfer` 则报错，不能重复执行。

它逐一读取 compute operand 的来源和消费 Place，将需求按 `(原 SSA Value, 目标 Place)` 合并。一个目标 Place 上的多个消费者共同使用一个 Transfer 结果。不同目标使用不同 Transfer 结果。

插入位置是：arguments 的 Transfer 放在 block 开头；其他生产者的 Transfer 放在该 CKKS 生产者之后。新 Transfer 的 `initialization` 只有在源是 block argument 或 Encode 结果时为 true。计算结果的跨 Place Transfer 留在 execution。

因此已有行为是：**输入和常量提前搬运，计算结果在生产后搬运，同目标只物化一次**。没有按显存压力分批上传、在首次使用前选择距离、释放后重建同目标副本、或用最后一次消费来结束副本驻留的逻辑。函数 return 的 operand 不是这个 Pass 的跨 Place compute operand 扫描对象，最终输出不自动搬回 Host。

来源：`D/lib/Dialect/CKKS/Transforms/MaterializeCommunication.cpp:45`、`:77`、`:106`、`:135`、`:182`。

### 4.6 EmitRuntimePlan：现有导出行为

导出要求单 block、无嵌套控制流，除 return 外每个 op 必须单结果且属于 ckks/dist。arguments 先取 ID；placement 后 CKKS 结果沿用 `dist.logical_id`；Transfer 结果从现有最大 ID 之后分配。

Encode 全部进入 initialization；Transfer 依据 `initialization` 属性进入 initialization 或 execution；compute 全部进入 execution；当前 exporter 写空 finalization。全局 ordinal 按 initialization、execution 顺序重新编号，ValueId 不等于执行序号。

Transfer 导出成 `hint=point_to_point` 的单目标动作。当前 exporter 不把 fanout 合成为 Runtime `Replicate`。return operands 原样成为 `final_outputs`，保留其实际 Place。

`dist.schedule_start/finish` 不进入 RuntimePlan；物理 byte size、lifetime 和 buffer alias 也不进入。Export 的 ValueDesc 只提取 PolyType 元信息，不记录 tensor shape；而 placement 的通信大小估算会乘 tensor element count。Runtime API 的每个 Value 当前表示一个后端明文/密文对象，没有通用 tensor-of-ciphertexts 容器。

来源：`D/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp:211`、`:329`、`:388`、`:617`。

## 5. RuntimePlan 和 OperatorSpec 的当前合同

### 5.1 RuntimePlan V1 结构

```text
format_version, plan_id
target: target_id, capability_version, world_size, device_counts,
        operator_spec {id, version, source_sha256}
可选 plaintext_bundle {id, version, manifest_sha256}
values: ValueDesc[]
external_inputs: ValueId[]
initialization: Instruction[]
execution: Instruction[]
finalization: Instruction[]
final_outputs: ValueId[]
```

ValueId、TransferId 在 C++ 中为 uint64，JSON 中为规范十进制字符串。普通 integer 元信息按 reader 的范围检查读取。三阶段的 ordinal 要连续、稳定。

InstructionBody 当前恰好三种 C++ variant：`EncodeOp`、`ComputeOp`、`CommAction`。JSON 接受 `encode`、`compute`、`transfer`、`replicate` 四类 kind。

| 动作 | 当前合同 |
| --- | --- |
| Encode | payload + 一个输出 ID；只允许 initialization；输出必须是 Host plaintext |
| Compute | kind、inputs、一个 output、一个 Place、操作专用 attrs |
| Transfer | 一个源值、一个源 Place、一个输出、一个不同目标 Place |
| Replicate | 一个源值、多个输出、多个互异且不等于源的目标 Place |

ComputeKind 为 AddCC/AddCP/SubCC/SubCP/MulCC/MulCP/Negate/Rotate/Rescale/ModSwitch/Relinearize/Boot。ComputeAttrs 只包含 rotate steps、rescale target level/scale、modswitch target level、Boot target metadata/profile/implementation；没有输出 buffer 或可破坏输入的标志。

CommHint 当前为 Auto、PointToPoint、Broadcast、Tree、Ring、HostStaged。hint 是实现提示，不是源失效或生命周期操作。

来源：`R/runtime/plan.hpp:11`、`:24`、`:56`、`:91`、`:137`；`R/runtime/json_plan_reader.cpp:200`、`:235`。

### 5.2 验证行为与内存功能的当前边界

PlanVerifier 校验：ValueDesc 唯一与基本元信息、external input 必须在 Host、使用前定义、一次定义、Place 合法、算子输入均在 compute Place、算子元信息转换合法、OperatorSpec 支持、TransferId 唯一、输出/目标/类型对应、通信不改变 CKKS 元信息、最终输出已定义且不重复。

它同时推导能力及密钥需求：Encode/Transfer/Replicate/HostCompute/BootNative/BootDecryptReencrypt；Rotate 产生具体 Place、rotation step、input level 的 Galois requirement；Relinearize 产生 Place/input level 的 Relin requirement。

验证器的 `defined` 集合只增长，没有“已释放”“源被消耗”“底层 buffer 被另一 ValueId 覆盖”的状态。ValueDesc 是否“unused”的检查是描述符是否在定义/操作/输出等处出现，不是运行时 liveness 检查。

JSON reader 使用 `require_members` 拒绝未知字段，并拒绝未知 instruction kind。当前 V1 没有通用扩展字段供内存指令自动透传。

来源：`R/runtime/verifier.cpp:178`、`:217`、`:280`、`:324`；`R/runtime/json_utils.hpp:36`；`R/runtime/json_plan_reader.cpp:235`。

### 5.3 OperatorSpec 的已有内容

OperatorSpec 记录 target、context id、N、RNS modulus bit widths、default scale、level bounds、rescale mode、各算子 support 和逐 level latency/noise，以及 Boot profiles、来源信息和版本。

Boot profile 记录 native/decrypt_reencrypt、input level 范围、输入/输出 components、output level/scale、latency/noise 和是否需要 secret key / Host compute。

当前 `OperatorSpec` C++ 数据结构不记录每 GPU 显存容量、Host 内存容量、算子输出可否 alias 输入、具体 scratch 字节数、数据生存期、常量缓存配额或预取策略。通信性能单独由 dacapo communication profile 输入。

来源：`R/runtime/operator_spec.hpp:13`、`:30`、`:45`；`R/runtime/operator_spec_reader.cpp`。

## 6. Runtime 的值管理、执行和持有关系

### 6.1 ValueStore 的实际功能

当前 store 是 `unordered_map<ValueId, Entry>`，Entry 为：

```cpp
Ready   { Place place; Api::Value value; }
Pending { Place place; size_t group; size_t local_slot; }
```

只有 `define_ready`、`define_pending`、`lookup`、`entries`，没有 erase/release/replace-storage 接口。重复定义相同 ValueId 报错。Pending 表示通信还未产出本地值句柄，不是 Host/Device 驻留缓存状态。

顺序执行 compute 时，Runtime 把每个输入句柄复制进 `vector<Value>`，调用 `api.compute(op, inputs)`，验证输出并定义新 ID。通信时同样传递本地输入句柄，保存 CommHandle、输出 ID 和 posted 状态。

**不存在最后使用后自动删除输入值。** 顺序模式中，中间值留在 store；通信 group 也留在 `groups_`。它们在下一次 `run()` 的开始被重置，或在 Runtime 析构时销毁。`run()` 返回前不会统一清空 store。

来源：`R/runtime/runtime.hpp:91`、`:132`、`:380`、`:404`。

### 6.2 run 的阶段、同步与返回

当前执行顺序：

1. 重置前一轮 Runtime store、groups、parallel 状态、bundle slots 与计时。
2. PlanVerifier、运行拓扑验证、bundle 加载、API preflight。
3. 绑定本 rank 外部输入，复制句柄并验证。
4. 执行 initialization；`finish_all_groups()` 等初始化通信结束。
5. 执行 execution 和 finalization，按选定 sequential / workers 模式执行。
6. 收尾所有通信，对本 rank final outputs 调用 API synchronize。
7. 构建 RunArtifact 并返回。

`setup` 包含验证、bundle 加载、preflight 与 input binding；`initialization` 包含 Encode 和初始化搬运，并有通信完成边界；`online_execution` 包含 execution、finalization、收尾通信和最终输出同步。

默认 `DiffMode::FinalOnly` 只把最终值放进返回 artifact；它不表示 Runtime 已释放所有中间值。`AllValuesAfterRun` 把所有 Ready 值复制进 artifact，用于差分；artifact 持有的 shared_ptr 也会延长底层对象寿命。

Source/final output 可以继续在 Device。`synchronize_final_outputs()` 只等完成，不执行计划外的 D2H。

来源：`R/runtime/runtime.hpp:132`、`:970`、`:991`；`R/experiments/dacapo_plan_vec_diff.cpp`。

### 6.3 Ready 不等于 GPU 工作已完成

Runtime 通过模板检测可选 `Api::posted_outputs(CommHandle&)`：它返回与 action outputs 一一对应的 optional Value。某个 Device output 可在 CUDA 提交完成后立即作为 Ready 值发布，内部仍携带 event/request。

如果 API 尚未提供本地输出句柄，Runtime 才保存 Pending；首次消费该值时 `finish_group` 调用 `api.wait` 并将其转为 Ready。对已 posted 的 Device 值，消费者可直接把句柄交给 compute，由 GPU API 建立 stream 依赖。

即使输出已 posted，最终仍要 drain 通信 group，包括只有本地发送、没有本地接收输出的 group。异步源 buffer 还需要活到发送/拷贝结束，这一物理条件目前由 API 的保留引用及 request 生命周期承担。

来源：`R/runtime/runtime.hpp:36`、`:289`、`:366`、`:404`、`:438`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:2207`。

### 6.4 PerDeviceWorkers 的实际执行模型

虽名为 `SequentialRuntime`，类中已有 `DeviceExecutionMode::PerDeviceWorkers`。initialization 仍先顺序执行并 drain；随后为 execution/finalization 建立各 worker 的 FIFO task list。

- compute 根据 `Place.index % worker_count` 分配，默认每个本地 device 一条 worker。每个 worker 按队列顺序提交。
- 对所有本地计算/通信结果预建 `ParallelValue`，用 mutex/condition_variable 等待句柄发布。
- 本 rank D2D 在源 worker 提交；H2D 在目标 worker；D2H 在源 worker。
- `PoseidonGpuApi::background_communication_issuer=true`，跨 rank 动作由每个 MPI rank 一条专用 issuer 线程按计划顺序提交。
- 同时有本地和远端目的的 Replicate 分成两条执行路径，保留原 transfer id。
- task 提交结束后 join，再 drain 尚未完成的通信，最终值被放回普通 store。

当前限制：该模式要求本地至少一个 Device，在线 compute 必须为 Device；不支持在线 Encode 和 `AllValuesAfterRun`；跨 rank 在线通信要求 Device-only endpoints。因此 execution 中的 Host Boot 不属于这个模式支持的 compute，跨 rank Host-source 上传可在 initialization 使用。

`parallel_values_` 与 `parallel_groups_` 也不会按最后一次使用删除，在后续 run/reset 或 Runtime 析构时销毁。并行模式复制了句柄，并没有把值所有权转移成一次消费。

来源：`R/runtime/runtime.hpp:490`、`:595`、`:766`、`:831`、`:894`；`P/src/poseidon/runtime_api/communication/README.md`。

## 7. Host 内存、CPU 数据对象和常量

### 7.1 CPU 的物理数据与内存池

Poseidon `Ciphertext` 和 `Plaintext` 通过 `DynArray` 持有连续 CPU 数据。CPU residues 是 64-bit，RNSPoly 描述各多项式/RNS limb 的数据视图。

CPU 已有 `MemoryPoolHandle`、`MemoryManager`、`MemoryPoolST/MT` 和 pooled `Pointer`。MemoryPoolHandle 使用 shared_ptr 保持内存池存活；全局池通常维持进程寿命，也可使用自定义/线程本地池。

已有低层操作及区别：

| 操作 | 当前行为 |
| --- | --- |
| `Ciphertext::release` / `Plaintext::release` | 重置对象元信息，将所持数据归还池 |
| `DynArray::release` | size/capacity 清零，释放 Pointer 的分配 |
| `DynArray::clear` | 只清 size，不改变 capacity |
| `DynArray::resize` | 新 size 不超过 capacity 时保留分配；超过时申请新块并复制 |
| `DynArray::reserve` | 申请指定容量并复制保留部分，旧块归还池 |
| `DynArray::shrink_to_fit` | 通过 reserve(size) 得到与 size 一致的容量 |
| pooled `Pointer::release` | 调用 pool head 的 add 把块归还，不代表操作系统 RSS 立即下降 |

直接复制一个 `Ciphertext` 会复制数据；复制 Runtime `PoseidonCpuValue` / `PoseidonGpuValue` 则主要复制共享句柄。两者不能混为同一种“copy”。

来源：`P/src/poseidon/basics/memorymanager.h:24`；`P/src/poseidon/basics/util/pointer.h:73`；`P/src/poseidon/basics/dynarray.h:352`；`P/src/poseidon/ciphertext.h:277`；`P/src/poseidon/ciphertext.cpp:13`；`P/src/poseidon/plaintext.h:221`。

### 7.2 CPU Runtime API

`PoseidonCpuValue` 内部是 `shared_ptr<Plaintext>` / `shared_ptr<Ciphertext>` variant，公开计算输入访问为 const。`PoseidonCpuApi::compute` 每次建立 `Ciphertext output` 再返回新的 Value；Negate、Rotate、Rescale 等分支先复制输入到 output，再处理这个输出，不消耗 Runtime 中的输入句柄。

CPU evaluator 有一些明确的 inplace 接口，例如 `square_inplace`、`multiply_plain_inplace`，内部还有 add/multiply/rescale 的 inplace helper。但当前 Runtime ComputeOp 没有 buffer reuse/inplace 参数，CPU API 没有“最后一次使用时接管输入”的协议。

CPU API 的计算是 Host 路径，`synchronize(Value&)` 为空。开启 CPU MPI 时通信把值 serialize 成 byte buffer，再用 MPI 请求发送/接收并 deserialize，不通过 GPU uint32 payload 路径。

来源：`P/src/poseidon/runtime_api/poseidon_cpu_api.h:33`；`P/src/poseidon/runtime_api/poseidon_cpu_api.cpp:296`、`:498`、`:609`、`:834`；`P/src/poseidon/evaluator/evaluator_ckks_base.h:98`、`:267`。

### 7.3 Encode 和 bundle 的 Host 生命周期

Bundle 保存的是编码前的 float64 slots，不是 CPU 已编码的 RNS plaintext，更不是 GPU plaintext。大常量由 exporter 默认以 4096 bytes 为阈值外化，blob 是 little-endian float64 数据，按内容 SHA-256 去重。

Runtime 初始化前扫描本 rank 的 bundle Encode，加载需要的内容到 `map<string, vector<double>> bundle_slots_`，校验 manifest/content 摘要、byte length、有限值和 slot capacity。同一 content 的 slots 在这个 map 中复用。

执行 Encode 时还会将 slots 复制进局部 `vector<double>`，随后在 Host 使用 `CKKSEncoder` 按 output level/scale 编码出 `Plaintext`。slots 缓存在整个 run 中保留，到下一次 run 的开始 clear。每个 Encode 输出是独立 ValueId；blob 内容相同不表示不同 level/scale 的编码后 plaintext 共用同一存储。

同一段内容可能同时存在：bundle 文件、Host float64 slots、编码后的 Host uint64 RNS plaintext、上传用的 pinned uint32 staging、Device uint32 plaintext。各表示由不同对象/容器管理。

来源：`D/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp:81`、`:160`；`R/runtime/plaintext_bundle.cpp:48`、`:66`；`R/runtime/runtime.hpp:305`、`:350`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1357`。

## 8. GPU 对象、RMM 分配和实际释放

### 8.1 GPU 存储层次

```text
PoseidonGpuValue
  ├─ shared_ptr<Host Plaintext / Ciphertext>
  └─ shared_ptr<GpuPlaintextData / GpuCiphertextData>
       ├─ meta：parms_id、scale、NTT、degree、q_count、p_count、components
       ├─ fields_：vector<GpuFieldData>，实际拥有显存
       │    └─ DeviceVector<GpuWord>，RMM 分配与释放
       └─ poly / polys_：GpuRNSPoly + GpuPolyShard，描述逻辑位置

Gpu*View / Gpu*ShardView：临时指针和形状视图，不拥有分配
PoseidonGpuValue.ready_：另一个 shared_ptr，保留完成 event / request
```

`GpuWord = uint32_t`，`GpuWide = uint64_t` 主要用于中间计算。GPU API 构造时验证 Q/P modulus 能放进 GpuWord。

`GpuFieldData` 只知道 device、buffer pointer 和 element count，不理解 c0/c1、level、RNS 区间。`GpuPolyShard` 使用 field_index/field_offset、limb_begin/count 和 coeff_begin/count 把逻辑分片映射到实际分配。

默认单设备密文一个 field，各 component 连续排列，每个 component 一个 full shard；shard 内 limb-major：`local_limb × coeff_count + local_coeff`。Plaintext 是一个 polynomial。数据结构可以描述更多分片，但普通 Runtime value 验证/搬运只接受完整单设备布局。

来源：`P/src/poseidon/gpu/gpu_memory.h:27`、`:283`；`P/src/poseidon/gpu/gpu_rns_poly.h`；`P/src/poseidon/gpu/gpu_ciphertext.h:46`；`P/src/poseidon/gpu/gpu_ciphertext.cpp:165`；`P/src/poseidon/gpu/gpu_plaintext.h:48`。

### 8.2 DeviceVector 的分配合同

DeviceVector 不可复制、可移动，持有 pointer、size、bytes、device id、实际使用的 RMM resource pointer 和可选 resource owner。

- `allocate(size, device)` 先 release 原分配，检查大小溢出，切换 device，从 current device resource 申请新块。
- 分配和 deallocate 都使用 `gpu_execution_stream()`，即 `cudaStreamPerThread`。
- `release()` 调用保存的 resource 的 deallocate，再清空成员；析构调用 release。
- 没有独立 capacity/reserve 接口；这一层的 allocate 本身不会因原块够大而保留它，容量复用由更高层 ensure-capacity helper 判断。
- `copy_from_host` / `copy_to_host` 虽用 cudaMemcpyAsync，但随后立即 `cudaStreamSynchronize`，从调用方看是同步完成。
- `fill_zero` 使用 cudaMemsetAsync，不在该函数内同步。

DeviceVector 的 resource owner 保留分配使用的内部 pool 的寿命；registry 是 resource→weak_ptr owner，并有线程本地 weak cache。外部自行安装的 resource 不会自动获得所有权，除非相应 owner 已注册。

来源：`P/src/poseidon/gpu/gpu_memory.h:80`、`:116`、`:143`、`:203`；`P/src/poseidon/gpu/gpu_memory.cpp:10`。

### 8.3 GPU Runtime 内部内存池

`acquire_device_memory_pool` 按 CUDA device 共用 weak registry。如果当前 resource 是普通 `cuda_memory_resource`，API 创建 `pool_memory_resource<cuda_memory_resource>` 并设为该 device 当前 resource；若已有其他类型 resource，则沿用它，不再自行安装另一个池。

内部池初始尺寸：默认 `min(24 GiB, 当前 free_bytes / 4)`。free memory 小于 64 MiB 时拒绝初始化。可用 `POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB` 指定初始尺寸，需为正整数且不超过当前 free memory 的约 90%。**24 GiB 是默认初始尺寸的上限，不是运行总量硬上限**；此构造调用没有指定 maximum_pool_size。

池是 best-fit/coalescing、支持 stream 顺序和多线程的 suballocator。对象 deallocate 先归还 pool；pool 析构才把所持上游块归还。因此“值已经没引用”“buffer 已归还池”“CUDA 进程实际保留的显存下降”是三个不同状态。

API DeviceState 持有 pool；DeviceVector 也持有 owner。API 析构先对各 device 做 cudaDeviceSynchronize。存在外部输出对象时，内部 pool 可以比 API 活得更久；最后 owner 消失后 pool 析构并恢复旧 current resource。现有测试明确覆盖这一寿命关系。

来源：`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:80`、`:183`、`:228`、`:1204`；`P/third_party/rmm/include/rmm/mr/device/pool_memory_resource.hpp:95`、`:197`、`:420`；`P/src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp:201`。

### 8.4 GPU API 的额外保留引用

`PoseidonGpuValue::Storage` 是四种 shared_ptr 的 variant。Value 的复制共享原始对象；`device_ciphertext()` 也有 mutable overload，没有 copy-on-write、唯一所有者检查或统一消费接口。

`PoseidonGpuApi::retain_in_flight(values, resources)` 将每个 Value 的 storage shared_ptr 和显式临时资源存进 `in_flight_resources_`，由 mutex 保护。普通 Device compute 在提交后会保留 inputs、输出，以及复合 Rotate / 多次 Rescale 的中间对象。通信提交保留本地源，通信 wait 后保留输出。

**当前 `in_flight_resources_` 只追加，没有按 event 完成清理，也没有 run 边界清理。** `synchronize(value)` 只等完成，不清空该集合。因此这些资源至少保留到 API 析构；仅删除 Runtime 的 ValueId 或销毁局部输入 vector，并不能使这些 GPU 分配归还池。

多个引用可指向同一个 storage，增加 shared_ptr 引用数不代表复制了多份 payload。但新 SSA 结果、Rotate/Rescale 中间结果仍各自可能拥有独立分配。

通用 GPU MPI runner 在 warmups/iterations 中复用同一个 API 和 Runtime；Runtime 会重置其每轮 store，API 的保留集合仍跨轮存活。这是源码可确认的持有行为，不是实测内存增长数字。

来源：`P/src/poseidon/runtime_api/poseidon_gpu_api.h:68`、`:93`、`:260`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:951`、`:1450`、`:1605`、`:2351`、`:2720`；`P/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_runtime_e2e.cpp:421`、`:481`；测试 `poseidon_gpu_api_test.cpp:1057` 明确检查 synchronize 不释放输入存储。

## 9. 现有输出存储复用和局部原位处理

### 9.1 Runtime 计算接口没有存储选择

当前核心接口为：

```cpp
Value compute(const ComputeOp &op, const std::vector<Value> &inputs);
```

GPU API 每次普通 compute 先声明空 `GpuCiphertextData output`，调用 evaluator 后包装为新 Value 并记录完成事件。Runtime 没有传入一个可复用的 destination，也没有声明输入可破坏、结果和输入共享 allocation、或某个旧 ValueId 从此无效。

这与 evaluator 的 `operation(const source&, destination&)` 接口不同。后者能操作预分配对象，但当前 Runtime 调用路径的 destination 是新建空对象。

来源：`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1477`、`:1605`；`P/src/poseidon/gpu/gpu_evaluator.h:352`。

### 9.2 evaluator 的预分配输出与同对象输入/输出

`prepare_ciphertext_destination` 检查 destination 的 device、degree、Q/P shape、components 和逻辑 shard layout。非输入 alias 且匹配时可以保留原 destination 的 allocation；否则重新 allocate。

Add/Sub/Negate/AddPlain/SubPlain 等经过这个 helper。helper 遇到 destination 是某输入同一对象时会重新分配，不能据此称这些接口已有完整的“覆盖输入且复用原 allocation”合同。

Multiply、Square、Rescale、Relinearize 等分支显式识别 source/destination 为同一 C++ 对象的情况：先用单独 local_result 分配、计算，再 move 到 destination。**同对象调用可有单独结果空间，不等于原物理 buffer 原位复用。**

`multiply_plain`、一般 `drop_modulus`、NTT 等路径还有直接分配新 result 再 move 的实现。可预分配复用的算子和始终 materialize 新结果的算子并不一致；当前也没有把各算子的这种信息统一暴露给编译器的 capability 描述。

来源：`P/src/poseidon/gpu/gpu_evaluator.cpp:556`、`:587`、`:968`、`:1350`、`:1644`、`:1860`、`:2211`、`:2600`。

### 9.3 已有的明确局部原位操作

底层 `GpuElementwiseHandler` 已有 `add_plain_to_ciphertext_inplace` / `sub_plain_from_ciphertext_inplace`，直接修改 c0，保留 c1/c2。`GpuEvaluator::multiply_plain_accumulate` 把 product 累加到已有 destination，已有 shape/scale 兼容性检查和专用 kernel。

高精度 EvalMod 内部也有与“最后一次使用”相关的局部逻辑：

- `node_use_counts` 统计 polynomial combine 对 quotient/remainder 的引用数。
- `basis_last_combine_use` 记录某个 basis 在 combine steps 中的最后使用位置。
- 在零拷贝 ModDrop 路径允许且内部节点单使用、或 basis 到达最后 combine use 时，可以使用 mutable 对象缩短 Q-prefix。
- `drop_modulus_inplace` 只改 parms_id、q_count 和 shard limb_count，保留原 allocation 和原 component offsets。物理 stride/padding 可以大于新的逻辑 Q-prefix，并未把尾部显存释放。
- 条件由 `POSEIDON_EVALMOD_D2D_FREE_DATAFLOW`、`POSEIDON_EVALMOD_ZERO_COPY_MODDROP`、`POSEIDON_EVALMOD_Q_PREFIX_VIEWS` 等开关及 trace capture 状态控制。

这些是单次 Boot / EvalMod 内部 schedule 的实现细节，节点不是 RuntimePlan ValueId。它们不是 dacapo 的通用 liveness pass，也不生成 Runtime memory 指令。普通 Runtime Device value 的紧凑布局校验与这里允许保留 padding 的内部 view 合同不同。

来源：`P/src/poseidon/gpu/gpu_elementwise_handler.cpp:519`、`:556`；`P/src/poseidon/gpu/gpu_evaluator.cpp:1443`、`:4584`、`:6284`、`:6464`、`:6597`。

## 10. Host↔GPU、GPU↔GPU 的现有传输

### 10.1 两种已有 CPU/GPU 转换入口

| 入口 | 当前用途和完成行为 |
| --- | --- |
| `GpuUploader` | CPU/GPU 对象、密钥、矩阵、EvalMod 常量的 setup 转换；很多 field copy 使用 DeviceVector 的同步 Host copy |
| `PoseidonGpuApi::communicate_async` | 执行 RuntimePlan Transfer/Replicate，包含对象验证、分配、pinned staging、异步 payload copy 和完成句柄 |

两条路径都进行 CPU uint64 residues ↔ GPU uint32 residues 转换，但 `GpuUploader::upload_*` 本身不解释 ValueId 和计划，不是 Runtime 的显式通信指令。

来源：`P/src/poseidon/gpu/gpu_uploader.h:45`；`P/src/poseidon/gpu/gpu_uploader.cpp`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1612`。

### 10.2 本 rank Host → Device

当前执行过程：

1. 检查 source Host value、目标 ValueDesc、Place 和 kind。
2. 从 Host 对象读 degree、Q/P shape、components 与 metadata。
3. allocate 新目标 GPU 对象。
4. 建立 `PinnedHostBuffer`，用 CPU loop 检查并转换 uint64 residues 为 uint32。
5. 在目标 device 的 non-blocking copy stream 上提交 H2D `cudaMemcpyAsync`。
6. 记录 completion event，把输出 Device Value 作为可 posted 句柄返回。
7. 后续 compute 在自己的 execution stream 上等待该 event。

`PinnedHostBuffer` 通过 cudaMallocHost 分配、cudaFreeHost 析构；当前没有 staging-buffer pool/cache。这次传输的 CPU 转换 loop 和 pinned allocation 在提交 API 调用内完成，payload 的 GPU copy 异步完成。

传输源不失效，也不从 Runtime store 中消失；目标拥有新 allocation。H2D request 持有 staging，输出 ready event 持有 request，Runtime/API 还各自持有输出引用。

来源：`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:439`、`:484`、`:1741`；`P/src/poseidon/runtime_api/communication/cuda_local_transfer.cpp:103`、`:566`。

### 10.3 本 rank Device → Host

GPU API 建立 pinned uint32 staging，提交 D2H，并在 CommHandle 中保存 metadata/staging 的 deferred output。`posted_outputs` 此时没有 Host value。

`wait(handle)` 等 request 完成，再将 uint32 staging 转回 Host uint64 Plaintext/Ciphertext、恢复 metadata，然后产出 Host Value。Host materialization 需要 CPU 实际访问数据，因此这一段是 blocking completion 路径。

请求 wait 后，CommHandle 清理 request/deferred/staging 状态；若 request 被 Device output 的 ReadyEvent 等其他共享对象继续持有，其 lifetime 仍由剩余引用决定。`ReadyEvent::wait` 不重置内部 request，request 持有的 staging 也可继续存在到最后引用释放。

D2H 是 copy，不等同于迁移后释放 GPU 源。

来源：`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:1779`、`:2262`、`:2340`；`P/src/poseidon/runtime_api/communication/cuda_local_transfer.cpp:595`。

### 10.4 本 rank Device → Device

`prepare_full_object_copy` 验证完整对象并为 destination allocate 同形状存储，得到原始 `DeviceBufferCopy`：source ptr、destination ptr、bytes、source/destination CUDA device。

`CudaLocalTransfer` 已有路由：

| 路由 | 当前行为 |
| --- | --- |
| SameDevice | 同设备 D2D memcpy；底层可调用，但 RuntimePlan 不允许同一个 Place 到自身 |
| PeerToPeer | cudaMemcpyPeerAsync，需要 directed peer access |
| HostStaged | 源 D2H → pinned Host → 目标 H2D，用 events 串接两设备 copy stream |
| Auto | 同设备走 SameDevice；能访问 peer 走 P2P，否则 HostStaged |

API 构造时为可访问的设备对 enable peer access。HostStaged hint 明确选择该底层路径，其他现有 hint 在本地 backend 映射成 Auto；当前没有独立 tree/ring/broadcast 算法分派。本地 Replicate 对各目的逐个建立 copy。

底层路由改变不引入额外 RuntimePlan ValueId；HostStaged 的 pinned buffer 是实现内临时空间，不是 Host ValueDesc。

来源：`P/src/poseidon/runtime_api/communication/gpu_object_copy.cpp:30`、`:142`；`P/src/poseidon/runtime_api/communication/cuda_local_transfer.cpp:382`、`:472`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:374`、`:1822`。

### 10.5 传输流、allocation dependency 和 request 生命周期

计算和 DeviceVector allocation 使用 thread-local default stream；`CudaLocalTransfer` 对各配置 device 建立一个独立 non-blocking copy stream。

copy 前会记录 execution-ready event，使 copy stream 等目标分配等前序工作；读取 source 时等待 source_ready event；copy 后记录 completion event。HostStaged 还有 source staging 完成事件，目标 stream 等它后才能 H2D。

`CudaTransferRequest` 持有 copy-stream owner、提交过的 stream、辅助 event、completion event 和 staging。其析构若尚未 wait，会等 completion event，或在提交失败/未记录 completion 时同步已提交 stream，然后销毁 events。因此销毁最后一个 request 引用本身也可能阻塞。

`PoseidonGpuValue::ReadyEvent` 可包裹普通 compute event、CudaTransferRequest 或 NCCL Request。compute 消费者通常使用 `cudaStreamWaitEvent`，同一线程同设备的普通 compute producer/consumer 可以依靠同一 stream 顺序跳过额外 wait；CPU 显式 synchronize 使用 event/request wait。

来源：`P/src/poseidon/runtime_api/communication/cuda_local_transfer.cpp:134`、`:203`、`:472`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:736`、`:840`。

### 10.6 跨 MPI rank 的 GPU 通信

NCCL transport 由 MPI control communicator bootstrap 全局 GPU clique；每个本地 GPU 有 NCCL communicator 和 non-blocking stream。payload 用 uint32 buffer send/recv，MPI 不传 GPU 对象 header。

接收端直接用 output ValueDesc、context 和目标 CUDA device 计算 degree、q_count、components、bytes 并 allocate。NCCL submit 使用 grouped send/recv，group_end 后记录完成 event，Device output 可以立即 posted。recv stream 还等待 allocation-ready event。

支持：跨 rank Device→Device 完整对象；跨 rank Host→Device 时，源 rank 先把 Host 对象上传到其逻辑 device 0 作为 NCCL staging，再发送。这个 staging GPU 值属于 API CommHandle 内部，不是计划独立 ValueId。

不支持：跨 rank Device→Host 或 Host→Host 的 GPU backend 动作。GPU worker 模式的在线跨 rank 动作也有前述 Device-only 限制。

NCCL Request 保留 event 和通信状态，析构/显式 wait 等待完成；原始源 payload 的寿命由 API 的 in-flight storage 引用等维持。Host-source 的 source_staging 与 source_staging_request 由 CommHandle 持有，在 wait 收尾时清理。

来源：`P/src/poseidon/runtime_api/communication/nccl_mpi_transport.h:13`、`:35`；`P/src/poseidon/runtime_api/communication/nccl_mpi_transport.cpp:42`、`:398`、`:450`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:632`、`:1869`、`:1976`。

## 11. 普通 SSA 值之外的 GPU 内存

### 11.1 参数表

API 为每个配置设备建立 `GpuParameterData(context, device)`。它维护各 level 的 GpuLevelInfo / GpuParameterShard，包含 primes、modulus constants、NTT/INTT tables、rescale 常量、HYBRID decomposition 和 Q/P conversion tables 等 DeviceVector。

这些资源由 DeviceState 持有，普通 RuntimePlan `values` 没有对应 ID。它们不是 Encode/Transfer 动作产生的用户常量。

来源：`P/src/poseidon/gpu/gpu_parameter.h:28`、`:235`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:719`、`:1139`。

### 11.2 求值密钥的预加载与缓存

PlanVerifier 从 Rotate/Relinearize 生成 requirements；GPU preflight 在 initialization 前读取 requirements，按本地 Place/input level 加载需要的 keys，并在末尾同步各 device。普通在线 Rotate / Relinearize 从 cache 取 key；对应 level 未预加载则报错，不在 compute 中动态 upload。

DeviceState 中已有：

```text
relin_keys_by_q_count
galois_keys_by_q_count
galois_elements_by_q_count
```

Galois upload 只选需要的 elements；后续 preflight 增加新需求时合并集合并替换该 q_count 的缓存对象。Rotation 可由 binary basis 分解，产生多次 rotate 和中间 ciphertext。

底层 `GpuEvaluationKeyData` 的 owning layout 保留完整 `[Q_storage | P]`；`make_const_view(active_q_count)` 以非 owning Q-prefix view 读取不同运行层，P 的 offset 仍基于 storage_q_count，不进行 compact key copy。

**底层支持零拷贝多 level view，不表示 API 当前所有 q_count cache entries 共用一个 key allocation。** API 对每个 q_count 调用 uploader 并创建独立 cache object；当前没有将这几个 entries 统一映射到一个 owner 的代码。Native Boot profile 则另外提供 shared immutable key 接口。

Key 不属于 Runtime `ValueKind`，也不通过 Runtime Transfer/Replicate 管理；本地 cache 生命周期与 API/DeviceState 相连，没有通用逐最后使用驱逐逻辑。

来源：`R/runtime/verifier.cpp:252`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:2503`、`:2762`、`:2789`、`:2824`；`P/src/poseidon/gpu/gpu_key.h:49`、`:95`；`P/src/poseidon/gpu/gpu_uploader.h:180`。

### 11.3 Handler 的持久 scratch

`GpuEvaluator` 包含 elementwise、NTT、modswitch、keyswitch handler。工作区里有不属于输出 ciphertext 的 DeviceVector：

- `GpuModSwitchHandler::RescaleScratch` 保存 q_last、correction/NTT correction、两层/多层 drop 临时量、centered remainder 等；ensure-capacity 在够用时复用，扩容前部分路径 cudaDeviceSynchronize。
- `GpuKeySwitchHandler::PersistentWorkspace` 保存 relinearize 的 HybridScratch；相同 device/足够容量时复用。
- `GpuQPCiphertextBuffer`、`GpuQCiphertextBatchBuffer`、`GpuHybridKeySwitchWorkspace` 有自己的 ensure_capacity，较小逻辑请求可以使用更大已分配块。
- `GpuDoubleHoistWorkspace` 包含 source/outer hoist、baby tile、group accumulator、batch scratch、inner/result ciphertext，并有 `max_workspace_bytes`、`workspace_peak_bytes` 等局部控制/统计字段。

这些容量复用机制存在于特定 GPU 算法内部，不是 Runtime 的 ValueStore buffer pool，也没有统一写入 OperatorSpec/RuntimePlan。逻辑 shape 下降可能保留更大 allocation。

来源：`P/src/poseidon/gpu/gpu_modswitch_handler.h:95`；`P/src/poseidon/gpu/gpu_modswitch_handler.cpp:458`、`:555`；`P/src/poseidon/gpu/gpu_keyswitch_handler.cpp:933`、`:2200`；`P/src/poseidon/gpu/gpu_double_hoist.cpp:27`；`P/src/poseidon/gpu/gpu_double_hoist.h:214`。

### 11.4 native Boot 的常驻资源和工作区

当前 native Boot 的资源通过 setup 显式构造和安装：`GpuBootstrapProfileBuilder::build` 生成 CPU plans/keys/constants 并 upload；`configure_native_bootstrap` 按本地 device 和 operator_profile 安装到 `native_bootstrap_by_profile`。

每个 NativeBootstrapState 持有：GpuBootstrapData、shared const relin/galois keys、GpuBootstrapWorkspace。Workspace 包括 ModRaise、CoeffToSlot、EvalMod、scratch0…scratch5、basis/nodes vectors、C2S/S2C double-hoist 工作区和可选 trace buffers。

GpuBootstrapData 包含 GPU-resident matrix plaintext、QP 扩展对角线、EvalMod polynomial/basis/combine plan、预编码常量及相关 metadata。QP matrix plaintext 还已有 exact periodic `GpuCompressedPlaintextQP`：保存 `[Q|P][period]` residues，按 bit-reversal 周期映射重构，不是 float 量化。此表示属于 Boot/linear-transform 内部，不是普通 Runtime Encode value 的通用压缩格式。

native Boot 在 RuntimePlan 中仍是一条 Compute；内部中间节点、密钥、matrix plaintext 和 workspace 都没有独立 Runtime ValueId。Shared-key overload 能使同一设备的多个 Boot profiles 共享不变 keys，而不是重复安装每份 key payload。

另一方面，decrypt_reencrypt Boot 在 Host 解密/解码/重新编码/加密；数据进出 GPU 由显式 Transfer 描述。不能把 native Boot 与 Host 模拟 Boot 的临时量、密钥和传输需求视为同一路径。

来源：`P/src/poseidon/gpu/gpu_bootstrap_profile.h:19`、`:51`、`:72`；`P/src/poseidon/gpu/gpu_evaluator.h:127`、`:265`；`P/src/poseidon/gpu/gpu_plaintext.h:106`；`P/src/poseidon/runtime_api/poseidon_gpu_api.cpp:704`、`:1213`、`:1375`。

## 12. 现有“预取/释放”描述与实际落地状态

已有 `architecture.md` 明确描述首期策略：初始化时绑定输入、Encode 常量、把本次需要的值预加载到目标设备，并保留到运行结束。现有 compiler/Runtime 在输入和常量传输进入 initialization、初始化通信 drain 这两点上与此对应；API 的 in-flight 保留引用还可能把对象寿命延长到 API 析构或跨多次 run。

同一文档的后续大数据策略已经写过“内存规划 Pass，根据显存容量和最后一次使用决定驻留时间窗，插入 Prefetch 和 Release/Evict，必要时从 Host 重新物化”。这是**仓库已有目标描述**，不是本文提出的方案；当前没有相应 TableGen op、RuntimePlan kind、Runtime 执行分支或 API release/prefetch 合同。

| 与新增功能相关的能力 | 当前状态 |
| --- | --- |
| 对完整值显式 Host→GPU 搬运 | 已实现 Transfer，支持初始化及本地 execution 中的动作 |
| 显式 GPU→Host 搬运 | 已实现本 rank D2H，Host output 到 wait 后才 materialize |
| GPU→GPU / 多 rank 通信 | 已实现本地 CUDA copy 与跨 rank NCCL，受布局及端点限制 |
| 数据传输异步提交、消费时 stream dependency | 已实现 posted outputs + ReadyEvent/request |
| 输入/常量预加载 | 已实现，compiler 放到 initialization |
| 计算生产者后提前搬运 | 已实现 communication Pass 的插入位置；不是容量约束预取策略 |
| CPU/GPU 物理块释放与池复用 | 底层 release / RAII / allocator 已有 |
| Runtime 普通值最后使用后释放 | 未实现；store / parallel map 不做 last-use 删除 |
| API in-flight 完成后自动回收 | 未实现；保留集合只追加 |
| 任意 Runtime Compute 的输入 buffer 原位接管 | 未实现；无对应 IR/plan/API 标记或合同 |
| 某些 evaluator 预分配输出复用、内部 inplace | 已实现，算子与局部算法相关 |
| Prefetch / Release / Evict 显式计划指令 | 未实现，只有已有文档的后续描述 |
| GPU 容量约束、通用峰值/驻留规划 | 当前 placement 未实现 |
| Unified Memory prefetch | 本次检查的项目代码路径无 cudaMallocManaged / cudaMemPrefetchAsync 实现 |

来源：`R/docs/overview-design/architecture.md:284`、`:298`、`:398`；前文所列代码。

## 13. 现有计划实例和可复核的存储规模

### 13.1 已提交 MLP 计划的一段真实数据流

`P/runtime-artifacts/65536/plans/gpu/mlp-4gpu.runtime-plan.json` 的开头数据流可简写为：

```text
initialization ordinal 0:
  Transfer id=0, Host(rank=0) ValueId=0
      → Device(rank=0,index=0) ValueId=497
  ciphertext: components=2, level=39, scale_log2=40, ntt=true

initialization ordinal 1:
  Encode(bundle sha256:a46099…9699cb) → Host(rank=0) ValueId=2
  plaintext: components=1, level=31, scale_log2=40, ntt=true

execution ordinal 243:
  ModSwitch Device(0,0), input=497 → output=1, target_level=31

execution ordinal 244:
  Transfer id=122, Device(0,0) ValueId=1
      → Device(0,3) ValueId=619
  ciphertext: components=2, level=31, scale_log2=40, ntt=true
```

可以直接看到：输入在 Host 编码表示中存在；初始化上传定义新的 Device ID；ModSwitch 又定义一个新 ID；跨卡副本再定义另一个 ID。四个 ID 并不表达对同一 allocation 的顺序覆盖。该计划的最终输出为 ValueId 496，finalization 为空。

### 13.2 当前已提交负载的动作数量

以下从该目录 JSON 逐项计数，不是运行报告中的性能数字：

| 计划，相对 `runtime-artifacts/65536/plans/` | ValueDesc 数 | 初始化 Encode | 初始化 Transfer | execution Compute | execution Transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu/mlp.runtime-plan.json` | 497 | 121 | 0 | 375 | 0 |
| `gpu/mlp-1gpu.runtime-plan.json` | 619 | 121 | 122 | 375 | 0 |
| `gpu/mlp-4gpu.runtime-plan.json` | 814 | 121 | 122 | 375 | 195 |
| `gpu/mlp-4x4.runtime-plan.json` | 750 | 121 | 122 | 375 | 131 |
| `gpu/probe-1gpu-1999.runtime-plan.json` | 2001 | 0 | 1 | 1999 | 0 |
| `gpu/probe-4gpu.runtime-plan.json` | 2392 | 0 | 4 | 1999 | 388 |

上述计划的 finalization 全为空。MLP 是编译器产物；probe 是专门用于并行与通信实验的负载，不能把它的数值输出当作 MLP 结果。

### 13.3 单对象 payload 大小和所有描述符求和

根据当前完整 Q-only 布局，一个普通 GPU 值的 payload 为：

```text
N × (level + 1) × components × 4 bytes
```

对应 Host residues 的逻辑 payload 用 8 bytes/word。内部 QP full allocation 用 `(q_count + p_count)`；key/batch/hoist 的形状还有额外维度。

N=65536 时：

| level | Q limbs | plaintext，1 component | ciphertext，2 components | ciphertext，3 components |
| ---: | ---: | ---: | ---: | ---: |
| 39 | 40 | 10 MiB | 20 MiB | 30 MiB |
| 31 | 32 | 8 MiB | 16 MiB | 24 MiB |
| 15 | 16 | 4 MiB | 8 MiB | 12 MiB |

对上表计划中每个 Device ValueDesc 的逻辑 payload 求和，得到：

| 计划 | 各 device 的所有 ValueDesc payload 求和，MiB |
| --- | --- |
| MLP 1 GPU | Device(0,0)：6842 |
| MLP 4 GPU | Device(0,0)：2866；(0,1)：2439；(0,2)：2350；(0,3)：2233 |
| Probe 1 GPU，1999 Compute | Device(0,0)：32004 |
| Probe 4 GPU | Device(0,0)：9572；(0,1)：9588；(0,2)：9572；(0,3)：9540 |

这些是**全部描述符逻辑 payload 的静态加和，不是实测峰值、最小工作集或显存预算**。它没有模拟生存期，也没有包含 RMM 保留/对齐、参数表、求值密钥、Boot workspace、内部 Rotate/Rescale intermediates、通信 staging、事件和 CPU slots。现有 store 保留所有值的行为与这些总量有关，但不能据此直接报告实际进程占用。

来源：`P/runtime-artifacts/65536/plans/`；`P/runtime-artifacts/65536/profiles/`；`P/src/poseidon/gpu/gpu_ciphertext.cpp:219`；`P/src/poseidon/gpu/gpu_plaintext.cpp:125`。

## 14. 已有测试、观察工具和构建入口

### 14.1 与上述行为对应的测试源码

| 位置 | 已有覆盖或用途 |
| --- | --- |
| `D/test/runtime-plan/check.py` 和同目录 MLIR/JSON | Compute/export、inline/bundle、fanout placement、通信 profile/rules、Host Boot placement、lazy physical levels 与非法元信息 |
| `R/tests/runtime_tests.cpp` | 算子元信息、Host Encode/compute、bundle 内容复用/本地 rank 加载、Mock 通信、Device workers、双向依赖和失败路径 |
| `R/tests/runtime_plan_json_tests.cpp` | reader/schema 严格验证 |
| `R/experiments/dacapo_plan_vec_diff.cpp` | 用 AllValuesAfterRun 对 Host 和分布式计划的 Encode/Compute/Transfer 谱系及值做差分 |
| `P/src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp:201` | pool 比 API 活得更久，最后 GPU value 消失后恢复旧 resource |
| 同文件 `:394` | H2D 可立即 posted，消费者提交不先等 payload；D2H 不提前发布 Host 值 |
| 同文件 `:1014` | Rescale/metadata，synchronize 不释放已保留输入 |
| 同文件 `:1170` | 求值密钥 preflight preload，Mul/Relin/Rescale/composite Rotate |
| 同文件 `:1286` | 逻辑/物理双卡映射、Replicate、跨卡值及计算 |
| `P/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_transfer_test.cpp` | 跨 rank GPU 对象通信 |
| `P/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_runtime_e2e.cpp` | 任意多 rank GPU RuntimePlan，warmup/iterations、online 计时 |
| `P/src/poseidon/tests/runtime_api/poseidon_gpu_bootstrap_e2e.cpp` | Native profile 构造/安装和 Runtime Boot 执行 |

表中记录测试源码的用途，不宣称本次已重新运行，也不把现有测试当作 Release/Evict/Prefetch 已通过验证的证据。

### 14.2 现有观察手段

RuntimeTiming 有 setup、initialization、online、compute/boot 和 communication post/wait 计时。异步 API 的 compute duration 是 CPU 调用/提交时间，不是该 kernel 的 GPU completion wall time。

`POSEIDON_RUNTIME_TRACE` 输出按 rank 的 compute/comm_post/comm_wait CSV；`R/runtime/thread_trace.*` 记录线程 duration、等待与锁。Runner 还有 setup/warmup/online.iteration.N NVTX scopes；`P/scripts/analyze_nsys_kernel_gaps.py`、`analyze_nsight_online.py` 分析 GPU kernels、stream idle gaps 和 online 范围。

RMM 测试 scope 使用 `limiting_resource_adaptor` 的已分配字节计数观察 allocation；这不是 RuntimePlan 已提供统一 value residency/peak-memory 报告。DoubleHoist 的 workspace bytes 字段同样只对应局部工作区。

来源：`R/runtime/runtime.hpp:67`、`:201`；`R/runtime/thread_trace.hpp`；`P/src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp:142`；`P/scripts/analyze_nsys_kernel_gaps.py`。

### 14.3 构建关系

- `POSEIDON_BUILD_CKKS_RUNTIME_API` 默认 OFF；开启后接入 Runtime core / JSON 与 Poseidon CPU/GPU API。
- `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` 可以切换 Runtime checkout，默认 bundled submodule。
- `POSEIDON_BUILD_CKKS_RUNTIME_MPI` 控制 CPU MPI；`POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL` 独立控制 GPU MPI/NCCL。
- `POSEIDON_BUILD_CKKS_RUNTIME_TESTS`、`POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS` 控制对应测试。
- dacapo 是独立 MLIR/CMake 工程，普通 Poseidon 构建不编译它；当前推荐 LLVM/MLIR/Clang 18.1.8 环境。
- GPU 的 RMM/CCCL 等依赖已由本仓库维护供离线构建；RMM 栈版本说明见 `P/third_party/README.md`，当前记录为 RMM v24.12.01。

来源：`P/CMakeLists.txt:278`；`P/src/poseidon/runtime_api/CMakeLists.txt:1`；`P/src/poseidon/gpu/CMakeLists.txt:1`；`D/README.md`；`P/third_party/README.md`。

## 15. 按模块查阅源码的索引

| 模块 | 主要源码 |
| --- | --- |
| Earth 类型及优化 | `D/include/hecate/Dialect/Earth/IR/EarthOps.td`；`D/lib/Dialect/Earth/Analysis/`；`D/lib/Dialect/Earth/Transforms/` |
| Earth→CKKS lowering | `D/lib/Conversion/EarthToCKKS/EarthToCKKS.cpp`；`D/lib/Conversion/CKKSCommon/PolyTypeConverter.cpp` |
| CKKS 算子/验证 | `D/include/hecate/Dialect/CKKS/IR/CKKSOps.td`；`D/lib/Dialect/CKKS/IR/CKKSDialect.cpp` |
| Dist Transfer | `D/include/hecate/Dialect/Dist/IR/DistOps.td`；`D/lib/Dialect/Dist/IR/DistDialect.cpp` |
| 目标 Pass | `D/include/hecate/Dialect/CKKS/Transforms/Passes.td`；`D/lib/Dialect/CKKS/Transforms/{MaterializePhysicalLevels,AssignPlacement,MaterializeCommunication,EmitRuntimePlan}.cpp` |
| 编译管线入口 | `D/tools/optimizer.cpp`；`R/integrations/dacapo/generate_model_artifacts.py` |
| RuntimePlan / OperatorSpec | `R/runtime/plan.hpp`；`R/runtime/operator_spec.hpp`；相应 JSON reader；`R/runtime/verifier.cpp` |
| Runtime 值/线程/通信管理 | `R/runtime/runtime.hpp` |
| Bundle | `R/runtime/plaintext_bundle.*`；`D/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp` 的 PlaintextBundleWriter |
| CPU 内存对象 | `P/src/poseidon/{ciphertext,plaintext,rns_poly}.*`；`P/src/poseidon/basics/{dynarray,memorymanager}.h`；`basics/util/{pointer,mempool}.*` |
| CPU API | `P/src/poseidon/runtime_api/poseidon_cpu_api.*` |
| GPU Value / API / 内部 pool / in-flight | `P/src/poseidon/runtime_api/poseidon_gpu_api.*` |
| GPU allocation 与所有权 registry | `P/src/poseidon/gpu/gpu_memory.*` |
| GPU layout | `P/src/poseidon/gpu/gpu_{rns_poly,ciphertext,plaintext,key}.*` |
| GPU evaluator 和 scratch | `P/src/poseidon/gpu/gpu_evaluator.*`；`gpu_{modswitch,keyswitch,ntt,elementwise}_handler.*`；`gpu_double_hoist.*` |
| Upload 与 native Boot setup | `P/src/poseidon/gpu/gpu_uploader.*`；`gpu_bootstrap_profile.*` |
| 本地对象 copy 和流 | `P/src/poseidon/runtime_api/communication/{gpu_object_copy,cuda_local_transfer}.*`；`device_buffer.h` |
| GPU MPI/NCCL | `P/src/poseidon/runtime_api/communication/nccl_mpi_transport.*` |
| 已提交负载 | `P/runtime-artifacts/65536/` |

本文件中的接口、示例、持有关系与限制均用于解释上表版本的现有系统。任何新内存 IR、协议扩展或调度方法都不属于本文内容。
