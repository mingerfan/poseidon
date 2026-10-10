# 权重明文按需编码、分批上传和提前上传实施方案

状态：源码能力调查与后续任务方案，**尚未实现本文提出的新 Pass、V3 或 Fence**。

评估日期：2026-10-10。基线为 Poseidon `3b911694`（`integration/runtime-gpu`）、
ckks-runtime `641e2ee`（`integration/source`）、DaCapo `a5bae92`
（`integration/multi`）。本地与 188Server 的 `/home/xuming/poseidon` 使用这组代码。
部分早期设计文档的实现状态已落后于源码；本文按当前执行器、Verifier 和 GPU API 判断。

## 1. 结论与实现目标

**现有设施已经能在执行阶段上传一部分明文，再继续计算，再上传下一部分。**
Transfer 不限于初始化，GPU API 已有异步 H2D、独立拷贝流、完成事件和异步对象保活。
对一个能够预先把明文编码放进主机内存的小模型，只推迟 Transfer，主要是编译器调度工作。

**完整 Qwen 的有界权重流水还需要修改编译器、计划协议和 Runtime。** 原因是：

1. Encode 被强制放进初始化；只推迟上传，会让所有编码后的 RNS 明文堆在主机。
2. bundle 加载器启动时读入本 rank 所需的全部原始载荷，没有有界按需缓存。
3. `PerDeviceWorkers` 拒绝在线 Encode，也拒绝 Host Compute；当前 Qwen 计划包含 Host Boot。
4. Runtime 和 GPU API 可以快速连续提交异步任务；Release 后，GPU 尚在使用的对象仍占内存。
   当前没有为权重预取设置驻留预算、提交窗口或预算不足时的阻塞机制。
5. 异步 H2D 不等于已经获得计算重叠。当前 H2D 会等待目标执行流上的事件，具体时机可能
   让上传排在已提交计算之后。

目标是显式编译出“少量初始权重 + 执行阶段按窗编码/上传 + 最后使用后释放”。
先交付可证明内存受限的版本，再优化预取距离和计算重叠。不要同时重做 boot 插入、
CKKS 参数、矩阵布局、placement 或模型数学逻辑。

## 2. 当前源码支持什么

| 能力 | 当前状态 | 对这项工作的含义 |
| --- | --- | --- |
| execution 中的 Transfer | 已支持 | 可以表达计算途中上传，不必增加一个 Prefetch 数据算子 |
| execution 中本 rank 的 Host→Device Transfer | 顺序模式和设备 worker 的本机通信路径已有支持 | 多卡上传并非必须全在初始化 |
| 异步 H2D、pinned staging、完成事件 | 已支持 | 复用现有 CUDA 搬运，不另写一套复制代码 |
| 消费者等待输入事件 | 已支持 | 未完成的上传结果可以先发布句柄，计算流按事件等待 |
| Release、使用次数和在途对象保活 | 已支持 | 延迟释放的正确性设施可以复用；不代表已有内存预算保证 |
| execution 中 Encode | V1/V2 Verifier 禁止 | 顺序执行分派虽然认识 Encode，合法计划仍无法使用它 |
| 设备 worker 中 Encode / Host Compute | 明确拒绝 | 并行在线编码及当前 Host Boot 都需要新增 Host 任务执行路径 |
| blob 按需读取和有界缓存 | 没有 | 现有 `slots_by_content` 在启动时装入所需数据，并保留到本次运行结束 |
| 编译器权重使用区间、预取窗口、预算调度 | 没有 | 当前每个源值/目的地共用一个 Transfer，放在生产者后面 |
| 并行/在途显存上界 | 没有 | 现有报告按指令完成后再执行下一条估算，不能作为硬上限 |

源码入口如下，后续实现应先读这些函数：

- [Runtime](../third_party/ckks-runtime/runtime/runtime.hpp)：`run`、`load_bundle`、
  `execute_encode`、`execute_communication`、`compile_parallel_phase`、
  `local_communication_worker`、`execute_device_parallel_phases`、`execute_release`。
- [Verifier](../third_party/ckks-runtime/runtime/verifier.cpp)：Encode 阶段限制、SSA 定义/使用、
  Release 与 `reuse_input` 检查；[JSON reader](../third_party/ckks-runtime/runtime/json_plan_reader.cpp)
  目前只接受版本 1/2。
- [bundle 加载器](../third_party/ckks-runtime/runtime/plaintext_bundle.cpp)及
  [接口](../third_party/ckks-runtime/runtime/plaintext_bundle.hpp)：当前一次性构建 `slots_by_content`。
- [GPU API](../src/poseidon/runtime_api/poseidon_gpu_api.cpp)：`encode_plaintext`、
  `prepare_plaintext_upload`、`communicate_async`、`posted_outputs`、`retain_communication`、
  `retain_in_flight`、`collect_completed`、`drain`。
- [CUDA 搬运](../src/poseidon/runtime_api/communication/cuda_local_transfer.cpp)：
  `copy_host_to_device_async`、`record_execution_ready`；消费者等待在 GPU API 的
  `ReadyEvent::wait_on_execution_stream`。
- [通信生成](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/MaterializeCommunication.cpp)：
  `isInitializationValue`、`TransferDemand`、`materialize`。
- [计划布局](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/RuntimePlanUtils.cpp)：
  `isRuntimeInitialization`、`buildRuntimePlanLayout`。当前所有 Encode 都被归入初始化。
- [JSON 导出](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp)：
  Encode 分支直接写入 `initialization`。
- [Release 规划](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/PlanRuntimeMemory.cpp)、
  [内存估算](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EstimateRuntimeMemory.cpp)、
  [流水线注册](../third_party/ckks-runtime/third_party/dacapo/tools/optimizer.cpp)：新调度必须进入这些公共路径。

## 3. 这次完整模型暴露的问题

基线是合成权重、完整 24 层、两 token prefill、完整 LM head 和 96 个 KV 输出。
默认配置中的 512 是结构上界，本次只编译了位置 0/1。已有单卡/四卡计划都带 Release/reuse，
并已做单位明文消除和 Encode CSE。

| 当前计划 | 全部 RNS 对象峰值 | 单独计算密文存活峰值 |
| --- | --- | --- |
| 单卡 | 8.08 TiB | 192.87 GiB |
| 四卡，每卡 | 2.65、2.52、2.40、2.24 TiB | 8.69、16.70、11.37、9.42 GiB |

这些是顺序完成假设下的对象估算，排除密钥、参数表、工作区、暂存、缓存、内存池保留和对齐等。
完整模型尚未在 CKKS/GPU 上执行。四卡密文估算说明值得继续实现权重流水，不能据此承诺四张
32 GiB V100 已能运行。单卡在保持当前计算顺序和密文存活区间时，仅去掉权重峰值也装不下。

原始去重后的 bundle 为 14.70 GB；它与编码后的 RNS 权重大小是两件事。N=65536 时，
40 个 q limb、无 p limb 的一份 GPU 明文为 `65536 × 40 × 4 = 10 MiB`，同形状 Host
明文为 20 MiB，H2D packed staging 又需要约 10 MiB。计费必须使用实际 q/p limb，不能
按短 payload 的元素数估算编码后的明文；前缀掩码仍占完整多项式。

结果证据见 [最新完整编译记录](../test-results/qwen24-dacapo-188-20261010/README.md)及
[最新摘要](../test-results/qwen24-dacapo-188-20261010/unit-plaintext/summary.json)。大产物在：

```text
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-native
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-unit-plaintext/gpu-plans/{gpu1,gpu4}
```

无损源文件是 `qwen24-native/traced/trace_qwen24.mlirbc`，约 14.82 GB。诊断文本 MLIR 省略
常量，不能用于重新编译。复用现有字节码，不要重新生成数 GB 权重或重新跟踪模型。

## 4. 分阶段交付

### P0：验证已有执行期上传能力

先用手写小型 V2 计划，将两个或三个权重 Encode 留在初始化，把 H2D Transfer 分散到
execution，安排独立计算夹在上传之间；分别覆盖 Sequential 和单 rank 多设备 worker。
加入 Host/GPU plaintext Release，检查事件依赖、数值结果和上传/释放轨迹。

此阶段无需改协议。它用于证明上传基础设施可复用，不能作为完整模型的内存解决方案，
因为初始化后的 Host RNS 明文仍全部存在。现有大 Qwen 计划的 Host Boot 也不能直接交给
当前 `PerDeviceWorkers`；这里先使用没有 Host Compute 的小型探针。

### P1：有界在线编码和顺序提交 MVP

建议增加 RuntimePlan V3，允许 Host Encode 出现在 execution，继承 V2 的 Release/reuse，
并增加下文的窗口 Fence。保留 V1/V2 的旧限制，旧计划仍按旧语义运行；不要悄悄放宽 V2。

先支持 `world_size=1`、一个或多个本地 GPU、`DeviceExecutionMode::Sequential`。
此模式已有 Host Boot 分派，可先验证完整现有图的上传与内存行为。顺序提交仍会提交异步
GPU 工作，所以必须有窗口完成边界，不能把“顺序提交”当作“每个算子已经完成”。

同时实现按需 blob 读取、Host 编码预算、GPU 对象预算、pinned 预算、权重使用区间切分，
以及每个窗口结束时的完成等待与回收。首先交付预取距离为零的正确版本，再在窗口内提早上传。

### P2：单 rank 多卡 worker 与 Host 任务

为 Encode 和已有 Host Boot 增加一个串行 Host 任务执行路径，复用 `ParallelValue` 的发布、
依赖等待、使用次数、失败通知和释放机制。GPU worker 继续执行本地 Transfer 和 Device Compute。

GPU API 的 `encoder_` 同时被 Encode 和 Host Boot 使用，不能默认线程安全。首版把两者
放在同一个 Host worker 串行执行；不要直接在多个 GPU worker 上同时调用该 encoder。
Host Release 也需要正确路由，不应继续简单塞到 GPU worker 0。

每个窗口内可并行执行，窗口结束后等待本窗口的 Host/GPU/通信任务完成，才能提交下一窗口。
不要为全部窗口一次性启动可无限向前执行的任务队列。跨窗口的活值、使用次数和上下文
必须保留，不能像结束一次 `run()` 那样重置 ValueStore。第一版明确拒绝跨 rank streaming；
现有设备 worker 只接受 Device-only 跨 rank 在线通信，MPI/NCCL 协调需另做任务。

### P3：优化提前上传与窗口等待

取得真实的 blob 读取、CPU Encode、packing、H2D、算子耗时和实际内存统计后，调预取距离。
可以再用事件驱动的预算归还减少全窗口 Fence，但必须保持 P1/P2 已建立的内存和释放约束。
不要第一版就引入通用动态 DAG 调度器、自动驱逐所有密文或磁盘 RNS 编码缓存。

## 5. 推荐协议：显式指令、V3 和窗口 Fence

Prefetch 本质上是把已有 `Encode → Transfer` 提前到消费者之前；不需要一个运行时隐式
查找和搬运权重的特殊数学算子。V3 保留完整 ValueDesc 和显式 SSA 值。

建议新增不产生值的 `fence` 指令，例如：

```json
{"ordinal": 123, "kind": "fence"}
```

首版语义：当前 rank 本窗口内的已提交工作完成；通信结果被交付、完成的在途引用被清理。
Fence 不释放后续仍要使用的值，不改 level/scale，不解密，不能用它代替 Release。
首版只接受单 rank streaming，暂不定义分布式 barrier。

窗口示意：

```text
initialization: 绑定输入、必要输入搬运、少量决定常驻的明文
execution:
  Encode(w0) → Transfer(w0) → Release(Host w0)
  Encode(w1) → Transfer(w1) → Release(Host w1)  # 预算许可时提前
  Compute(x, GPU w0) → Release(GPU w0)
  Compute(y, GPU w1) → Release(GPU w1)
  Fence
  Encode(w2) → Transfer(w2) → ...
```

Fence 可复用完成通信及 `Api::drain()` 的底层能力，先实现较保守的窗口边界，不在每个权重
后全设备同步。别在每个 Fence 从头扫描所有历史通信组；只处理本窗口未完成的组，及时
清掉已交付的句柄和暂存。丢弃组或重用索引前，必须证明没有 Pending 值还引用它。
`drain()` 当前只处理 API 持有的在途工作，不能代替 Runtime 的通信结果交付。

需要同步更新 JSON schema/reader、Verifier、计划打印、Runtime dispatch、编译器
`dist.fence`、布局、内存规划和导出。布局目前只允许 Release 没有结果，新 Fence 也必须
作为无结果指令处理。Fence 应有调度副作用，不能被 CSE/DCE 删除或随意跨越。
格式版本检查以 V3 为准；不认识 V3/Fence/在线 Encode 的执行器直接报错。

## 6. 编译器 Pass 的边界与算法

建议新增 `PlanPlaintextStreaming`，放在下面位置，所有 RuntimePlan 导出入口共用：

```text
既有 scale/boot 优化 → Upscale 转换 → Encode CSE → 物理 level
→ AssignPlacement → MaterializeCommunication
→ PlanPlaintextStreaming（新）
→ PlanRuntimeMemory → EstimateRuntimeMemory → EmitRuntimePlan
```

第一版固定当前计算、placement、boot 和密文通信顺序，只安排来自 Encode 的权重。
计算输入、输出及最终 CKKS 元信息应保持；新增或拆分的 Encode/Transfer 使用唯一 ID。
不能只在 JSON writer 挪数组，内存分析和所有消费方必须看到同一份真实 IR 顺序。

建议做一次扫描，收集每个编码值的消费位置、目的设备、物理大小和共享关系。按当前计算
顺序贪心形成有界窗口，然后在窗口内按首次使用时机安排 Encode/Transfer：

1. 维护窗口开始时仍存活的密文和常驻明文，以及窗口中新建对象的字节数。
2. 加入下一步计算所需的权重；如果超过任一设备/Host/pinned 预算，就结束当前窗口。
3. 容量允许时，提前加入后续将使用的权重；预取距离用预计时间或实际字节数控制。
   不用“提前 1000 条指令”作为唯一标准，Rotate 与 Boot 的时间差很大。
4. 单个步骤连同必要活值和资源余量都装不下时，编译器报出设备、需要字节数及超限对象。
   不偷偷切回全量初始化或无限缓存；进一步切模型/改变 placement 属于后续工作。
5. 形成窗口后，再运行 Release/reuse 规划和估算，不复用旧计划的 Release 位置。

同一个权重可能跨很远的位置使用，现有“每源值/目的地一个 Transfer”的共享会拉长驻留。
首版允许按窗口重新编码/上传，使用同一个 bundle content、相同 level/scale/NTT 和新
ValueId/TransferId；宁可增加少量编码和传输，也不要为了 CSE 保留大权重到最后。
多设备间隔很远的上传也可以拆开 Host Encode，避免等最后一张卡时持有巨量 Host RNS。

小型且频繁使用的系数可以常驻，但也算入预算；不能默认所有共享值都常驻。
窗口拆分后不要再跑能把相同 Encode 跨窗口合并的 CSE。保留现有单位明文优化，满槽单位
与补零前缀掩码的区别、编码 scale 都不能改变。

修改公共 `isRuntimeInitialization`，让在线 Encode 的阶段能显式表达；同时修改
`buildRuntimePlanLayout`、通信初始化标记和 exporter 的 Encode 分支，防止后面的 Pass
又把已经安排到 execution 的权重挪回初始化。校验不存在定义前使用、跨 Fence 的非法
移动、重复 ID、已释放后使用或覆盖仍有消费者的输入。

算法应接近 `O(V+E)`，最多增加索引排序；不要每安排一个权重就重建整个计划、重扫活值
或排序全部对象。当前图约 489 万计算步骤，原来的二次复杂度问题不能重新出现。

## 7. 内存约束：首版采用保守窗口上界

至少分别配置并报告：每张 GPU 的对象预算、Host 编码明文预算、pinned staging 预算和
原始 blob cache 预算。预算及预取参数由编译器选项提供，运行时明确验证；名字可以采用
`--runtime-plan-plaintext-schedule=eager|stream` 及相应 `*-budget-bytes` 选项。
这些选项目前不存在，实现时应保持默认 eager 行为。

GPU 对象预算要给密文和预取明文共同计费，不能把整张 32 GiB 都交给权重。密钥、参数表、
算子工作区和 allocator 保留/碎片的余量需根据该 context/设备的实测配置；启动时核对实际
环境。已有顺序对象峰值加一个任意常量，不能冒充实际并行上界。

还要区分长期常驻余量与在途临时区。当前 Rotate、Rescale、keyswitch 等操作可能为每次
调用分配临时对象并保留到事件完成；一个窗口同时在途多个调用时，不能只预留“最大单个
算子的工作区”。首版用该参数/算子实现的保守临时分配上界，累计计入窗口；只有已证明
安全串行复用的 workspace 才能按一份计费。缺少该上界时报告应明确标成 RNS 对象估算，
不能宣称已经给出了总显存保证。资源配置和 allocator 实测要与实际后端匹配。

最简单的可审核上界是：

```text
本窗口对象上界 = 窗口开始的活对象 + 本窗口所有新增分配
```

窗口内不因逻辑 Release 提前扣除在途对象。原位复用按实际分配计费，算子临时区另外预留。
这是保守上界，会牺牲一些窗口长度，但先避免将“API 已接手”错误等同于“内存已回收”。
Host 源明文、pinned 副本、GPU 目的明文可能同时存在，三处必须一起计费；请求预算在
`prepare_plaintext_upload` 分配目的对象和 staging 之前完成。

运行时不能绕过窗口提交后续工作。优化为动态预算后，只能在相关事件完成、所有使用者
结束且底层分配可复用后归还额度。不能在提交 Transfer 或遇到 Release 时立即归还。
等待预算不能持有 ValueStore/通信组/allocator 锁，也不能阻塞负责完成旧消费者的 worker。
缺少合法进展路径应报出超限或循环依赖，不能无限等待。

验收同时记录对象 allocated/used、内存池 reserved、CUDA 空闲量和实际峰值。内存池把空闲
块留在 GPU 是正常行为；对象减少不等于进程实际显存立即下降。

## 8. bundle 和 Host 编码

把“一次性加载 manifest 和所有 blob”拆成：启动读取并验证 manifest/建立长度索引；
执行 Encode 时才读取需要的 blob、检查长度/内容/有限值/槽容量，随后编码。
扫描初始化和执行阶段的本 rank Encode 引用，不能继续只扫描 `initialization`。

保留已有文件完整性和数值验证语义，不另外启动一轮全 bundle 校验。被实际读取的文件
在读取路径内完成校验；缓存命中不重复读文件。不要将未经校验的数据交给 encoder。
默认不建全量 RNS 磁盘缓存，也不将数十 GB blob 复制到 `/tmp`。

首版可不缓存原始 blob，或做简单的字节数受限 LRU。编码完成后的 blob 缓存与 Host
plaintext 生命周期分别管理。共享 content 的 level/scale/NTT 可能不同，不能直接复用
同一编码对象；任何编码缓存的 key 还必须包含 context 和实际编码元信息。
尽量减少当前 `execute_encode` 的多余整份 vector 拷贝，但不能借此改变补零或负零处理。

## 9. 提前上传的性能陷阱

`copy_host_to_device_async` 先在目标执行流上记录 `destination_ready`，拷贝流等待该事件。
这是当前分配与写入顺序的正确性保护。如果下一份权重的上传在一个长 kernel 入队后发起，
这个等待可能把上传也排到长 kernel 后面。这里只能据源码指出风险，实际重叠需要测量。

先尝试在窗口内把后续 Transfer 提到独立计算之前，用现有顺序保护取得重叠。若测量表明
仍需缩小等待范围，再分别表达“目的分配已可写”和“输入数据已就绪”事件。不要直接删除
`destination_ready` 等待，否则 RMM 异步分配及内存重用可能与拷贝写入竞争。

用 Nsight 或现有 trace 确认 H2D 与 kernel 的实际重叠，并分别统计读取、Encode、packing、
copy 和消费者等待时间。真正瓶颈可能是 CPU 编码、几十万小文件或 Host Boot；仅有
`cudaMemcpyAsync` 调用不能证明预取加速。V100 的实际 copy/compute 并发能力也应实测。

## 10. 实现顺序与验收

建议把工作分成独立可审阅的提交：

1. P0 手写计划探针与明确的上传轨迹，证明现有 execution Transfer 路径。
2. V3/在线 Encode/Fence 的协议、Verifier 和顺序 Runtime 支持；V1/V2 旧行为保持。
3. 按需 blob loader 和有界 Host 数据路径；读入/缓存/释放统计。
4. 编译器窗口调度、跨窗口权重拆分、Release/reuse 重规划及窗口上界报告。
5. 多卡 worker 中串行 Host 任务和窗口推进，覆盖 Host Boot 与失败唤醒。
6. 小模型实机验收后，才编译完整四卡计划并调预取距离。

必须覆盖的正确性测试：

- V1/V2 的 execution Encode 仍被拒绝；V3 的合法在线 Encode/Fence 通过，非法 SSA/
  重复 ID/释放后使用/不支持的执行模式被拒绝。
- 小型真实 CKKS 模型的 eager、零距离 stream、提前上传 stream 数值和最终元信息一致。
  满槽 ±1、补零前缀 ±1、共享权重、不同编码 scale/level、跨窗口重新编码都要覆盖。
- Host→GPU0/GPU3 使用时间不同，Host Release 早于某个实际完成事件，重复输入使用、
  Transfer 尚未完成就消费、输出延迟交付，均无提前销毁或重复交付。
- 人为延迟 Encode、copy 或 compute，内存仍受窗口约束；Fence 保留后续仍需使用的
  密文和最终输出；窗口之间不重置上下文、ValueStore 或使用次数。
- 一个权重/一个不可切分计算超过预算时明确失败；GPU、Host、pinned 和 blob cache
  分别超限；多 worker 抛错后其他等待者能退出，无预算等待死锁。
- 至少 20000 个操作的规模回归；数百万指令场景不按窗口反复全图扫描，也不累积
  已完成 pinned buffer 或庞大在途句柄资源。

实机验收先跑小型线性/非线性探针，再跑一层或小词表模型，最后才尝试完整 24 层四卡图。
同时提供计划级窗口对象上界与实测峰值、每卡密文/明文/密钥/工作区占用、Host RSS、pinned
峰值、缓存占用、H2D 字节量/次数、重复编码次数及耗时。明确区分“成功编译”“内存能容纳”
和“完整数值结果通过”，不能把编译完成写成模型能跑。

复用已完成的小型 CPU CKKS scale-absorption 测试和 DaCapo CTest，不为了文档重新跑
大模型哈希校验。新增行为需要有针对性的正确性与内存测试。`/tmp` 只放少量脚本/日志；
大型字节码、模型和计划继续留在 `/home` 或明确指定的工作目录。

## 11. 可直接交给实现者的任务说明

请基于本文列出的三个提交实现 P0→P2，保留现有数学/boot/placement 和单位明文优化。
首版限制单 rank、本地 1/4 卡；采用显式 V3 在线 Encode 和有完成语义的窗口 Fence，
先保证 GPU/Host/pinned/blob 内存受限，再增加窗口内提前上传。复用当前 Transfer、事件
和异步保活，补齐 Host Encode/Boot 执行及按需 bundle 读取，不能只把初始化数组搬到
execution 或只修改 JSON 导出器。预算无法满足时明确失败，不回退到全量加载。

交付代码、版本化协议说明、小模型数值与故障测试、窗口上界及实测内存/耗时报告。
完整四卡 Qwen 是否能跑以实测为准；单卡当前密文峰值问题另行处理。提交时按 DaCapo、
ckks-runtime、Poseidon 的顺序更新子模块和推送，并保持本地与 188Server 仓库一致。
