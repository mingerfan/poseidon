# 让完整模型分批读取和上传权重：目的与后续实现任务

**这项工作的目的：让 GPU 只放接下来要用的少量权重，用完一批，再换下一批。**
这样才有机会把完整 24 层模型放到四张 32 GiB GPU 上运行。

**实现进度（2026-10-10）**：分批权重调度、V3、Fence、按需读取和本机 CPU/GPU worker 已实现，验证情况见[实现与测试记录](poseidon-plaintext-prefetch-validation.zh-CN.md)。完整模型是否实际装得下、最终数值是否通过，需要分别查看实测结果。下文“现有代码”的描述保留实现前基线，作为设计依据。

## 先说明现在为什么会占这么多内存

目前编译出的计划，会在计算开始前，把所有需要的权重编码并上传。这里的“编码”是把普通数字变成 CKKS 运算需要的多项式表示，大小会明显增加。不能用原始权重文件的大小判断编码后是否放得进显存。

这次完整 Qwen 的原始去重数据约为 14.70 GB，但当前计划的内存估算是：

| 计划 | 每卡明文、密文对象的峰值 | 如果只看密文 |
| --- | --- | --- |
| 单卡 | 8.08 TiB | 192.87 GiB |
| 四卡 | 2.65、2.52、2.40、2.24 TiB | 8.69、16.70、11.37、9.42 GiB |

这些数字没有包括密钥、算子临时内存等，而且假设上一条指令完成后才执行下一条。因此，它们是对象大小估算，不能作为实际总显存的保证。

**四卡值得继续做的原因**是：四卡的密文估算最多约 16.70 GiB，剩下的空间有可能容纳少量权重和其他必要数据。具体够不够，要实现后实测。单卡的密文就有约 193 GiB，保持当前计算安排时，仅分批上传权重仍然装不下。

## 要把运行方式改成什么样

目前的方式可以理解为：

```text
读取全部权重 → 编码全部权重 → 上传全部权重 → 开始计算
```

希望改成：

```text
第一批：读取需要的权重 → 编码 → 上传 → 计算 → 用完释放 → 等这批完成
第二批：读取需要的权重 → 编码 → 上传 → 计算 → 用完释放 → 等这批完成
……
```

“一批”是编译器根据内存容量划出的一段任务，可以小于一层。不能假设一整层一定放得下。

在一批任务里面，还可以提前准备稍后要用的权重。例如 GPU 正在用 w0 计算时，CPU 编码 w1，并尝试把 w1 上传到 GPU。这样，GPU 算完 w0 后就少等一会儿。这就是本文说的**提前上传**。

分批解决“能不能装下”，提前上传解决“少等多久”。先把分批版本做正确，再优化等待时间。

## Fence 到底有什么用

Fence 在这里相当于一条“等这一批做完，再提交下一批”的指令。它主要防止 CPU 把任务提交得太快，导致 GPU 上同时积压太多份权重。

为什么已经有 Release（释放指令），还需要等待？因为 CPU 走到 Release 时，GPU 可能还没有算完：

1. CPU 提交“用 w0 计算”的任务，很快就往下走。
2. CPU 遇到 `Release(w0)`，表示后续计划不会再用 w0。
3. GPU 此时可能仍在使用 w0。为了保证正确，现有 API 会继续保留这份数据。
4. 如果 CPU 随即又上传很多份新权重，旧的尚未真正释放，新的又占了显存，还是可能爆内存。

所以，**“后面不再用了”和“GPU 已经用完了”是两件事**。Release 表达前者，Fence 等待后者，并让已完成任务占着的旧内存可以回收、复用。

例如只允许少量权重同时占用显存时，计划可以安排成：

```text
读取、编码、上传 w0 / w1
用 w0 / w1 完成这一批计算
Release(w0 / w1)       ← 后续不再用这两份权重
Fence                 ← 等计算、上传完成，清理旧引用
读取、编码、上传 w2 / w3
继续下一批计算
```

没有这个边界，CPU 可能已经提交到 w100，而 GPU 还在处理 w0。分批后，CPU 必须先等当前这批完成，才能继续提交下一批。

现有 CUDA 事件已经负责“上传完成后，计算才能读取这份权重”。**Fence 额外负责限制提前提交多少任务**，两者用途不同。

Fence 不会自动释放后面还要用的密文或输出，也不会改变计算结果。跨批次还需要的数据必须保留。内存池也可能保留空闲块供后面复用，所以回收对象后，进程显示的显存不一定立刻下降。

这是首版比较简单的办法，会在批次之间损失一些并行机会。以后可以改成：某份数据真正用完、空出足够内存，就放行下一次上传；届时可以减少整批等待。

## 现有代码能做多少，还差什么

**已有上传能力可以复用，但编译器和运行时还没有把它们组合成完整的分批方案。**

| 现在已有的能力 | 还缺少的部分 |
| --- | --- |
| 可以在计算途中执行 Host→GPU Transfer（上传） | 编译器目前通常把权重上传安排在初始化 |
| 异步上传、上传完成事件、计算等待事件 | 没有按内存容量限制连续提交多少任务 |
| 用完后的 Release，以及 GPU 完成前保留数据 | 没有批次结束时等待并回收的 Fence |
| 顺序执行器认识 Encode（编码） | 当前 V1/V2 计划校验不允许在计算途中编码 |
| 可以读取权重数据包 bundle | 当前启动时读入所需原始数据，缺少按需读取和容量限制 |
| 多 GPU worker 可以执行本机上传 | 目前拒绝执行阶段的 Encode 和 Host Compute，需要补 CPU 任务执行路径 |

只把上传推迟还不够。如果仍在启动时编码全部权重，大量编码结果会堆到主机内存里。因此，**读取、编码、上传、释放都要一起分批安排**。

还有一个独立限制：完整计划含 10,835 个在 CPU 上解密再重加密的 Boot 操作。现有多 GPU worker 模式不接受这些 Host Compute，所以当前四卡计划不能直接交给它运行。顺序执行模式有这条 CPU 执行路径，可以作为首版入口；它也能操作多张本机 GPU。

## 建议按这个顺序实现

1. **先验证已有上传路径。** 用两三个权重的小计划，启动时编码，计算途中分次上传。分别验证顺序执行和本机多 GPU worker 的事件等待、数值和释放。这个小实验不需要改计划格式，也不代表大模型内存问题已经解决。
2. **做能控制内存的首版。** 新增 V3 计划，允许计算途中 Encode，加入 Fence；原始数据按需读取。先支持一台机器上的一张或四张 GPU，用顺序执行模式，每批结束后等完成再推进。先不提前上传，再在批内增加提前上传。
3. **补多卡并行执行。** 增加一个 CPU worker，串行处理 Encode 和现有 Host Boot；GPU worker 处理上传和 GPU 计算。每批内部可并行，完成后再进入下一批。
4. **测量后再加速。** 分别测读取、编码、上传、GPU 等待的时间，再决定提前准备多少权重，以及是否减少整批 Fence。

首版只支持一台机器、一个 rank。跨机器 MPI/NCCL 协同另做，不混进这次实现。

---

## 给后续实现者的具体任务

以下保留源码和正确性约束，供接手的 AI 或开发者使用。上面的目标不变：**先让内存受控，再优化上传与计算的重叠。**

### 1. 从这些代码开始

调查日期：2026-10-10。源码基线为 Poseidon `3b911694`、ckks-runtime `641e2ee`、DaCapo `a5bae92`。后续文档提交不改变这组运行时代码。

| 需要改的部分 | 源码入口和关键位置 |
| --- | --- |
| 编码、上传、释放、worker 和批次推进 | [runtime.hpp](../third_party/ckks-runtime/runtime/runtime.hpp)：`load_bundle`、`execute_encode`、`execute_communication`、`compile_parallel_phase`、`execute_device_parallel_phases`、`execute_release` |
| 新版计划读取与校验 | [json_plan_reader.cpp](../third_party/ckks-runtime/runtime/json_plan_reader.cpp)、[verifier.cpp](../third_party/ckks-runtime/runtime/verifier.cpp) |
| 按需读取原始数据 | [plaintext_bundle.cpp](../third_party/ckks-runtime/runtime/plaintext_bundle.cpp)、[plaintext_bundle.hpp](../third_party/ckks-runtime/runtime/plaintext_bundle.hpp)：替换启动时全量填充 `slots_by_content` 的做法 |
| 上传分配、完成事件、异步任务保留的数据 | [poseidon_gpu_api.cpp](../src/poseidon/runtime_api/poseidon_gpu_api.cpp)：`prepare_plaintext_upload`、`communicate_async`、`retain_in_flight`、`collect_completed`、`drain` |
| 实际 CUDA 上传与流之间的等待 | [cuda_local_transfer.cpp](../src/poseidon/runtime_api/communication/cuda_local_transfer.cpp)：`copy_host_to_device_async`、`record_execution_ready` |
| 编译器生成通信和划分初始化 | [MaterializeCommunication.cpp](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/MaterializeCommunication.cpp)、[RuntimePlanUtils.cpp](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/RuntimePlanUtils.cpp) |
| 重新安排 Release、估算、导出 | [PlanRuntimeMemory.cpp](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/PlanRuntimeMemory.cpp)、[EstimateRuntimeMemory.cpp](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EstimateRuntimeMemory.cpp)、[EmitRuntimePlan.cpp](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp) |
| 所有导出流水线注册 | [optimizer.cpp](../third_party/ckks-runtime/third_party/dacapo/tools/optimizer.cpp) |

### 2. 新版计划和 Fence

新增 RuntimePlan V3，继承 V2 的 Release/reuse，允许执行阶段的 Host Encode。保留 V1/V2 的旧限制和默认行为；不支持 V3 的执行器应明确报错。

建议 Fence 没有输出值，JSON 形如：

```json
{"ordinal": 123, "kind": "fence"}
```

Fence 的执行要求：

- 停止提交下一批任务，等待本批 CPU、GPU 和通信任务完成。
- 把通信结果交付给 Runtime，再清理 API 因异步任务保留的旧引用。`Api::drain()` 可复用，但它本身不能代替 Runtime 的结果交付。
- 保留后续批次仍需使用的值、上下文、使用次数和最终输出，不重置 ValueStore。
- 只处理当前未完成的通信组，不在每个 Fence 从头扫描全部历史记录。清理或重用组索引前，确认没有 Pending 值仍引用它。

同步修改 schema、reader、Verifier、打印、dispatch、编译器指令、计划布局和导出。布局目前只允许 Release 无结果，需要接受 Fence。Fence 有执行顺序上的作用，不能被 CSE/DCE 删除或随意移位。首版不把它定义成分布式 barrier。

### 3. 编译器怎样划分每一批

建议新增 `PlanPlaintextStreaming` Pass，所有 RuntimePlan 导出入口使用同一条公共流水线：

```text
既有 scale/boot 优化、Encode CSE、物理 level
→ AssignPlacement → MaterializeCommunication
→ PlanPlaintextStreaming（新增）
→ PlanRuntimeMemory → EstimateRuntimeMemory → EmitRuntimePlan
```

第一版保持模型计算、boot 插入、设备分配和密文通信的顺序，只重新安排 Encode 产生的权重明文。按计算顺序扫描一次，记录每份权重在哪里使用、要传到哪张卡、编码后多大，然后逐步加入当前批：

- 加入下一步计算所需的权重；任何一项内存预算放不下，就结束本批。
- 还有空间时，加入稍后会用的权重，安排在首次使用前编码和上传。预取距离参考耗时和字节数；不能只用固定指令条数，因为 Rotate 与 Boot 耗时差很多。
- 跨很远位置重复使用的权重，允许分批重新编码、上传。使用相同原始数据和编码参数，但生成新的 ValueId/TransferId。常用小系数可以保留，但必须计入预算。
- 拆批后不再运行会把这些 Encode 跨批合并的 CSE。保留现有单位明文优化，以及满槽 ±1、补零前缀掩码、编码 scale 的区别。
- 调度完成后重新规划 Release/reuse，不能沿用原计划的释放位置。
- 如果一个不可拆分的计算连同必需数据仍装不下，明确报告哪张卡、需要多少内存、哪个对象超限。不回退到全量加载或无限缓存。

阶段划分也必须一起改：`isRuntimeInitialization`、`buildRuntimePlanLayout`、通信的初始化标记、导出器的 Encode 分支。目前这些地方会把 Encode 放回初始化。不能只改 JSON 数组，否则分析和实际执行会看到不同的顺序。

避免定义前使用、重复 ID、释放后使用、覆盖尚有用途的输入。当前图约 489 万计算步骤，算法应接近 `O(V+E)`，必要时做索引排序；不要每安排一份权重就重扫全图。

### 4. 内存预算怎么算才可靠

分别限制每卡 GPU 对象、Host 编码结果、上传用的 pinned buffer、原始数据缓存。新增参数可采用 `--runtime-plan-plaintext-schedule=eager|stream` 和相应 `*-budget-bytes`；这些参数目前不存在，默认保持 eager。

首版对每批采用保守估算：

```text
这批的对象上界 = 批开始时仍需保留的对象 + 批内所有新增分配
```

在 Fence 前，不因为遇到 Release 就扣掉可能仍被 GPU 使用的对象。原位复用按实际分配计费。还必须满足：

- GPU 预算同时算密文和明文，并为密钥、参数表、临时工作区、内存池和碎片留出空间。
- Rotate、Rescale、keyswitch 等调用的临时内存可能直到事件完成才释放。多个调用连续提交时，不能只预留“最大单个算子的临时内存”；首版累计计入本批，除非已证明可以安全共用。
- Host 编码结果、pinned 上传副本、GPU 目标对象可能同时存在，要分别计入预算。上传开始前就可能分配目标对象和 pinned buffer，必须先检查容量，再分配。
- 后续如果改为按完成事件放行上传，只有事件完成、所有引用结束、底层内存可复用时，才能归还额度。等待容量时不能持有 ValueStore/通信组/allocator 锁，也不能堵住负责完成旧任务的 worker。

例如 N=65536、40 个 q limb、无 p limb 时，一份 GPU 明文约 10 MiB、Host 明文约 20 MiB、上传副本约 10 MiB。编码后大小按实际 q/p limb 计算；原始数据只有少量元素，也可能需要完整多项式。

缺少实际临时内存上界时，报告只能称为 RNS 对象估算，不能承诺总显存一定受限。实测同时记录对象占用、内存池保留量和 CUDA 空闲量。

### 5. 原始数据和 CPU 任务

启动时只读取 manifest、建立长度索引；执行 Encode 时再读取对应原始数据。初始化和执行阶段的 Encode 都要纳入本 rank 引用检查。读取时完成已有的长度、内容、有限值、槽容量校验；缓存命中不重复读取，不另做一轮全数据包校验。

首版不缓存原始数据，或使用按字节数限制的简单 LRU。原始数据缓存与编码后对象分别管理。同一 content 的 level/scale/NTT/context 不同时，不能直接共享编码结果。减少 vector 拷贝时，保留补零和负零的原有语义。

多卡模式先用一个串行 CPU worker 执行 Encode 和 Host Boot，因为两者共用的 `encoder_` 不能假设线程安全。复用现有值发布、依赖等待、使用次数和失败通知。Host Release 也要正确路由；不要统一塞到 GPU worker 0。一批出错时，必须唤醒其他等待线程，不能挂住。

### 6. 提前上传是否真的加速，要实测

当前 `copy_host_to_device_async` 会让上传流等待目标计算流上的 `destination_ready` 事件。如果上传在长 kernel 已经入队后才发起，上传也可能排在它后面，未必重叠。

先尝试在本批内把后续 Transfer 提到独立计算之前。不要直接删掉现有等待：它还保护 RMM 异步分配和内存重用的正确顺序。若确需改事件，分别表达“目标内存可写”和“输入数据就绪”。

用 Nsight 或现有 trace 看真实重叠，分别统计读取、Encode、packing、H2D、计算等待。瓶颈也可能在 CPU 编码、文件读取或 Host Boot；调用了异步复制接口不等于性能已经提升。

### 7. 怎样验收

先跑小型真实 CKKS 计划，再跑一层或小词表模型，最后尝试完整 24 层四卡图。至少覆盖：

- 旧 V1/V2 仍拒绝执行阶段 Encode；V3 接受合法计划，拒绝非法 ID、释放后使用和不支持的模式。
- 全量初始化、分批无预取、分批预取的数值及最终 CKKS 元信息一致。覆盖满槽 ±1、补零前缀、共享权重、不同 scale/level、跨批重新编码。
- 上传未完成就提交计算、Release 早于 GPU 完成、同一输入重复使用、GPU0/GPU3 在不同时间使用同一原始数据，都没有提前销毁或重复交付。
- 人为拖慢编码、上传或计算，内存仍受限制；跨批密文与输出仍有效。超限和 worker 出错能明确退出，没有死锁。
- 至少 20,000 个操作的规模测试；大图不逐批反复扫描全图，也不累积已完成的 pinned buffer 和任务句柄。

交付计划估算与实测峰值、每卡密文/明文/密钥/工作区、Host RSS、pinned 与原始缓存占用，以及上传次数/字节量、重复编码次数和耗时。**“编译成功”“实际装得下”“完整结果通过”分别报告。**

## 复用已有模型和产物

本次模型是合成权重的完整 24 层 Qwen、两个 token 位置、完整 LM head 和 96 个 KV 输出。配置中的 512 是结构上界，本次只编译了位置 0/1。

证据见[完整编译记录](../test-results/qwen24-dacapo-188-20261010/README.md)和[摘要](../test-results/qwen24-dacapo-188-20261010/unit-plaintext/summary.json)。服务器大文件在：

```text
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-native
/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-unit-plaintext/gpu-plans/{gpu1,gpu4}
```

重新编译应复用 `qwen24-native/traced/trace_qwen24.mlirbc`（约 14.82 GB）。诊断文本 MLIR 省略了常量，不能作为编译输入。不要重新生成大模型或另做一次全量哈希检查。

`/tmp` 只放少量脚本和日志；大型模型、字节码、计划继续放在 `/home` 或指定工作目录。不做全量 RNS 磁盘缓存。

交接时按上述顺序交付代码、协议说明、小模型测试和内存/耗时报告。按 DaCapo、ckks-runtime、Poseidon 的顺序提交和更新子模块，并保持本地与 188Server 代码一致。完整四卡是否能跑以实测为准；单卡密文过大的问题需要另外处理。
