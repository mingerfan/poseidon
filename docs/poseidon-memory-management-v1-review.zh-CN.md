# Poseidon 内存管理整体实现与测试记录

2026-10-08。实施方案的第 1–7 步均已实现并验证，供整体审阅。此次修改覆盖
外层 Poseidon、嵌套 Runtime 和嵌套 dacapo 编译器，按功能分别提交。

## 修改了什么

1. **Runtime 执行释放和原位。** Release 等尚未提交的使用归零后清掉引用，
   仍保留值表项；异步工作由 API 持有到完成。原位只覆盖内部计算的单次使用
   输入 0，旧 ValueId 随即失效。不支持的后端在执行前报错。
2. **CPU/GPU 直接使用原来的密文存储。** CPU Negate、Rotate；GPU AddCP、
   SubCP、Rotate。连续三次操作检查真实多项式/设备分配地址保持不变。
   GPU 标准 Rotate 使用一份单分量排列暂存，密钥切换直接累加回原密文。
   预旋转密钥分支也写回原分配。工作区按完成事件保留，组合旋转复用工作区。
3. **修正一次并发清理等待。** CUDA/NCCL 请求正在被另一线程等待时，清理
   函数尝试取锁，取不到就留到下次处理。跨线程门控测试确认清理不会等待
   未完成工作，并且消费者使用本次原位操作的新完成事件。
4. **编译器自动插入 Release/reuse。** 统一初始化/执行顺序和 ValueId；
   保护外部输入、通信副本、多次使用及最终输出。`dist.release` 没有结果并
   声明 Free 副作用；重复运行内存规划 Pass 报错。导出器支持 V2，默认 V1
   的旧管线回归通过。
5. **编译器内直接估算。** 直接分析 CKKS/Dist IR 和 OperatorSpec，没有调用
   Python 原型或先转 JSON。普通输出与输入同时记入，传输另计目标副本，
   原位只计同一块，Release 扣除内部对象，初始化保留量带入执行阶段。
   外部输入按调用方一直持有计入。报告按 rank/Host/GPU 分开，包括闲置设备。
6. **可复用的测试入口。** 新增编译器生成短计划的脚本；MPI 执行工具支持
   执行模式、逐轮解密检查、可选数值预期和 RMM 活跃字节统计。计数器及其
   内存池会活到 API 缓存销毁之后。

审阅代码可按下面顺序看：

- [Runtime](../third_party/ckks-runtime/runtime/runtime.hpp)：`compute_output`、
  `execute_compute`、并行计算，以及 Release/提交计数。
- [CPU 原位](../src/poseidon/runtime_api/poseidon_cpu_api.cpp)、
  [GPU 原位与事件](../src/poseidon/runtime_api/poseidon_gpu_api.cpp)、
  [GPU Rotate](../src/poseidon/gpu/gpu_evaluator.cpp)、
  [密钥切换工作区](../src/poseidon/gpu/gpu_keyswitch_handler.cpp)。
- [共用顺序/编号](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/RuntimePlanUtils.cpp)、
  [内存规划](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/PlanRuntimeMemory.cpp)、
  [估算器](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EstimateRuntimeMemory.cpp)、
  [导出器](../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp)。
- [手算用例](../third_party/ckks-runtime/third_party/dacapo/test/runtime-plan/check-memory.py)、
  [GPU 专项测试](../src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp)。

## 如何开启

模型生成命令加入 `--runtime-plan-memory --runtime-plan-memory-report`。
只要 Release 时再加 `--runtime-plan-memory-no-reuse`；直接使用 hecate-opt
时对应 `--runtime-plan-memory-reuse=false`。

输出同名前缀的 `.runtime-plan.json`、`.memory.before.json`、`.memory.json`。
两份内存报告分别分析实际规划前后的 IR。峰值位置和最大对象的编号与计划
共用；JSON 保存字节数，终端显示 MiB。独立的 `plan-runtime-memory` 和
`estimate-runtime-memory` Pass 也可直接使用。

规划后不再运行会合并、删除或重排计算的优化。GPU SubCP 通过手写计划测试，
没有新增 MLIR 数学算子。原有 Python 内存估算文件未作修改。

## 测试结果

- 本地和 188Server 的 Runtime：9/9 CTest 目标，包括 MPI 两、四进程。
- 编译器：2/2 CTest 目标，覆盖原有导出、设备分配、通信、V1，以及手算的
  普通/释放/原位、跨 rank、跨阶段、实际 level、最终输出、多次使用、编号
  一致性、重复规划、无效 OperatorSpec 报错。
- CPU API：6/6 项；GPU API：本地 16 项、远端 17 项，远端包含双卡。
  原位测试包括三次连续操作、正负/组合/零旋转、预旋转密钥、三分量 CPU
  Negate、调用方输入不变、异步事件、延迟释放和多轮回收。
- 188Server：4 张 V100，单卡、四卡单进程、两进程各两卡。NCCL 2×2
  初始化和实际传输通过。编译器短计划的三种配置在串行和工作线程模式各跑
  预热 1 轮、检查 10 轮，按已知结果检查误差 ≤1e-4。
- 现有 degree 8192 无 Boot MLP：三种拓扑 × 三种配置，都预热 1 轮并检查
  10 轮。每张卡每轮的活跃分配返回同一基线，净变化均为 0；重复解密结果
  一致。普通与优化计划的全部 4096 个解密槽对比，最大差异 1.64e-6，
  检查阈值 1e-4。
- 单卡/四卡另用现有 MLP 输入与 Python/Mock 结果核对，沿用原测试阈值
  `0.1 + 0.006 × |参考值|`。三种配置都通过。它们与参考值的最大绝对差异
  约 0.525，普通计划也有相同差异；本次只确认内存改动没有扩大这个偏差。
- 模型生成脚本与 CLI 的内存开关、规划前后报告实际运行通过。

远端共 36 个执行检查组通过，其中 33 个输出 JSON 报告；最后补跑 CPU/GPU
API 和 Runtime 检查也通过。原始记录和汇总位于
[测试目录](../test-results/memory-v1-review/)；
[汇总 JSON](../test-results/memory-v1-review/summary.json) 保存逐拓扑比较及内存数值，
[远端执行脚本](../test-results/memory-v1-review/run-188.sh) 可复现主要测试。

2026-10-10 提交前，本地重新构建并通过 Runtime 9 项、编译器 2 项、CPU/MPI
和单卡 GPU API 回归，以及 Python 估算器 9 项测试。GPU 原位测试改用累计
分配字节数检查 AddCP/SubCP 不分配存储：完成清理会降低活跃字节数，原先
要求活跃量前后相等会误报。密文地址不变和多轮活跃量回到基线的检查继续保留。
重新生成的单卡编译器短计划还在串行和工作线程模式下各完成 1 次预热和
3 轮数值/显存检查；最大误差不超过 `1.03e-6`，每轮活跃分配净变化均为 0。

## 估算和实测怎么对照

MLP 的计划对象峰值如下，单位 MiB。2×2 使用四卡计划的同一计算分配，
将设备 0–3 映射到两个 rank；没有重新调度。

| 拓扑 / GPU | 普通 | 仅 Release | Release + reuse |
| --- | ---: | ---: | ---: |
| 单卡 0 | 227.06 | 88.34 | 88.34 |
| 四卡 0 | 96.22 | 16.06 | 16.06 |
| 四卡 1 | 82.78 | 14.31 | 14.31 |
| 四卡 2 | 79.88 | 13.66 | 13.66 |
| 四卡 3 | 82.81 | 18.16 | 18.16 |

此模型只有两处可原位 AddCP；Rotate 的输入还有其他使用，无法覆盖。
原位减少累计分配量，但没有降低这个模型的对象峰值。

RMM 记录的是执行期间活跃分配，扣除预热后的固定基线再观察峰值增量。
优化计划单卡的峰值增量约 177.78 MiB；四卡为 50.84、46.84、47.31、46.75
MiB；2×2 为 53.09、47.97、50.41、52.94 MiB。缓存的密钥/参数另在基线中，
例如远端单卡基线约 271.38 MiB。内存池保留但未使用的空间不计入活跃分配。

实测增量高于报告是允许的：报告假设每条指令完成后再执行下一条，只计
明文/密文对象；GPU 实际异步执行，还需要 Rotate 工作区和通信暂存。
仅 Release 与加 reuse 的实测峰值也会随并发时序浮动，不能把一次测得的
差异当作原位的性能或峰值保证。各卡峰值没有相加。

## 验证范围

本次多卡在同一台机器上，服务器同时有其他 GPU 工作，时间仅供检查过程
参考。没有验证跨节点 NCCL、完整 native Boot 模型或性能收益。原位 Rotate
的工作区不会被编译器报告忽略成“零成本”：它明确列在尚未计入项中，实机
RMM 统计则会记录这些活跃分配。
