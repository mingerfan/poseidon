# 明文分批与提前上传：实现和验证记录

已实现 V3、Fence、按需 bundle 读取、分批编译，以及本机 GPU worker 配合串行 CPU worker 的执行路径。默认仍为 eager，V1/V2 的限制保持不变。首版只支持一个 rank；完整模型的编译、实际显存是否足够、最终数值是否正确，必须分别判断。

代码入口和协议见 [RuntimePlan V3](../third_party/ckks-runtime/docs/runtime-plan/v3/README.md)，测试原始记录见 [188Server 测试目录](../test-results/plaintext-streaming-188/README.md)。

## 编译和执行方式

公共导出流水线在 placement 和通信物化之后运行 `PlanPlaintextStreaming`，随后重新安排 Release/reuse，再做内存估算和导出。只移动、复制 Encode 及其直接上传；密文计算与通信的相对顺序保持不变。跨批重复使用的权重获得新的 ValueId/TransferId。调度之后不能再运行跨批 Encode CSE。

示例参数如下，数值是本次完整四卡实验的配置：

```text
--runtime-plan-plaintext-schedule=stream
--runtime-plan-gpu-budget-bytes=25769803776
--runtime-plan-host-budget-bytes=4294967296
--runtime-plan-pinned-budget-bytes=536870912
--runtime-plan-raw-budget-bytes=1048576
--runtime-plan-workspace-bytes-per-op=16777216
--runtime-plan-prefetch-bytes=268435456
```

GPU 预算是每卡预算，使用前需要另外留出密钥、参数表、内存池等空间。Host 预算同时计算编码明文和 Host 密文。pinned 预算计算一批内 Host/Device 传输暂存；raw 预算计算一份文件缓冲及解码后的 double 数组。首版没有原始数据缓存。

`prefetch-bytes=0` 时，在消费者之前准备权重；正数时，将一批中受字节数限制的一段准备任务提前。尚未实现根据算子实测延迟自动选择预取距离。一个不可拆分计算放不下时，编译器报告位置、ValueId、需要量和预算，不回退到全量加载。

Fence 先交付本批未完成通信的结果，再 drain API，清理任务句柄和异步引用。后续仍需要的密文、输出、上下文和使用次数继续保留。worker 模式先完成并 join 本批队列，再进入下一批；Encode 和 Host Boot 共用一个串行 CPU worker，Host Release 也路由到它。

## 已完成验证

| 验证 | 结果 |
| --- | --- |
| DaCapo 回归 | 8/8；另补显式 Fence 导出 V3、CSE/DCE 不删除 Fence 的回归 |
| ckks-runtime 回归 | 5/5 个测试程序；V1/V2 拒绝执行期 Encode，V3 接受合法计划，检查严格 Fence 字段 |
| 运行时规模 | 两种执行模式各 20,000 次传输、625 个 Fence；检查批内延迟交付、旧对象回收 |
| 真实 GPU API | 19/19；包括 GPU0/GPU3 跨批权重、保留密文、CPU worker Host Boot |
| MLP 数值 | 单卡/四卡、现用现传/预取、顺序/worker 共 8 种组合全部通过 |
| MLP 重复内存测量 | 每种组合预热一次再运行三次，全部回到 GPU 活跃分配基线，pinned 结束占用为 0 |

MLP 每次执行 121 次 Encode、100 次按需 blob 读取、5 个 Fence。与已有 eager CKKS 结果的最大差异为 `1.55e-6`，输入和输出 level/scale 一致。与浮点参考的误差按已有 MLP 容差判断；不能把 CKKS 调度间的微小差异与浮点参考误差混为一谈。

本次四卡 MLP 每卡运行新增分配峰值约 8–13 MiB；预热后常驻基线约 163–229 MiB/卡。单卡新增分配最高约 83.53 MiB，常驻约 271.38 MiB。pinned 实测最高 4.78 MiB。基线合并了密钥、参数缓存等，目前没有将每个类别单独拆开。报告另记录内存池保留量、CUDA 剩余量和进程 RSS；CUDA 剩余量是采样值，不能替代活跃分配峰值。

## 提前上传的测量结论

Nsight 对两种四卡 worker 调度各记录了三次在线运行。每次有 122 次 H2D，共 34,177,024 字节。两种调度均没有观察到同卡 H2D 与 kernel 重叠，不能宣称预取已经加速。现有目标内存就绪等待没有被删除。

一次独立准备阶段 trace 记录了 122 次 packing，CPU 总耗时约 13.10 ms。四卡预取 worker 的一次非 trace 数值测试中，Encode 合计约 330 ms，读取约 23 ms。调用时间在多个 worker 之间可能重叠，不能直接相加当作总运行时间。worker Fence 时间包含本批队列执行和 join，顺序模式则主要是通信收尾与 drain，两者统计口径不同。

## 完整 24 层四卡实验

复用服务器上已有 `qwen24-native/traced/trace_qwen24.mlirbc`，没有重新生成模型，没有把省略常量的诊断 MLIR 当输入，也没有额外扫描全包做内容哈希。产物保留在 `/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-plaintext-streaming`。

完整编译和导出成功，退出码为 0，耗时 1,248.74 秒（20 分 49 秒），进程峰值 RSS 为 93.26 GiB。此次完整实验使用 DaCapo `b60fb1d`；后续 `e473cfb` 补充了整数溢出/槽容量检查和重复编码计数，已通过相应回归，但没有为这些检查重新编译完整模型。

分批报告给出 22,353 批、1,325,109 次 Encode、1,332,419 次明文上传。总明文上传量约 11.56 TB；分批减少同时驻留的权重，不减少全部计算需要搬运的数据。最新编译器报告另外提供 `distinct_encode_definitions` 和 `reencodes`；本次完整编译的旧版报告没有这两个字段。

| 位置 | 批次对象上界加配置的临时空间预留 |
| --- | --- |
| Host | 2.72 GiB |
| GPU0 | 20.25 GiB |
| GPU1 | 24.00 GiB |
| GPU2 | 20.75 GiB |
| GPU3 | 21.64 GiB |
| pinned | 512 MiB |
| raw | 512 KiB |

这份上界在 Fence 前不归还 Release 的额度，并保守地将后来可原位复用的输出也计入分配。它仍然排除了密钥、参数表、未测定的算子临时空间、编码工作区、内存池保留与碎片。`total_device_memory_bound=false` 是有意保留的限制。

完整计划的导出和 GPU 尝试结果在测试目录中分别保存为 `qwen24-compiler-report.json` 与 `qwen24-runtime-smoke-report.json`。执行器被配置为使用两个确定性 one-hot 输入，检查路径支持多个输出；它没有完整模型数值 oracle，不等价于完整模型精度验收。

完整运行尝试在预设 600 秒上限后由测试脚本发送 SIGTERM，实际含退出清理耗时 603.84 秒，退出码 -15，Host 峰值 RSS 15.01 GiB。四卡每两秒采样的显存使用量始终为 0，未生成结果文件；因此**尚未验证完整模型能否装入四卡，也未验证完整数值结果**。原始采样见 `qwen24-runtime-samples.jsonl`。这次是测试时限终止，没有测到 CUDA OOM。

目前启动读取器仍完整加载约 8.76 GB JSON。代码检查确认，共用严格解析入口使用 nlohmann DOM 回调，而该版本的回调解析器在每次结束对象时扫描父容器寻找 discarded 元素；大对象数组存在平方级开销。本次未取得函数级运行采样，不能把全部等待时间归到某个函数，但尚未进入 GPU 的现象与该加载瓶颈一致。计划 I/O 改造属于后续独立工作，不由本次按需权重读取解决。
