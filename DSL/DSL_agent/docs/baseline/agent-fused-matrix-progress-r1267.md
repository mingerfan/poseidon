# PyTorch 融合矩阵调用进度（r1267）

Ubuntu 连接正常，当前没有运行批次。分支为 lhy-agent-dsl，HEAD 为
7588e48093631eaf8385506e3db058e7ca3cfe7e。

## 已确认

Agent 草案增加 addmm、baddbmm、addbmm 的函数、Tensor 方法和关键字写法，
复用既有矩阵分解。原地更新顺序、公开权重、广播及来源绑定继续保留。

| 检查 | 结果 |
|---|---|
| 控制、拒绝和兼容测试 | 26 passed；0 failed/errors/skipped |
| addmm 明文检查上下文 | 14 |
| baddbmm 明文检查上下文 | 14 |
| addbmm 明文检查上下文 | 13 |
| 输入与输出 | 656 组明文输入，16,096 个输出值 |
| 原函数与规范化 / 分解最大差异 | 4.44e-16 / 4.44e-16 |

检查上下文包含不同调用和组合，单列计分，保留原 Benchmark 数量。
验证使用固定 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1 CPU，
在独立网络隔离沙箱内完成。结果及源码快照位于：
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-fused-matrix-source-r1266。

首次 r1265 的一项诊断测试失败已经保留；复验修正错误信息，并增加签名及输出
元数据绑定检查。旧通过记录和源码绑定保持原样。

## 当前边界

新增实现位于 scripts/baseline/benchmarks/tools/fused_matrix_intake_draft_v1。
当前为内部可信捕获草案；应用来源格式与生成契约保持现有版本。
这批编译、密态执行、Provider/API 调用均为 0。

704 个生产文件、8 个前置 common-call 草案文件、648 个后端文件和
6 个 reference/产物门限文件核验一致。x86 配置、现有修改和历史产物保留。
本轮新增 Agent 草案、开发测试及文档，未进行安装、下载、参数变更或 Git 提交推送。

## 推断与待验证

两套独立明文参考和原始 Torch 计算一致，支持将这些写法接入统一图；
实际 Poseidon 编译、密态数值和 Agent 生成仍分别待验收。

## 下一步

已冻结的 r1263 三例离线修复验收与 r1232 三例真实 Agent 提案保持原绑定，
仍等待其对应批准。先完成 common-call 代表性 Poseidon 回归，再在明确契约边界
联合接入，并为融合矩阵新增分区安排真实后端与 Agent 验收。

[机器记录](agent-fused-matrix-progress-r1267.json)。
