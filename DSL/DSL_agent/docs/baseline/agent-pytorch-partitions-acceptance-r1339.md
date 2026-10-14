# PyTorch 静态 chunk/unbind 接入与验收 r1339

可信 PyTorch 输入新增 `poseidon-trusted-pytorch-v17`，支持函数与 Tensor 方法的
chunk/unbind，并沿用工厂、公开权重、近似配置、有序视图和输出接口。

开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。
仅修改 Agent 捕获、来源契约、能力登记、测试与文档。
Poseidon、CKKS Runtime、Dacapo、数学 reference、参数和误差门限保持。

## 输入与实现

chunk 的分块数按真实轴长度计算；尾块可短，实际返回数可能少于请求数。
支持 rank 1–32 的静态 chunk，以及 rank 2–32 的静态 unbind；
各配置和算子仍受现有元素、轴和节点上限约束。
rank 1 unbind 的标量别名视图保留为未开放能力。

可信捕获将拆分端口规范化到原有静态索引。
原图、规范化图、端口形状、共享视图与实现哈希均绑定；
修改共享视图保持原顺序，clone 快照保持独立。
旧来源 v1–16、JSON、CLI 和预算接口保持。
[用法与清单](operator-decomposition-and-python.md#pytorch-v17-静态-chunk-与-unbind)。

## 验收

| 范围 | passed | failed / errors / skipped |
|---|---:|---:|
| 新规划、来源、传输完整性与独立参考 | 26 | 0 / 0 / 0 |
| 旧输入、能力、MHA、GELU、静态捕获与算术兼容 | 94 | 0 / 0 / 0 |
| 六模型 Poseidon CPU 编译、数值和程序包新输入 | 6 | 0 / 0 / 0 |

chunk 的长度/数量结果额外与锁定 Torch 2.0.1 的 102 个形状组合对照；
unbind 检查六个轴上下文。高 rank 边界来自静态测试，密态测试使用下表的小型形状。

| 模型 | 输入 | 检查内容 |
|---|---|---|
| chunk_branches | [2,5] | 不等长末块、分支、命名输出 |
| chunk_projection | [2,2] | 固定 Linear 后拆分与融合 |
| chunk_public_weight | [1,2] | 非重复公开权重拆分后 Linear |
| unbind_fusion | [2,2] | 方法调用、分量算术 |
| unbind_mutation | [2,2] | 共享视图写入与修改前快照 |
| unbind_sequence | [2,2,2] | 函数关键字、负轴、transpose 和 tuple 输出 |

来源完成隔离捕获与独立参考检查。
每模型真实 Hecate tracing、Earth/CKKS 编译、RuntimePlan、OperatorSpec、
plaintext bundle 和输出绑定检查后执行 Poseidon CPU 密态计算；
程序包重载后再执行四组新输入，复用时生成与编译调用均为 0。

合计 **6 次编译、12 次密态调用、48 组输入、224 个输出值**。
最大绝对误差 **5.074545579852874e-7**，
逐项门限仍为 `1e-5 + 1e-4*abs(reference)`。
这批是规则基线，真实 Agent 新生成通过数为 0。

SDK 身份：`e3cdaa8c413fed4dad3920ec7df7b37765af8362b7b1e0becacefa397e9a2760`。
旧身份与历史绑定保留；后端/数学参考 654 项和两份历史请求字节核验一致。
本机验证为 ARM CPU；x86 配置保持。

## 保留的失败与后续

首轮 26 个测试出现 1 个断言失败、5 个测试数据/异常预期错误；
修正测试后按同一实现重验通过，首轮记录保留。
最初验收工具在 prepare_source 前读取 task.json，六个排队任务没有完成来源捕获。
原报告及计数纠正记录保留，随后经现有接口继续相同任务和原截止时间，
没有重置额度或重复密态执行。

新能力的真实 Agent 生成和修复仍需冻结当前请求后单独批准。
剩余两项 MHA 也需新身份绑定；历史中断 HTTP 保持未知，不自动重试。
规模、应用精度、密文交接、bootstrap、CUDA 与整网络目标继续，
本次六模型通过只覆盖表列上下文。

本批 API、安装、下载、后端和参数修改、commit/push 均为 0。
原生验收约 406.51 秒，新产物约 49.93 MiB；
全局产物约 25.80 GiB，磁盘余量约 42.58 GiB。

[机器报告](agent-pytorch-partitions-acceptance-r1339.json)。
