# Agent 资源检查与 GELU 输入进度

2026-10-08，ARM Ubuntu，`lhy-agent-dsl`。

已批准的 r1304 三项真实 Agent 测试全部首过：Linear、近似 LayerNorm、公开权重 split→Linear。
实际 3 次生成、3 次 HTTP，21 组／119 值，最大绝对误差 1.2783e-7。
[付费批次报告](agent-v12-live-rebound-results-r1304.md)保留独立源码和调用绑定。

## 本批改动

数值验收、程序包执行和 CLI replay 接入同一个 RuntimePlan 存储下界检查。
按固定 Runtime 的保留行为计算密文／明文系数和常量槽位内存；
确定超过原 4 GiB 沙箱上限时返回结构化资源失败。修复使用原预算。
预检文件变化、删除和符号链接替换归为完整性异常。峰值内存、评估密钥和工作区仍为未知项。

可信 PyTorch v15 支持 nn.GELU 和 F.gelu 的 `none`／`tanh` 来源变体，
保持对应原函数参考与显式多项式计算参考。六例隔离捕获通过，96 组／576 值。
三种 tanh 写法完成规则程序的编译、密态与程序包验收；新接口真实 Agent 生成待单独测试。

## 验收结果

| 范围 | 结果 |
|---|---|
| 控制与旧接口兼容 | 164 通过，0 失败／错误／跳过 |
| 历史真实 Agent Linear＋三个 GELU 规则程序 | 4 通过；28 组／161 值，最大绝对误差 4.87e-9 |
| 完整性异常修正后的 GELU tuple 输出回归 | 1 通过；7 组／56 值，最大绝对误差 8.22e-10 |
| MHA v16 来源草案 | 24 控制、4 明文上下文通过；64 组／768 值 |

四例联合结果与末项补验使用各自源码身份；末项是重复模型，独立统计。
MHA 草案覆盖多头 self attention、不同 K/V 维度、布尔／padding mask，以及 `(tensor, None)` 输出。
生产来源接入、沙箱捕获、DSL 编译和密态待验收。
[草案说明](../../scripts/baseline/benchmarks/tools/static_mha_source_v16_draft_v1/README.md)。

初次资源哈希／存储解码错误、执行 metadata 不一致及开发运行器的参考 API 类型错误均保留。
metadata 修正后，四例完成编译、密态、新输入执行；新输入参考改用规定的 float64 数组 API，
以独立后审计补齐已有输出比较，原失败报告未改写。
共 9 次编译、14 个密态阶段，包含重复补验，4 个不同模型；[机器报告](agent-resource-gelu-progress-r1305.json)记录来源与绑定。

## 后续

接入 MHA 的显式 v16 来源、输出和参考契约，再做沙箱捕获及代表性 Poseidon 验收。
GELU 三例真实 Agent 测试已准备[冻结提案](agent-gelu-live-proposal-r1306.json)，等待新的有界授权。

本批后续工作 API 为 0。Poseidon、CKKS Runtime、Dacapo、参数、reference 与门限保持；
654 个保护文件核验一致。x86 配置保留，实际执行为 ARM CPU。
旧 76 项真实 Agent 的 56／20 记录保持，bootstrap、CUDA、容量和整模型目标继续按原计划推进。
安装、下载、commit、push、PR 均为 0。

数据门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
