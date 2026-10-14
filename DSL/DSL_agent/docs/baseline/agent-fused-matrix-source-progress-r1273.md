# 融合矩阵来源接入进展（r1273）

可信 PyTorch 草案新增 addmm、baddbmm、addbmm 来源接入，来源包重载和
统一提交接口已经完成离线检查。当前生产入口保持原契约；V14 尚在独立草案中。

## 接入范围

| 操作 | 形状与计算 |
|---|---|
| addmm | 二维矩阵相乘，加可广播偏置 |
| baddbmm | 相同批数的三维矩阵相乘，加可广播偏置 |
| addbmm | 批量矩阵相乘、沿批轴求和，再加偏置 |

支持函数、Tensor 方法、关键字写法，公开 alpha/beta 与公开矩阵权重。
输入采用静态 float64；有限公开系数绝对值至多 1,024。
来源保持原始 FX、规范化步骤、输入输出结构及权重身份。
重载时通过数据记录校验拓扑、参数、类型、形状和后续来源绑定。

## 验收结果

| 范围 | 结果 |
|---|---|
| 新的双文件来源模型 | 9 例，真实隔离捕获；144 组／3,488 个值；最大差异 2.22e-16 |
| 融合写法明文回归 | 41 个上下文；656 组／16,096 个值；最大差异 4.44e-16 |
| 旧 V12/V13 来源 | 2 例；32 组／368 个值；旧契约仍拒绝新增融合调用 |
| 来源包重载 | 3 份通过，属于来源包校验 |
| 统一提交／取消 | 3 项通过，使用注入的后端身份；生成与执行为 0 |
| 数据记录校验 | 9 个正例、45 个拒绝例；标准 Python 校验，无模型或数值库导入 |

各组单列，存在重叠，不能相加作为新增模型总量。
来源和明文检查使用 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1 CPU，
数据记录检查使用系统 Python 3.10.12。

同一适配代码下，44 个当前测试方法有通过记录。
r1268 的文件名夹具错误和 r1269 的错误信息断言已分别补验；
r1268、r1269 原始整体状态仍为 failed，r1270 为 passed。
本结果是合并补验证据，并非单轮 44 项全绿。错误及 skip 均为 0。
明文双参考保持原检查门限；密态验收门限仍为 1e-5 + 1e-4*abs(reference)。

## 边界与后续

本轮编译、Poseidon 密态执行、解密、真实 Agent 生成均为 0。
物理布局与编译程序包的导出、重载、重复密态执行尚未验证。
已有 704 个生产文件、648 个后端文件、6 个 reference 文件及 x86 配置保持。
改动只在 ARM Ubuntu 的独立 Agent 草案、测试和文档中完成，
未提交或推送。

r1263 的三例 common-call 离线续验与 r1232 的三例付费验收仍待各自批准，
两份冻结绑定未改变。先按原范围补齐 Poseidon 回归，再联合接入来源契约，
分别验收物理布局、编译、密态执行与程序包。

[草案说明](../../scripts/baseline/benchmarks/tools/fused_matrix_source_draft_v1/README.md)
· [机器可读记录](agent-fused-matrix-source-progress-r1273.json)
· [合并验收](../../scripts/baseline/benchmarks/tools/fused_matrix_source_draft_v1/acceptance.json)

原始证据：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-fused-matrix-intake-r1268`、
`r1269`、`r1270`；数据校验 `agent-fused-matrix-host-checks-r1272`，
汇总 `agent-fused-matrix-source-audit-r1273`。

汇总绑定：`75fad1285ce2cfab6eda17fd561029b5c9bfaf24f5880feae9e09ac55abd8a35`。
