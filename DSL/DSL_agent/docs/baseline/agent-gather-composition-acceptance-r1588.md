# gather 与模型计算组合验收

三份 PyTorch v37 模型均由真实 Agent 首次生成成功，并通过 Poseidon CPU 数值验收。

| 模型 | 输入 | 输出 | 结果 |
|---|---|---|---|
| gather → Linear ＋残差 | [5] | [4] | 通过 |
| Linear → gather ＋双输入融合 | 两个 [2,3] | [2,3] | 通过 |
| transpose → gather →命名输出与 mean | [2,3,2] | [3,1,2]、[3,1] | 通过 |

## 结果

- 原始 PyTorch 与独立数学参考：3/3，48 组输入。
- 规则程序：3/3，3 次编译、9 次密态调用。
- 真实 Agent：3/3，3 次生成、3 次 HTTP，未使用修复或传输重试；3 次编译、9 次密态调用。
- 真实 Agent 数值检查：24 组输入、152 个输出值，最大绝对误差 **1.920067218e-7**。
- 三份程序包的重载、新输入、重复执行和缓存复用全部通过。
- 失败、skip、未运行和未结算 HTTP 均为 0。

独立审计核对了原响应与程序字节、来源记录、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、输入输出绑定及解密结果。固定门限仍为 `1e-5 + 1e-4*abs(reference)`。

## 配置

来源 `poseidon-trusted-pytorch-v37`，生成指导 `explicit-v19`，固定 w45。SDK `605f3b5e330454bfe8ff6b146d54c878e570d1500264f1f9ebd36dd189227a1b`；ARM Ubuntu / Poseidon CPU。Dacapo、CKKS Runtime、Poseidon、参数和网络均保持原样。

主批次和后置审计已终态，调用账本已结算。旧进度快照与准备工具的导出路径失败记录保留。[机器可读报告](agent-gather-composition-acceptance-r1588.json)。

## 后续

补充常见 PyTorch 张量收缩写法，进入已有张量和矩阵运算链路。随后继续通用组合、容量与布局、应用精度、bootstrap、CUDA 及整模型工作。本批只验证表中三种组合。
