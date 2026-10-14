# PyTorch 按元素 gather 验收

显式选择 `poseidon-trusted-pytorch-v37`，可接收 `torch.gather` 和 `Tensor.gather`：固定公开 int64 索引、重复索引、静态轴及负轴表示。输入与索引 rank 相同；输出形状等于索引形状，非索引轴上的索引尺寸可小于输入。默认来源 v18 和 latest 别名 v34 保持原行为。

Agent 在规范化阶段将 gather 转为现有切片、拼接和 reshape，并保留独立结果存储；Dacapo、CKKS Runtime、Poseidon 和参数均保持原样。

## 验收结果

| 验收 | 结果 | 实际编译 | Poseidon 密态调用 | 生成／HTTP |
|---|---:|---:|---:|---:|
| 三个原始 PyTorch 来源，真实 Agent 生成（r1577） | 3/3，首次生成通过 | 3 | 9 | 3/3 |
| 当前应用能力描述下，原 Agent DSL 重放（r1580） | 1/1 | 1 | 3 | 0/0 付费调用 |
| 应用能力、反馈和旧接口检查 | 77 passed，0 failed/skipped | — | — | 0 |

真实生成覆盖一维重复索引、二维按行不同索引、三维中间负轴。24 组输入、112 个逻辑输出值，最大绝对误差 **5.21765730288e-7**。三份程序包均完成重载、新输入、重复执行和缓存复用。

当前 SDK 的补验使用完全相同的模型源码、权重和历史 Agent DSL。新来源身份重新捕获，旧响应与报告保留；8 组输入、32 个输出值，最大绝对误差 **1.12553953824e-7**。它属于历史真实 Agent 源码重放，未重新调用模型。独立审计均通过，门限保持 `1e-5 + 1e-4*abs(reference)`。

## 配置与范围

- 来源版本：`poseidon-trusted-pytorch-v37`；生成指导 `explicit-v19`。
- 当前组件 SDK：`605f3b5e330454bfe8ff6b146d54c878e570d1500264f1f9ebd36dd189227a1b`。
- r1577 生成 SDK：`900393362406e0d051d1ecca04fe619d89344ea916963012e4549bf797348c68`。
- 执行：ARM Ubuntu，Poseidon CPU，固定 w45；Linux x86_64 配置保留。
- 高 rank 的数学与规划检查另计；本批真实密态范围为 rank 1–3。
- 动态／加密索引、空索引、负索引、稀疏梯度和 out 修改仍按输入契约拒绝。
- 编译、密态和数值检查为有限样本验收；gather 与其他算子的未见组合继续单独验证。

准备阶段的旧身份拒绝及审计工具导入失败保留。修正工具后仅做新绑定和只读审计，未放宽检查或重跑付费生成。

## 后续

继续扩展通用组合、规模与布局、原模型应用精度、bootstrap 和整模型；CUDA 实机验收保持独立。gather 通过属于通用化计划的一项能力。

[机器可读报告](agent-tensor-gather-acceptance-r1582.json)。完整证据位于 Ubuntu checkout 的 `scripts/baseline/benchmarks/tools/tensor_gather_live_audit_v1`、`gather_capability_acceptance_audit_v2` 及 ARM work root 的对应 results 目录。
