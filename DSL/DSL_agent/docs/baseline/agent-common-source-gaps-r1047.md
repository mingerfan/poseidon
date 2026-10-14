# PyTorch 常见写法兼容性

本批使用当前生产 v11 入口，实际检查六种静态 PyTorch 写法。检查仅到隔离捕获和明文参考，未生成 DSL，也未运行编译器或密态后端。

## 检查结果

| 写法 | 当前结果 | 失败位置或验证范围 |
|---|---|---|
| `nn.MultiheadAttention(4, 2, batch_first=True)` | 输入拒绝 | 模块调用按单个 `input` 解析，三个 Q/K/V 参数触发 `Module input arguments` |
| `F.max_pool2d` | 输入拒绝 | functional 的 boolean-dispatch 目标没有映射 |
| `nn.GELU(approximate="tanh")` | 输入拒绝 | 当前原函数语义只接受 `approximate="none"` |
| `nn.ReLU(inplace=True)` | 输入拒绝 | 原地激活及别名语义尚未接入 |
| `transpose → contiguous → view` | 接入通过 | 16 组独立明文参考，原模块、规范化和分解差异均为零 |
| `torch.div(x, 2.0)` | 输入拒绝 | 公开量除法没有映射 |

五项拒绝均在输入层发生，尚无编译或后端结果。它们不能作为对应数学计算无法由 DSL 表达的证据。`contiguous` 的已登记 C-order 范围已有实际证据，初始的“缺少映射”假设被排除。

## 下一批实现范围

1. functional MaxPool 和公开量除法：补静态参数映射与形状规则，复用现有 MaxPool 分解及加乘基础图；保留原函数参考。除法检查零、非有限公开量和常量预算，加密分母需显式倒数近似。
2. MultiheadAttention：增加模块专用参数解析和公开投影权重绑定，组合已有 Linear、矩阵乘、SDPA 与输出投影；先登记 batch、head、Q/K/V、mask 和返回结构范围。不能将裸 SDPA 的通过直接作为完整 MHA 的通过。
3. GELU tanh 变体：保留来源中的原函数版本，增加对应独立参考和显式近似分区；不能把它默认为 `approximate="none"`。
4. 原地激活：先分析值别名、共享消费者及修改顺序，只有证明等价的情况才能转成纯图。直接清除 `inplace` 标志不满足语义要求。

每个新增可执行分区仍需三个不同上下文及组合回归，随后分别验收真实后端和获准的真实 Agent 生成。本报告没有宣称这五项已经实现。

## 与当前验收的关系

151 项离线提案的模型、DSL、691 个源码、5 个上下文文件、5248 个案例/证据及8个新旧驱动保持原绑定；后端、reference、历史记录及现有修改均保持。新探针和结果独立保存，没有追加进该提案。

本批仅新增开发工具、公共小型来源探针及报告，未修改生产接入代码。冻结的 1,200 模型集保持，六种写法未记为六个新增 Benchmark 模型。

旧 `operator-scope-v1.json` 的 pending 分类不能直接用于当前状态：v11 的部分 SDPA 和 public mask 已完成接入层检查；MHA 和有状态缓存接入仍是独立工作。输入、分解、编译、密态和 Agent 状态需要分别登记。

当前 151 项仍未批准或启动；没有批准收据及新后端目录。本批 API、安装、下载、Hecate tracing、编译、密态执行、解密、提交和推送均为 0，数值门限保持。

[机器可读记录](agent-common-source-gaps-r1047.json) · [v11 接入结果](agent-pytorch-common-v11-acceptance-r1045.md) · [当前离线提案](agent-common-pytorch-offline-proposal-r1042.md)
