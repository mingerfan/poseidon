# MHA 输入适配草案验收

当前完成固定 Torch 2.0.1 MultiheadAttention 的 Agent 输入草案及明文数学验收。生产入口仍为 v11，本批结果仅用于后续输入接线。

## 实现

公开模块配置、投影权重和调用参数规范化为数据请求，生成带来源绑定的数学图与分解图。复用已有 Linear、matmul、张量变形、广播、归约、exp 和 reciprocal 注册规则。

保留 self/cross attention、独立 K/V 特征宽度、attention weights 和静态 None 输出。布尔 MHA mask 中 True 表示阻断；float mask 的负无穷转换为精确阻断标记。bias/zero token 的分母、概率和输出贡献按固定源码处理。因果提示要求实际的规范因果 mask。

新增检查覆盖调用签名、shape、别名、公开权重、mask、近似配置、资源边界、内容篡改和结构输出。区分性测试分别改变 mask、额外 token 或公开权重，确认输出及绑定发生相应变化；原验收模型保持。

## 结果

| 检查 | 结果 |
|---|---:|
| 测试方法 | 25 passed，0 failed/errors/skipped |
| 已知本地 FX 上下文 | 12/12 |
| 明文输入组 | 192 |
| 逐项输出值 | 1,920 |
| 独立原函数参考与固定 Torch 最大差异 | 5.551115123125783e-17 |
| 独立近似参考与分解最大差异 | 6.661338147750939e-16 |
| 本批小输入下近似与原函数最大差异 | 4.384270724244743e-13 |

12 个上下文包括无 batch 自注意力、多头自注意力、sequence-first 交叉注意力、独立 K/V 宽度、布尔 mask 与 padding、逐头有限偏置、负无穷 padding、bias token、zero token、两类额外 token 组合及两个因果上下文。

每例采用四类基础输入和十二组固定随机探针。逻辑输入 rank 2–3、N≤2、E≤4、H≤2、L/S≤3；分解节点数 55–68。验收使用显式六阶 Taylor exp（[-0.5,0.5]）和八轮 Goldschmidt reciprocal（[0.5,8]）。这些小范围观察不代表实际应用精度认证或密态深度可执行。

原始 r1061 有 29 个子测试错误（22 个测试方法），共同根因为草案使用 `permute.axes`，已有图契约要求 `dims`。修复仅位于新草案。原失败、源码和日志保留；r1062、r1063 分别绑定各自源码。

当前 r1063 绑定：
`7b4c5f20728222c7817acc0a4bd2bc0aa8bc2c056e858b3229f3af0722a158e2`。

## 状态与边界

**已确认：**当前源码的数学、拒绝、完整性和区分性测试通过。5952 个待批准冻结文件、648 个后端文件、旧 reference 和历史记录均保持；原 Git 状态行全部保留，当前分支为 `lhy-agent-dsl`。现有除法/MaxPool 和 GELU 草案绑定保持。

**基于证据的判断：**现有 Agent 图算子足以表达本批 MHA 小模型数学计算；接线可在 Agent 输入层实施。物理密文容量和编译可行性不能由明文通过推定。

**尚未验证：**生产隔离捕获、请求/程序包接线、物理布局、Hecate tracing、真实编译、Poseidon 执行、解密和真实 Agent 生成。float32 原来源精度与原模型应用指标也仍待认证。

当前登记范围采用原生 eval、CPU float64 状态和 rank 2/3 静态输入；训练、自定义模块、错误 causal 提示、全阻断行、非法/超容量配置返回明确拒绝。一般框架数据精度转换、视图与原地别名将按独立语义处理。

本批安装、下载、参数/门限调整、后端修改、付费调用、编译、密态执行、解密、commit/push/PR 均为 0。冻结 CKKS 门限继续为 `1e-5 + 1e-4*abs(reference)`；本批明文差分使用 `atol=rtol=1e-12`。

## 文件与下一步

代码与测试：`scripts/baseline/benchmarks/tools/static_mha_source_draft_v1/`。原始结果：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-mha-source-draft-r1063/`。

[机器记录](agent-mha-source-draft-acceptance-r1064.json) · [当前后端提案](agent-common-pytorch-offline-proposal-r1042.md)。

r1042 的 151 项 Poseidon 离线验收及 25 份程序包复用仍待当前授权，收据和后端结果目录均不存在。r870 原 24 项已于 r878 关闭。

接下来在安全切换边界，将除法/MaxPool、GELU 和 MHA 接入新显式输入版本，贯通原函数来源、结构化反馈和输出结构；之后对新增或受影响范围执行独立后端及真实 Agent 验收。原地 mutation/视图别名、容量、bootstrap、密文交接、应用精度、ResNet20 和 CUDA 继续独立推进。
