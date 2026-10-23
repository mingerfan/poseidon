# 高层算子版本参考回归

JSON v2 的兼容检查已扩展到当前登记的 34 个高层算子。110 个不同结构的上下文全部通过，每个算子至少三个上下文；共比较 1,760 组固定输入，最大分解误差约 4.44e-16。

本批属于**明文参考与分解回归**。没有进行 tracing、编译、密态执行或真实 Agent 生成。

## 范围

| 内容 | 结果 |
|---|---|
| 既有公开来源图 | 65 个，原文件保持 |
| 补充参考上下文 | 45 个，由开发工具确定性构造 |
| V1／V2、原图／分解图、Torch 比较 | 110/110 |
| 每个登记高层算子的不同结构上下文 | 至少 3 个 |
| 独立输入组 | 1,760；不同实现对同一输入的比较不重复计数 |
| 审计工具拒绝与完整性测试 | 7 passed，0 failed/errors/skipped |
| 新增 Benchmark 发布模型、密态及 Agent 通过 | 0 |

范围包括矩阵、张量组合、池化／归约、归一化、RoPE、eval Dropout，以及显式近似的激活、倒数、开方和 Softmax 类计算。非多项式函数比较其声明的近似计算；原函数误差另列，应用精度保持未认证。

原图、公开常量、输入和命名输出保持。非矩阵算子的基础图逐字一致；涉及矩阵的图允许按新规则规范化。已有 1,200 模型集保持，补充上下文不加入其规模。

## 复现

使用当前固定 Python 环境运行：

~~~bash
PYTHONPATH=scripts/baseline python3 -B \
  scripts/baseline/benchmarks/tools/audit_operator_graph_versions.py \
  --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/operator-version-audit-next" \
  --frozen-proposal docs/baseline/agent-json-matrix-offline-proposal-r914.json
~~~

审计绑定来源图、工具、运行源码，以及实际加载的 Python、NumPy 和 Torch CPU 实现。结果写入新目录；历史报告保持。后续源码更新时，使用当时的新提案身份。

## 接下来的工作

- 先执行待批准的 48 项 Poseidon CPU 验收，确认新入口与原失败案例的实际编译和数值结果。
- 将登记算子的 rank、轴、广播及参数范围整理成明确的能力描述；当前 34 项的 `input_contract` 仍是通用文字，实际规则负责检查。
- 补充常见 PyTorch 复合 API。当前直接捕获没有 `nn.MultiheadAttention` 和 `F.scaled_dot_product_attention` 映射；已有 Attention 组合图单独计分。
- 分组卷积的接入与形状规则已存在，不能只根据算子名称把它列为未实现；真实后端范围仍按具体回归记录判断。
- 近似的应用精度、更大物理布局、bootstrap、密文分区和 CUDA 分别继续推进。当前参考通过不能替代这些验收。

48 项提案的 676 份运行源码、786 份案例文件和4个驱动保持原绑定 `3c3168bd8919638722c688f3fe1fc90b6ea6c0cfd585c7ac68f6b41ae5df69c5`，尚未启动。

[机器报告](agent-operator-version-reference-acceptance-r923.json) · [JSON v2 用法](operator-decomposition-and-python.md#json-矩阵计算图) · [待批准的后端验收](agent-json-matrix-offline-proposal-r914.md)。
