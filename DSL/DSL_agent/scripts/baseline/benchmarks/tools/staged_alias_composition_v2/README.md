# 别名与算子组合草案

此目录保存内部 v3 草案及先前 v2 记录。生产 PyTorch 接入仍使用现有版本；本目录用于接线前的数学与资源验证。

## 接口

`targets/pytorch_alias_composition.py` 提供 `plan(source, approximations, resource_profile=...)` 和 `verify(...)`。source 为可信捕获层准备的有序数据图，格式为 `poseidon-ordered-tensor-source-v3`；它是内部规范化表示，尚未登记为应用提交格式。

别名节点登记 view、reshape、transpose、slice、clone、contiguous 和公开原地算术等操作。值节点以 registered 指定既有算子名、输入和属性，复用 operator_graph／model_decomposition 的类型、形状和分解规则。矩阵标量结果沿已有规则使用一元素 carrier。公开权重保持不可修改。

输出包含原请求、数学图、分解图、存储修改事件、节点来源、资源身份及完整性绑定。后续布局、编译、密态和 Agent 状态分别记录。

## 已检查的组合

39 个上下文覆盖 Linear、Conv1d/2d、矩阵乘、归约、张量连接与声明近似的 LayerNorm/GroupNorm，分别组合整体修改、部分视图写回和复制。矩阵例包括公开左／右权重、向量、批量及 dot。

r1137 为当前接受记录：21 tests passed，624 组明文、7600 值，最大差异 4.1633e-17。独立有状态 NumPy 计算与现有高层／分解参考比较。初次测试错误、旧源码和报告保留。

## 接线要求

可信 FX worker 需要把别名操作和既有登记调用转换为有序表示，同时保存权重来源、dtype、输出结构、节点来源及新版本身份。模型种类或 fixture 名称不参与分派。现有 fixture 专用 FX 映射不能直接替换生产捕获器。

原函数与规范化函数的参考验证各使用独立输入副本，公开状态保持检查继续执行。完成隔离捕获后再做布局、真实编译、Poseidon 数值及真实 Agent 验收。

当前只完成输入层草案和明文差分；未运行新 Torch 捕获、frontend、后端或 API。正常应用路径不依赖本目录报告或实验编号。
