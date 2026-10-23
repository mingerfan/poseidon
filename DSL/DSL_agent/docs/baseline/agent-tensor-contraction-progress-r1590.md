# 广义张量收缩接入进度

下一项接入 `torch.tensordot`。它将选定轴上的乘积相加，可统一表示点积、矩阵乘法、多轴收缩及外积。固定 Torch 2.0.1 的[函数实现](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/functional.py)作为接口依据；本机核验该版本提供函数入口，Tensor 方法入口缺失。

## 已完成

Agent 自有数学规划草稿将运算分解为 expand、permute、reshape、matmul 与结果复制；保留轴、广播形状、输出形状和来源绑定。物理密文成本尚待布局与后端检查。

15 个上下文、60 组输入已通过固定 Torch、独立数学解释和分解后计算差分，门限为 `1e-12 + 1e-12*abs(reference)`。覆盖整数 dims、显式／负轴、多轴收缩、标量输出、外积、收缩轴单元素广播及 rank32。另有 14 个拒绝项和 2 个绑定篡改项通过。

首轮标量测试的 NumPy 数据类型问题已修正；随后观察脚本的变量遮蔽错误通过只读报告核验恢复。原日志和源码保留，测试未重复运行。

## 待完成

显式来源版本、公共轴参数绑定、FX 规范化、来源重载和旧接口回归，之后进行 Poseidon 数值验收、程序包复用与真实 Agent 测试。生产入口尚未开放；编译、密态和 API 调用均为 0。

草稿和检查位于 `scripts/baseline/benchmarks/tools/tensor_contraction_component_v1`。[机器可读进度](agent-tensor-contraction-progress-r1590.json)。

上一批[三种 gather 组合](agent-gather-composition-acceptance-r1588.md)已经真实 Agent 首次生成全部通过。后端库、参数、网络、x86 配置和冻结门限保持原样，通用化计划继续。
