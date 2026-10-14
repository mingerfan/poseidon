# PyTorch 常见写法接入检查（r1036）

日期：2026-10-06。开发位置：ARM Ubuntu /home/lhohy/Code/Poseidon，分支 lhy-agent-dsl。

这批工作只检查模型导入。现有生产入口在 18 种常见写法中接入 13 种；独立 Agent 适配副本解决其余五项，18/18 通过隔离捕获和明文参考。修补尚未接入生产入口，也没有编译 DSL 或执行密态计算。

## 检查范围

18 种写法组成六个数学对照组，不计为 18 个新模型，也没有改变冻结的 1,200 模型 Benchmark。

| 数学结构 | 写法 | 生产入口 | 独立副本 |
|---|---|---:|---:|
| Linear → square → Linear | Sequential、ModuleList 循环、functional | 3/3 | 3/3 |
| 深度 Conv2d → BatchNorm → 自适应平均池化 → Linear | module、mixed、functional | 3/3 | 3/3 |
| Linear 与倍数残差相加 | 运算符、torch.add(alpha=2)、Tensor.add(alpha=2) | 1/3 | 3/3 |
| LayerNorm | module、functional、mean/rsqrt 展开 | 3/3 | 3/3 |
| 缩放点积 Attention | torch.softmax、functional softmax、融合 SDPA | 2/3 | 3/3 |
| 推理 Dropout | module、functional、Dropout2d | 1/3 | 3/3 |

原五项都停在 Agent 输入层。两项加法要求 alpha=1；其余三项没有登记对应调用。它们不是新发现的 Dacapo、Runtime 或 Poseidon 执行失败。

## 修补与结果

副本只修改两个 Agent 文件，增加显式记录的 FX 规范化：

- 静态 alpha 加减法分解为乘法与加减法。
- 推理 Dropout 转为恒等映射，检查概率、训练状态、原地修改和空间变体维度。
- 融合 SDPA 分解为转置、矩阵乘、公开缩放、Softmax 与矩阵乘。

本版 SDPA 要求 Q/K/V 为已声明输入，dropout=0，暂不接入掩码和 causal 模式；投影后 Q/K/V 仍是后续缺口。空间 Dropout 同样要求已声明输入的形状。其他调用继续交给现有检查器，不作通配接入。

最终单元测试 9 passed，0 failed/errors/skipped。18 个来源副本通过 288 组参考比较，原模块、规范化模块、数学解释器和 Torch 参考分别核对。最大规范化误差 5.55e-17，最大分解误差 8.67e-17。全部 18 份副本报告被生产身份检查拒绝，不能当作生产入口通过记录。

10 个实际沙箱负例全部拒绝：动态/布尔 alpha、out 修改、训练/原地 Dropout、非法空间维度、causal/随机 SDPA、未知调用、加密数据相关分支。初始审计有一项诊断期望错层：动态 alpha 在 Torch tracing 时提前拒绝；已按原日志复核，没有重跑或改写旧报告。

初次单元测试的 Dropout1d 非法维度、一次副本日志变量冲突，以及原五项失败均保留。后续副本和结果使用独立目录与绑定。

非线性配置只用于接入和分解回归。当前探针下 LayerNorm 近似误差约 0.217，Attention 约 0.977；没有认证原函数或应用精度。上述微小分解误差不能替代近似精度验收。

## 运行状态与授权

原 r870 的 24 项离线验收已在 r878 完成。本次核验其执行计划、审计、终态和文件哈希，没有重复执行：24/24 Poseidon CPU 数值通过，三份包复用通过，最大绝对误差 5.926e-7。它属于当时源码绑定。

当前 r1025 的 118 项后端验收和 18 份包复用仍未启动，也没有批准收据。690 份源码、5 份上下文源码、3,303 份案例/证据、4 个驱动冻结保持。648 个后端文件、6 个 reference 文件及历史记录核对保持。

本批生产代码、后端、SDK、参数、reference、误差门限和 x86 配置保持。Hecate tracing、编译、密态执行、解密、新 Agent 生成、API、安装、下载、commit、push 均为 0。本批 work root 证据约 1.24 MiB，检查时磁盘余量约 45 GiB。

## 下一步

执行已准备且获具体授权的当前后端范围；在安全边界后接入新版本的常见调用规范化，再补受影响后端回归和另行获准的真实 Agent 测试。继续补投影后的 SDPA、掩码/causal 组合和更大图容量。

Bootstrap、密文分区、原模型应用精度、ResNet20、CUDA、x86 原生执行和空白机器安装保持各自验收状态。

[机器记录](agent-common-pytorch-intake-r1036.json) · [现有组件接口](application-agent-component-v1.md) · [待批准离线范围](agent-application-source-offline-proposal-r1025.md)

独立副本复现（在既有 poseidon-cpu 纯 Nix 环境内，使用新的 results 目录）：

~~~bash
LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONPATH=scripts/baseline \
"$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/venvs/hecate-2.0.1-cpu/bin/python" -B \
  scripts/baseline/benchmarks/tools/audit_common_pytorch_prototype.py \
  --previous /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-pytorch-idioms-r1031/public-intake/report.json \
  --prototype scripts/baseline/benchmarks/tools/fixtures/common-pytorch-capture-prototype-v3 \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/common-idioms-next
~~~
