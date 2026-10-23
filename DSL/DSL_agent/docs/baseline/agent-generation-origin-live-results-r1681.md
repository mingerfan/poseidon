# 高层模型上下文：真实 Agent 验收

本批 6 项全部通过，其中 4 项首轮通过，2 项经自动修复通过。失败、未运行和跳过均为 0。生成源码与原始 Provider 响应一致，六份程序包通过重载、新输入和重复执行。

## 验收结果

| 模型 | 生成次数 | 首轮 | 最终数值 | 包复用 |
|---|---:|---|---|---|
| LayerNorm 输入变换 | 1 | 通过 | 通过 | 通过 |
| 函数式 LayerNorm，rank 3 | 3 | 修复后通过 | 通过 | 通过 |
| Conv2d 组合 | 1 | 通过 | 通过 | 通过 |
| BatchNorm1d 组合 | 1 | 通过 | 通过 | 通过 |
| AvgPool2d 组合 | 1 | 通过 | 通过 | 通过 |
| LayerNorm 基础 | 2 | 修复后通过 | 通过 | 通过 |

资格验收共 24 组输入、104 个输出值，最大绝对误差 3.1647e-07；程序包复用共 24 组、104 值，最大绝对误差 2.1331e-07。逐项门限保持 1e-5 + 1e-4*abs(reference)。

真实链路为 Hecate tracing、Dacapo-modified、Earth/CKKS IR、RuntimePlan / OperatorSpec / plaintext bundle、CKKS Runtime、Poseidon CPU、解密和独立参考比较。复用执行的 Agent / 编译调用均为 0，执行器只获得评估密钥。

本批使用显式近似计算；原函数应用精度尚未认证。LayerNorm 的“输入变换”指归一化之前的公开运算。

## 保留的修复记录

- rank-3 LayerNorm：首轮编译累计 scale810>720；第二轮编译和密态执行成功，但最大误差约0.5801；第三轮通过。
- 基础 LayerNorm：首轮编译累计 scale750>720；第二轮通过。
- 数值失败候选对原张量直接使用 suffix_gather 掩码，遗漏窗口成员；既有反馈要求先窗口求和、再 gather 和广播。

旧失败、原始响应和绑定全部保留。编译失败不计密态通过；数值失败的实际密态调用单列。未手写替换候选，未更改模型、reference 或门限。

## 调用与产物

实际 9 次生成 / 9 次 HTTP，传输重试 0；9 次编译尝试，7 次编译成功；19 次密态调用，其中 1 次数值失败、6 次资格验收、12 次程序包执行。

Provider 报告总 token 535,289，其中 prompt 170,003、completion 365,286，reasoning 包含在 completion 中。货币费用尚未核验。

复核 183 个编译产物文件、172 个程序包文件的哈希。累计结果 39.432 / 48 GiB，磁盘余量 29.835 GiB。

SDK：6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429
审核绑定：23ec347a0691b6248a17a5ba4373fac0ca17ac1e4888cea5ab93b49e8565ef9c
批次绑定：af729fb959624174bb99215bbaca3cd0d21a98efe24228b4ef313412155da7ce

## 后续

高层数学上下文已通过本批正常应用入口验收。接下来补齐 Softmax / LogSoftmax、GroupNorm / InstanceNorm 等复合近似的源码校准，分别记录原函数误差与近似执行误差。容量、更多算子组合、真实 bootstrap、CUDA 实机、整模型和 ResNet20 仍按完整计划推进。

Poseidon、CKKS Runtime、Dacapo、参数、网络及 x86 配置保持；安装、下载、commit、push、PR 均为 0。当前无活动批次，已核对真实进程。

[机器可读报告](agent-generation-origin-live-results-r1681.json) · [接口说明](agent-generation-origin-context-r1680.md)
