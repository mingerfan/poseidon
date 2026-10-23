# 公开常量算术接入结果 r1384

可信 PyTorch 增加显式 v25，用于在 forward 中计算已绑定的公开常量，例如 `(x - mean) / (std + eps)`。修改范围为 Agent 的来源解析、公开值规范化、来源验证和能力描述。Poseidon、Runtime、Dacapo/Hecate、参数和数值门限保持。

| 检查 | 结果 |
|---|---|
| 单元与组件检查 | 204 passed，0 failed/error/skipped |
| 清单构造与旧默认选择 | 15/15，默认 v18、latest=v21 |
| 新来源上下文 | 加、减、乘、除、取负，各 3 例，15/15 |
| 隔离捕获及来源检查 | 15/15，实际 bubblewrap 与原资源限制 |
| 现有来源参考 | 240 组／1,312 值 |
| 额外独立数学检查 | 60 组／328 值，最大差异 2.220446049250313e-16 |
| 正式应用接入 | 3 个 v25 与 3 个旧来源，6/6 |
| 旧请求兼容 | 3 个请求内容、request_id 保持 |
| 新编译／密态／Agent 生成 | 0／0／0 |

新上下文包含均值／标准差归一化、两公开张量广播、布尔 mask 转浮点、alpha、倒数、除法链、Linear 权重乘法，以及转置和 reshape 组合。公开运算仅接受认证的公开依赖；输入相关运算保持在图中。数值类型、有限范围、广播容量、状态总量、原地修改和来源篡改有拒绝检查。

应用检查实际执行 submit_source → prepare_source → 来源捕获／参考 → 请求准备，随后取消任务；未进入生成或后端资格验收。程序包的 provenance-v25 和 reference-v25 格式已接线，新编译包重载仍待后端批次。

首轮 75 项检查记录 3 failures／17 subtest errors，根因是新版本 tracing 门禁及旧测试的未来版本假设；下一轮剩余 1 error，是纯公开倒数进入了密文算术的元数据检查。集中修正后 75/75，通过的首轮独立检查也在受影响批次内复核。一次开发驱动未显式选择 ARM 平台，在进入 Nix 前被门禁拒绝。初始探针使用错误参数的记录也保留；修正探针后确认原拒绝位于来源解析层。原日志与修改前文件均保留。

已确认：当前 ARM 源码的捕获、规范化、独立参考和应用接入通过；后端身份与 r1383 相同。

基于证据的推断：公开算术被导入层消解，后续沿用现有基础算术和 Linear 图；仍需当前绑定下的真实编译与密态验收。

尚未验证：v25 的 Poseidon 数值资格、编译包复用和真实 Agent 新生成。x86 本批没有运行。原通用算子、布局容量、bootstrap、整模型和 CUDA 目标继续。

唯一开发 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。10 个已有生产模块与 1 个旧测试作局部适配，新增公开算术模块、测试及开发工具。既有修改与未跟踪文件保留。API／HTTP、安装、下载、网络变更、后端／参数修改、commit／push 均为 0。

SDK：`8e88db16925fad029bdeaea53ebb2ed927ebdb9aece66dd7a252923ccb711809`。

[机器记录](agent-public-arithmetic-component-checkpoint-r1384.json) · [使用说明](operator-decomposition-and-python.md#pytorch-v25-公开常量算术) · [18 项离线方案](agent-public-arithmetic-offline-proposal-r1385.md)
