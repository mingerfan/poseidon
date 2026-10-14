# 分块权重与 Linear 容量验收

ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl。[核心 Poseidon CPU 迁移](poseidon-capacity-acceptance-r386.md)已通过。本批补齐公开张量绑定、跨输入块 Linear 和预算内的 batched Linear 输出。当前模型、reference、参数与误差门限保持原值。

## 实现

- 非均匀公开张量按广播形状分块，包含 gain、offset 和权重；候选使用绑定后的公开向量。
- Linear 根据公开索引选择逐行归约或旋转加权求和；支持跨输入块的 batch 边界、尾块和公开 bias。
- capacity-public-tensors-v2 记录所选映射。旧 v1 请求按原版本重建，模型和原候选保持原样；规则路径为 capacity-chunks-v3。
- I/O 各 1024 元素、最多四块；常量 256 项、请求 128 KiB、源码 64 KiB、golden 展开工作量 1024 的限制保持。

## 静态与参考

112 passed / 0 failed / 0 errors / 0 skipped。涵盖两套独立参考、公开绑定拒绝、逐项路由、零权重、尾部、batch 边界及组件兼容性。每个新增数学模型使用 16 组明文探针；受阻模型也通过双参考比较。受限 AST 算术解释是开发检查，密态结果另计。

27 个历史 Agent 请求、21 个先前容量请求及 15 个公开向量 v1 请求精确重建。15 个 v1 请求中选原 Linear 请求与候选通过组件重新编译、密态执行和逐项比较。

## 真实 Poseidon CPU

所有执行使用 Hecate → dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU。

| 批次 | 通过 | 阻塞 | 密态输出值 | 最大绝对误差 | 重载 |
|---|---:|---:|---:|---:|---:|
| r392/r393：9 新模型与 6 兼容回归 | 15 | 0 | 11,424 | 2.18998e-8 | 3 包 |
| r398：3 batched Linear 与 5 兼容回归 | 8 | 2 | 9,516 | 5.00867e-8 | 1 包 |
| r399：原 v1 请求组件调用 | 1 | 0 | 32 | 3.15176e-10 | — |

每例使用零、有符号、固定随机、边界四组输入；执行完成后解密并逐项比较。24 次通过执行对应 18 个不同数学模型，含重复回归与不同源码绑定，不能将执行次数当作新模型数量。4 包重载比较 5,184 个值，最大绝对误差 9.32353e-9；重载生成与编译调用均为 0。

新公开张量、跨块 Linear、Linear→square→Linear 各有三个上下文。batched Linear 正例为 [128,4]→[128,4]、[1,192,3]→[1,192,3]、[1,2,128,4]→[1,2,128,4]，输出总元素分别 512、576、1024，均为实际权重矩阵计算。

这是规则基线验收，当前新增真实 Agent 生成数 0，付费调用 0。先前 27 份历史 Agent 的 Poseidon 验收保留原报告与绑定。

## 保留的阻塞

原矩形 batched Linear 两例保留：

| 原模型 | 计算 | 层 | 原因 |
|---|---|---|---|
| capacity_batched_linear_1 | [1,130,3]→[1,130,5] | 请求准备 | 279 个独立公开向量；估计 1677 项工作 |
| capacity_batched_linear_2 | [1,2,64,4]→[1,2,64,6] | 请求准备 | 280 个独立公开向量；估计 1679 项工作 |

它们的数学图合法、双参考一致，但当前公开绑定超过预算，未进入编译或密态执行。需要压缩公开映射和通用矩形 batch 路由；未更换模型、放宽限制或记作通过。

跨块 Conv/Pool、部分轴归约、非对齐连接、v2 helper/定向与 ResNet20 物理容量继续开发。12 个原深多项式编译容量失败保留。r235 bootstrap 独立试验已执行并消耗批准，有符号第二刷新 15 个值超门限，结果见 [r388](poseidon-bootstrap-ratio128-result-r388.md)；当前 Runtime bootstrap 仍受阻。

## 绑定与复现

r392/r393 执行源 582、应用源 525 文件；r398/r399 执行源 584、应用源 527 文件。前后批次分别绑定，旧证据没有重标。各批次 plan.json 保存实际 compiler、frontend、Poseidon 库、参数配置、补丁和平台身份；[机器可读报告](poseidon-capacity-linear-acceptance-r400.json) 保存路径、哈希与全部状态。

r390 一项测试首先命中请求哈希检查，调整测试以检查模型绑定后 r391 全通过。r396 五项错误中包含矩形容量阻塞和旧版本在 compact 递归中丢失的问题；修复版本传递后 r397 通过，原失败记录保留。

下一条命令重载已验收的 1024 元素 Linear 程序包：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-routing-numerical-r398/capacity_batched_linear_4/program --execute
~~~

本批无新安装、下载、付费、编译器/Runtime 核心、库、参数或门限修改，也没有 commit/push/PR。x86 原生、空白安装、CUDA 暂缓；全项目目标继续推进。
