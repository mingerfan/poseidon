# GroupNorm / InstanceNorm 源码准备与 Agent 验收

2026-10-14，ARM Ubuntu；分支 lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。

6 项当前真实 Agent 程序全部通过 Poseidon CPU 数值验收，3 项首轮通过、3 项自动修复后通过。每份程序包完成新输入与重复输入执行。成功源码逐份与原始 Provider 响应一致。

| 案例 | 生成次数 | 结果 | 最大绝对误差 |
|---|---:|---|---:|
| group_module | 2 | 修复后通过 | 1.242e-07 |
| group_function_affine | 2 | 修复后通过 | 2.452e-09 |
| group_module_batch | 1 | 首轮通过 | 1.019e-09 |
| instance_module1_unbatched | 1 | 首轮通过 | 5.296e-07 |
| instance_function_affine | 1 | 首轮通过 | 1.360e-09 |
| instance_module2_unbatched | 2 | 修复后通过 | 1.143e-09 |

## 范围

准备接口 v3 支持 GroupNorm 分组方差加 eps、InstanceNorm 通道方差加 eps 的观测与 rsqrt 配置冻结。分别覆盖模块、函数、批次变化和非批次输入；本批输入 rank 2–3，每例 8 个元素，分到 period-4 密文，采用 general-tensors-v1 与原 w45 参数。

rsqrt 采用显式 Newton 2 次迭代，区间 [1e-4,4]。数学定义、权重、独立 reference 和误差门限保持原样。原生 forward、dtype、状态与随机数状态检查由源码准备验收执行。

## 执行和来源

真实 Hecate tracing → Dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU → 解密与独立参考比较。

- 静态控制：118 passed，failed/error/skipped 均为 0。
- 源码校准、配置冻结、隔离捕获：6/6。
- 当前真实 Agent：6/6，9 次生成、9 次 HTTP；9 次编译尝试，6 次成功。3 次编译失败保留。
- 真实密态调用：18 次；其中 6 次首次验收、12 次程序包执行。
- 首次验收：24 组输入、192 个输出值，最大绝对误差 5.29644192842e-07。
- 包复用：24 组输入、192 个输出值，最大绝对误差 4.35999977497e-07。
- 门限：abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。
- 哈希审计：173 个程序包文件、155 个编译产物文件。
- Provider total_tokens：460748。reasoning 已包含在 completion 内；货币费用未核验。

原规则基线 6 项编译失败、人工精确因子合并见证 6 项通过，保留为独立来源。本次真实 Agent 程序由模型生成及自动修复，人工见证未进入生成请求。

## 绑定与交接

审计绑定：c4aad41268339d34b3d3f6534ec59c1a734840a3f01403624a64d799a1e7d7df。
当前 SDK：6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429。
原始结果：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normalization-source-live-r1693。
机器可读报告：agent-normalization-source-live-results-r1694.json。

当前结果验证已声明近似计算的 CKKS 数值行为。原函数与近似计算的误差、应用精度指标单独评估；完整归一化配置、容量、bootstrap、CUDA 与整模型覆盖仍有后续工作。

本批沿用已批准测试预算，新增产物约 42.4 MiB。累计原始证据 39.802/48 GiB，磁盘可用 29.458 GiB。后端库、参数、reference、门限、网络配置保持；安装、下载、提交、推送、PR 均为 0。

下一步：复用这 6 项和复合近似 6 项已交付程序的来源，在独立样本上评估原函数、近似计算及分解误差；不重复生成或重新编译。
