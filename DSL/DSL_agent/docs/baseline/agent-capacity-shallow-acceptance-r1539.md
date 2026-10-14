# 容量布局 v10 阶段验收

显式 `capacity-public-tensors-v10` 已完成 10 个不同模型的规则基线验收。两个原失败模型均通过真实编译、Poseidon CPU 密态执行和数值比较。当前新 Agent 生成尚未开始。

## 改动

- 广播按块内连续或等间距位置组合旋转，减少重复展开。
- Linear 在既有工作量和常量预算内比较对角与分块布局，优先较浅的合法表达。
- 完整末轴行可用非二幂宽度归约；跨块行沿用已有路由。
- 程序包清单较大时保存等价紧凑 JSON，128 KiB 上限保持。
- v10 通过 `--unified-packing` 或组件 `capacity_packing_version` 显式选择；来源版本及旧默认分别保持。

规划值是局部成本估计；编译及执行状态在真正验收前为 `unknown`。

## 结果

| 验收 | 结果 |
|---|---|
| 当前绑定相关控制 | 112 passed，0 failed，0 skipped |
| 规则基线 | 10/10 个不同模型通过，分属两个源码绑定 |
| 当前 SDK 补验 | 5/5；5 次完成编译、9 次密态调用 |
| 独立数组审计 | 52 组输入、8,140 个输出值 |
| 最大绝对误差 | 1.83955517041e-06 |
| 程序包 | 10 份重载与缓存复用；3 个模型新输入和重复执行 |
| 本阶段真实 Agent 调用 | 0 |

10 个模型为 rank_axis_8、rank_linear_16、broadcast_inner、broadcast_tail、suffix_three、suffix_five、suffix_cross_chunk、linear_rows、linear_tail、compat_rank_permute_32。当前 SDK 验收覆盖其中 5 项；另外 5 项保留前一绑定的已审计证据。

全阶段记录 11 次完成编译、17 次密态调用，包含一次数值通过、清单交付失败的原尝试。它不额外增加通过模型数。门限保持 `1e-5 + 1e-4*abs(reference)`，数学解释器与 Torch reference 按 `1e-12 + 1e-12*abs(reference)` 复核。

## 原失败

r1534 的 5 项在 tracing 前缺少受控沙箱模块挂载；补齐清单后复验。
r1536 的 rank_linear_16 数值通过，程序包清单为 132,743 字节，超过 128 KiB；紧凑存储后完整重验通过。
旧失败、模型、候选源码、产物及绑定全部保留。

这些是规则程序的后端结果；当前 Agent 生成质量单独测试。高 rank、最大容量及任意组合没有由这 10 项获得全面认证。

## 身份与资源

- Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e480`。
- 当前 SDK：`711489d834dcb22010d77c03138595bf8b6f360b2a44fc8b4596ab9fdf0e7baa`。
- 独立审计：`6e1e0d07a2cb2db05ada7aac3eb1228a9cfc6c88a1974d7b3b48ccba2851f139`、`f2794768dce2b063471eea151deddc442ec023c742ae4ef05b5372e5271de220`。
- 产物 37.38/48 GiB；磁盘余量 31.98 GiB。
- Poseidon、Runtime、Dacapo、参数及网络保持；API、安装、下载、提交和推送均为 0。

下一批为 9 个 v10 上下文的真实 Agent 生成，沿用现有测试总授权；冻结范围及额度后执行。Bootstrap、CUDA 实机、应用精度和 ResNet20 继续独立推进。
