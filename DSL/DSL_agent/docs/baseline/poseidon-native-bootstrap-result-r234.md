# Poseidon CPU bootstrap 独立参数试验 r234

本次试验已结束，**数值验收失败**。使用用户批准的独立参数；当前 Agent 配置保持原样。

## 执行结果

| 检查 | 结果 |
|---|---|
| 参数 | N=32768，Q17×48 + P1×60，乘积实际 876 位，tc128 表上限 881 位；Poseidon 接受 |
| 构建 | 使用现有 CPU 库，配置、编译和链接通过 |
| 零输入：刷新 | 16,384 槽全部通过，最大绝对误差 4.9833e-7 |
| 零输入：平方及 rescale | 全部通过，最大绝对误差 3.5527e-15 |
| 零输入：第二次刷新 | 全部通过，最大绝对误差 4.4919e-8 |
| 有符号输入：刷新 | 16,384 槽全部超限，最大绝对误差 1.0398136e-4 |
| 剩余检查 | 按方案停止；有符号输入后续两项、随机和边界输入各三项均未运行 |

共计划 12 项阶段检查：3 通过、1 失败、8 未运行。运行中的加密随机性保留；输入值和参考固定。零输入通过不能证明非零消息刷新正确。

门限为 `1e-5 + 1e-4 * abs(reference)`。有符号输入分别为 0.125、-0.25、0.375、-0.5，平均误差方向均朝向零；允许误差分别为 2.25e-5、3.5e-5、4.75e-5、6e-5。

## 已确认与待定位

真实 CPU bootstrap 完成了 CoeffToSlot、EvalMod、SlotToCoeff 并返回密文，输出 level=2、log2(scale)≈48。零输入后续平方的 level=1，再次刷新回到 level=2。执行器调用只使用评估密钥；解密只用于验收，没有中间解密再加密。

相较旧 Q13×60 探针，本次不再在 EvalMod 抛出 scale 越界，但非零输入误差仍不合格。误差方向一致提示近似或归一化误差，尚未定位到唯一根因；不能认定仅增加 scale、模数或修改 Agent 表达即可解决。

下一步先核对 EvalMod 多项式、倍角和归一化的独立数学参考。修改算法、参数或 Runtime 接线前，另行给出具体方案。当前不能开放 Agent 的 bootstrap 能力。

## 资源与证据

总耗时 32.66 秒，新增产物约 9.47 MiB；探针地址空间限制 6 GiB。采样观察到进程 RSS 约 5.39 GiB，这不是完整峰值。原 report 中的 peak_child_rss_kib 是 launcher 统计，不能用于声称探针峰值；audit.json 已注明。

原始授权、源码快照、二进制、逐槽比较和报告：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-bootstrap-experiment-r234/`

[机器可读审计](poseidon-native-bootstrap-result-r234.json) · [批准方案](poseidon-native-bootstrap-proposal-r233.json)

未修改 Dacapo、Runtime 核心、Poseidon 库、Agent 参数或候选沙箱限制；无依赖安装、下载、付费调用、提交或推送。该结果属于独立 CPU 库试验，不是 Runtime 接入或真实 Agent 验收。
