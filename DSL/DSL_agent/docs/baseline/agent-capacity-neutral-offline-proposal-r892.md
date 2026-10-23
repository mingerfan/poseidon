# 规则表达修复：12 项离线验收提案

状态：待批准。源码、模型、权重、请求和工具已冻结；本提案没有启动编译或密态执行。

## 案例

| 用途 | 案例 |
|---|---|
| 八项原失败 | attention_core_0–2、channel_broadcast_mix_0–2、norm_depth_branches_1–2 |
| 四项兼容回归 | indexed_batched_matmul_1、cnn_norm_head_1、public_weight_1、static_split_stack_1 |

12 项均已完成当前版本的隔离 PyTorch 捕获和每例 16 组参考验证。数学图、公开权重、layout、请求哈希及 w45 配置保持。候选为已通过静态差分的规则源码，真实 Agent 新生成为 0。

## 验收与预算

复用现有工具，真实 tracing → Earth/CKKS IR → RuntimePlan／OperatorSpec／常量 bundle → Poseidon CPU 密态执行 → 四组输入解密及逐项比较。成功后检查 channel_broadcast_mix_0、indexed_batched_matmul_1、public_weight_1 三包的重载、新输入各两次与缓存复用。原例失败时保留失败，包检查记为未运行。

全批及独立审计合计最多 30 分钟，每例编译/执行验收最多 180 秒。编译≤2、链接≤1、密态串行；复用同一 Nix 环境。新增产物≤4 GiB、全局≤32 GiB、磁盘余量≥4 GiB。API、安装、下载、SDK 重建、后端与参数修改均为 0。参考和误差门限保持 `1e-5 + 1e-4*abs(reference)`。

r870 的 24 项 split 批次已完成并关闭；此次 12 项不属于其案例范围，需要新的有界离线窗口。失败和超限保留现场，独立案例继续；环境、完整性或预算故障停止相关批次。

## 绑定与下一命令

绑定：`f9addcd26424038837ee78709bd3a66477481073ac2ac1426de21813db7e6ed1`。精确源码、案例、工具、依赖和参数身份见[机器提案](agent-capacity-neutral-offline-proposal-r892.json)。

批准并建立运行范围后执行：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-capacity-neutral-preflight-r881/launch-backend-r893.py
```

[静态验收](agent-capacity-scheduling-static-acceptance-r885.md)。
