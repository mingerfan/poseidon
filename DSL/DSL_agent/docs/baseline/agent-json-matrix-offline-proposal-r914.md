# JSON 矩阵与组合离线验收提案

状态：待授权。此提案替代未执行且源码绑定已过期的 r892；原文件与历史记录保留。

## 范围

共 48 项：原八个编译失败、四个兼容回归，以及 JSON v2 十二组各三例。全部使用已冻结的可信规则源码，实际 Agent 生成和 API 调用为 0。

每项检查真实 Hecate tracing、Earth／CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、Poseidon CPU 密态执行、最终解密和四组输入逐项比较。

六份程序包检查导出、重载、三组新输入重复执行及缓存复用。另对三个显式近似域模型检查越界输入在加密前被拒绝。

## 预算

整批最多 45 分钟；首次编译验收每例最多 180 秒。复用一份已验证 Nix 环境，候选保持独立沙箱；编译≤2、链接≤1、密态串行。新增产物≤4 GiB、总量≤32 GiB、磁盘余量≥4 GiB。

API、安装、下载、SDK重建、后端库、参数、reference、门限及Git发布变更均为0。达到边界停止并保留现场，未执行项记为 not_run。

冻结门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

当前产物 24.81 GiB，可用磁盘 45.21 GiB。

## 案例

| ID | 来源 | 前次后端状态 |
|---|---|---|
| attention_core_0 | trusted_pytorch_v7 | failed |
| attention_core_1 | trusted_pytorch_v7 | failed |
| attention_core_2 | trusted_pytorch_v7 | failed |
| channel_broadcast_mix_0 | trusted_pytorch_v7 | failed |
| channel_broadcast_mix_1 | trusted_pytorch_v7 | failed |
| channel_broadcast_mix_2 | trusted_pytorch_v7 | failed |
| norm_depth_branches_1 | trusted_pytorch_v7 | failed |
| norm_depth_branches_2 | trusted_pytorch_v7 | failed |
| indexed_batched_matmul_1 | trusted_pytorch_v7 | passed |
| cnn_norm_head_1 | trusted_pytorch_v7 | passed |
| public_weight_1 | trusted_pytorch_v7 | passed |
| static_split_stack_1 | trusted_pytorch_v7 | passed |
| matrix_matrix_0 | declared_json_v2 | not_run |
| matrix_matrix_1 | declared_json_v2 | not_run |
| matrix_matrix_2 | declared_json_v2 | not_run |
| batched_cipher_0 | declared_json_v2 | not_run |
| batched_cipher_1 | declared_json_v2 | not_run |
| batched_cipher_2 | declared_json_v2 | not_run |
| batch_broadcast_0 | declared_json_v2 | not_run |
| batch_broadcast_1 | declared_json_v2 | not_run |
| batch_broadcast_2 | declared_json_v2 | not_run |
| higher_rank_cipher_0 | declared_json_v2 | not_run |
| higher_rank_cipher_1 | declared_json_v2 | not_run |
| higher_rank_cipher_2 | declared_json_v2 | not_run |
| vector_left_0 | declared_json_v2 | not_run |
| vector_left_1 | declared_json_v2 | not_run |
| vector_left_2 | declared_json_v2 | not_run |
| vector_right_0 | declared_json_v2 | not_run |
| vector_right_1 | declared_json_v2 | not_run |
| vector_right_2 | declared_json_v2 | not_run |
| public_right_0 | declared_json_v2 | not_run |
| public_right_1 | declared_json_v2 | not_run |
| public_right_2 | declared_json_v2 | not_run |
| public_left_0 | declared_json_v2 | not_run |
| public_left_1 | declared_json_v2 | not_run |
| public_left_2 | declared_json_v2 | not_run |
| scalar_chain_0 | declared_json_v2 | not_run |
| scalar_chain_1 | declared_json_v2 | not_run |
| scalar_chain_2 | declared_json_v2 | not_run |
| public_dot_0 | declared_json_v2 | not_run |
| public_dot_1 | declared_json_v2 | not_run |
| public_dot_2 | declared_json_v2 | not_run |
| matrix_chain_0 | declared_json_v2 | not_run |
| matrix_chain_1 | declared_json_v2 | not_run |
| matrix_chain_2 | declared_json_v2 | not_run |
| scalar_domain_0 | declared_json_v2 | not_run |
| scalar_domain_1 | declared_json_v2 | not_run |
| scalar_domain_2 | declared_json_v2 | not_run |

程序包复用：channel_broadcast_mix_0, indexed_batched_matmul_1, public_weight_1, higher_rank_cipher_2, public_left_1, scalar_domain_1。

## 绑定与启动

提案绑定：`3c3168bd8919638722c688f3fe1fc90b6ea6c0cfd585c7ac68f6b41ae5df69c5`。冻结 676 份运行源码、786 份案例／证据文件和4个驱动；SDK工具与参数身份保持，Agent适配器身份按新源码更新。

需要本次授权的原因：原24项窗口已完成，十二项旧提案未执行且绑定过期；新增48项后端验收使用独立范围和时间预算。

批准后生成独立范围收据，再执行：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-operator-graph-v2-preflight-r902/launch-backend-r915.py
~~~

[机器提案](agent-json-matrix-offline-proposal-r914.json) · [当前接入结果](agent-json-matrix-static-acceptance-r913.md)。
