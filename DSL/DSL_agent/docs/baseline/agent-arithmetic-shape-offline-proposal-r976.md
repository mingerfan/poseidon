# 当前源码下的离线密态验收方案

待批准，尚未启动。绑定：§f635a2d5c96f1e6845fc3b1279b2253f582862cd4d2263fd2bb81dca1e650b81`。

范围共 **103 项**：

- 保留原 94 项：八个原编译失败、四项兼容回归、36 个 JSON 矩阵、21 个 JSON 算术及 25 个 PyTorch v8 来源。
- 增加九个 PyTorch v9 方法结果 shape/size/numel 来源；原 v8 失败保持。
- 检查 12 份成功程序包的重载、三组新输入各重复执行两次，以及缓存复用。失败项记录原失败层，不用其他模型替换。

所有候选为固定规则程序，验收目的为当前 Poseidon CPU 编译执行及新输入兼容；真实 Agent 生成另行计分。

| 预算与要求 | 上限 |
|---|---|
| 整批墙钟 | 2 小时，包含检查与包复用 |
| 单例 | 180 秒 |
| 分片 | 48 / 48 / 7，依次执行 |
| 密态执行 | 串行 |
| 编译／链接 | ≤2 / ≤1 |
| 新产物 | 4 GiB |
| 全局产物／磁盘余量 | ≤32 GiB / ≥4 GiB |
| API、下载、安装、SDK 重建 | 0 |
| 后端、参数、模型、reference、门限修改 | 0 |

复用一次已验证 Nix 环境，每个候选继续独立沙箱、目录、网络隔离、AST 与资源检查。完整性、环境或预算故障停止相关批次并保留现场；独立模型失败继续记录。超限、未运行与失败均不计通过。

当前 683 份运行源码、2,193 份案例／证据文件及四个新驱动已冻结并核验。原 94 项数学图、请求、公开权重和候选不变；受捕获身份变化影响的 37 份 PyTorch reference 在新目录核验，原结果保留。实际 SDK 身份与原提案一致。

旧 r950 的 1,367 份证据及四个驱动完整保留，原提案未执行且源码已过期。此前 r870 的 24 项已完成关闭，不能授权本批。此次批准用于当前新绑定的有限后端窗口。

批准前只有来源准备和拒绝启动检查。授权收据尚不存在，后端输出目录尚不存在。

[精确案例、请求、源码与工具哈希](agent-arithmetic-shape-offline-proposal-r976.json) · [接入验收](agent-pytorch-arithmetic-shape-acceptance-r974.md)。

批准并生成对应范围收据后的入口：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/frozen_arithmetic_shape_backend/launch.py
~~~
