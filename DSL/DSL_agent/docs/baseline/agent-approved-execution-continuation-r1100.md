# 真实 Agent 续跑（r1100）

DeepSeek TLS 连接已恢复，原 76 项批次已按原范围续跑。固定 Nix 环境中的 HTTP worker 验证了服务端证书与域名；代理参数仍为 0。

| 项目 | 配置 |
|---|---|
| 案例 | PyTorch 状态权重／函数式写法 15 项，矩阵与 InstanceNorm 60 项，Attention decode 1 项 |
| 模型、公开请求与依赖 | 与原冻结清单一致；新续跑绑定保留旧提案和停止记录 |
| Provider | DeepSeek deepseek-flash，high，stream |
| 单次 | 1,200 秒，384,000 tokens，最多 3 次传输重试 |
| 本次剩余额度 | 最多 303 次生成预留／1,215 次 HTTP；包含旧尝试的累计上限仍为 304／1,216 |
| 首项剩余额度 | 3 次生成／15 次 HTTP |
| API／密态／编译／链接并发 | 1／1／2／1 |
| 产物／磁盘余量 | 总量 32 GiB／余量至少 4 GiB，每例最多 4 GiB |
| 时间 | 15 项组保留 2026-10-07 13:23 北京时间截止；其余两组首次启动后各 12 小时 |

当前绑定为 `81c396aeb46344fdce88d746ea781e784afd0647b58d37ff848027028d57c01e`。旧 TLS 失败及已用额度保留，未知调用结果不会自动重试。预算控制测试 9 passed，0 failed/errors/skipped。

首项已经取得真实模型响应，完成编译和 Poseidon CPU 密态执行；第一次候选未通过数值比较，当前由原生成模型有界修复。整批状态以运行报告为准，尚未宣布完成。

程序包通过后，继续检查重载；PyTorch 组还使用 3 组新输入进行独立 reference 比较。原 151 项离线验收及已完成的包复用不重跑。

运行目录：Ubuntu `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-pending-continuation-r1099`。入口为 `scripts/baseline/benchmarks/tools/approved_pending_execution_v3/run.py`，当前已启动一个顺序执行进程。

本轮仅调整开发执行工具与交接文档；695 份生产源码、后端、参数、模型、reference 与误差门限保持。无安装、下载、commit、push 或 PR。

## 首项结果

2026-10-07T10:25:38.942397+08:00：MLP 经 2 次生成（一次修复）通过真实数值验收、程序包重载和 3 组新输入检查。旧 TLS 尝试另计。截至该快照，已通过 3 项（MLP、Conv→Pool、残差），正在执行 BatchNorm，尚未启动 72 项；整批继续顺序执行。
