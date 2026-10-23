# 当前批准执行结果（r1098）

离线验收完成。真实 Agent 批次因 TLS 证书验证失败停止，未取得候选。

| 范围 | 结果 |
|---|---|
| 151 项规则基线 | 128 项 Poseidon CPU 数值通过；23 项编译失败 |
| 25 项程序包复用 | 20 项通过；5 项因原候选失败未执行 |
| 当前程序包输出交付 | 3/3；使用已验收的密态输出，无新增密态调用 |
| 76 项真实 Agent 请求准备 | 76/76；原模型、公开请求及 request_id 一致 |
| 真实 Agent 执行 | 首项服务失败；其余 75 项未启动 |

离线累计比较 712 组密态输入、3,936 个输出值，最大绝对误差约 2.08e-7。细项见[离线验收](agent-common-pytorch-offline-acceptance-r1097.md)。

首项 `pytorch_state_mlp` 记录 1 次生成预留、1 次 HTTP 尝试，错误为 `transport_certificate_failed`，未得到响应或候选。Ubuntu 的固定 Nix Python、系统 Python，以及 Mac 现有代理都收到自签名证书；Ubuntu 当时将 API 域名解析到 `198.18.2.23`。这属于服务网络故障，尚未进入模型生成、编译或数值验收。具体 DNS／代理配置原因仍待确认。

原 76 项范围与预算保持，剩余最多 303 次生成预留、1,215 次 HTTP；首项剩余 3 次生成、15 次 HTTP。已启动的 15 项组保留原 12 小时截止时间，其余两组尚未启动。费用未取得服务商 usage，金额未知。网络恢复后先通过无凭据 TLS 校验，再建立续跑绑定；旧失败记录、原请求和已通过结果保留。

生产源码 695 份、后端 648 份和原 reference 全部核验保持；本轮本地改动仅涉及执行工具与文档。分支为 `lhy-agent-dsl`，无安装、下载、参数变更、commit 或 push。数值门限仍为 `1e-5 + 1e-4*abs(reference)`。

结果目录位于 Ubuntu `/home/lhohy/poseidon-work/platforms/aarch64-linux/results`。完整状态见[JSON](agent-approved-execution-status-r1098.json)，授权去重清单见[当前执行范围](agent-current-execution-authorization-r1089.md)。
