# v22 构造语法：第二批真实 Agent 验证

## 范围

本批按用户“继续付费 API 测试、最近新增构造语法逐项真实 Agent 覆盖”的要求，
重新生成冻结 v22 清单中的 8 个候选。不是复用旧 response 的离线回放，
也不把重复生成计为新增语法或新增模型家族。

8 个案例仍是同一仿射模型的不同指定构造形式；12 个观察项需要实际执行，
并通过有限输出影响检查。不能据此宣称整个上游 DSL 已覆盖或得到形式化证明。

此前四个版本的重新审计已通过：v19 30/30、117项；v20 10/10、10项；
v21 15/15、19项；v22 8/8、12项。历史审计结果：
`/home/lhy/poseidon-work/results/recent-construction-audit-_ouybh_5/report.json`。
该审计没有新 API 调用，也没有重新运行 FHE。

## 冻结运行配置与证据

- 批次：`/home/lhy/poseidon-work/results/agent-batch-go_bnk71/report.json`。
- DeepSeek `deepseek-flash`，high，max_tokens=384000。
- API 并发上限 10（本批 8 例），native 执行并发 2。
- 单请求硬超时 1200 秒，最多 3 次传输重试和 3 轮反馈修复。
- `.env` 读取凭据，不打印或改动密钥。
- 真实执行后端为 upstream SEAL HEVM CPU，不是 Poseidon GPU。
- 数值阈值仍为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 不修改输入、权重、reference、安全参数或冻结构造要求。

网络预检：
`/home/lhy/poseidon-work/results/deepseek-tls-probe-k3aynz2u/report.json`。
默认路径 3/3 通过预期的无凭据 HTTP 401 检查，证书及 hostname 验证开启；
127.0.0.1:6478 的三次连接均失败。因此只在本批命令指定
`--loopback-proxy-port 0`，不修改全局代理。不把短请求成功视为长请求可靠性证明。

## 执行记录

本批已结束：8/8最终通过、7/8首次通过；12/12观察项均有真实Agent候选，
通过独立审计的执行与有限输出影响检查。

| 案例 | 语义要求 | API请求次数 | 最大绝对误差 |
|---|---|---:|---:|
| ou-neg | cipher对象数组一元负号 | 1 | 1.309394e-8 |
| ou-zero | 0-D对象数组取负、标量Expr结果 | 1 | 7.347958e-9 |
| ou-view | view取负、fresh结果独立 | 1 | 1.138296e-8 |
| ou-positive | 公开对象一元正号复制、源后续修改 | 1 | 1.743414e-8 |
| ou-negative-call | np.negative入口 | 1 | 1.211277e-8 |
| ou-positive-call | np.positive入口 | 3 | 1.869381e-8 |
| ou-plain | 构造Plain取负并保留类型 | 1 | 2.766749e-8 |
| ou-mixed | 同一数组cipher/数值/bool分别影响输出 | 1 | 9.154261e-9 |

12项为：operator.USub、operator.UAdd、call.USub、call.UAdd、cipher_cells、
plain_cells、public_cells、boolean_cells、zero_dim、input_view、fresh.USub、fresh.UAdd。
独立审计还验证mixed来自同一次操作，view与fresh.USub来自同一次操作。

`ou-positive-call` 首次候选引用了未定义的 `public_weight`，在 static_check
阶段失败；原始响应与反馈保留，由 Agent 接收诊断后修复，不人工替换源码。
修复请求第一次传输达到1200秒硬超时；第1次传输重试收到响应并最终通过。
这是1轮语义修复、1次传输重试，不是2轮语义修复。不能据此断言超时根因是TLS。

## 统计、审计及清理

- API请求10次：9次返回usage，1次超时无usage；无未完成或最终失败案例。
- 已收到usage：prompt 61943、completion 73291、合计135234 tokens。
  超时请求是否被服务商计费、实际货币费用未查询；135234不是完整账单结论。
- 32组真实密态输入、128个输出值逐项通过；最大绝对误差2.7667490964944363e-8，
  加权MAE4.804414790917505e-9，最大非零reference相对误差7.37799759065183e-8。
- 批次耗时1293.896秒；平均语义反馈修复0.125轮/案例。
- 独立审计：`/home/lhy/poseidon-work/results/object-unary-agent-audit-_am9bk3l/report.json`。
- 批次SHA256：`a9f68a91b21ceb1179deae96c1843d060c8ad01792464a3de4bb6f69ca313e9a`。
- 审计SHA256：`68dc5d1fa549cf2185f70683ab7664036c1c749851cc617f021959ac4c33a4c4`。
- 已清理本批可再生FHE密钥5431840080 bytes（约5.06GiB），8例密钥目录残留0。
  原始随机密钥不可恢复，但可以重新生成；源码、请求/响应、IR、编译产物、
  解密值、诊断与清理记录保留。API凭据没有删除或改动。
- 结束时WSL可用189.14GiB，无本批付费worker残留。
- 相关回归51项全部通过、0跳过、0失败；显式启用本批真实Agent证据与既有
  人工密态正反例。回归输出中的模拟provider重试不属于新的付费调用。

首次独立审计被过早并行启动，与仍运行的付费批次争用 nix-portable launcher
互斥锁而退出。该失败属于本地审计调度，不属于 Agent 生成或 TLS 失败；
在付费批次结束后串行重跑已成功，没有绕过互斥锁或修改环境。

## 复验

以下仅离线审计，不触发新付费请求：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 480s python3 scripts/baseline/audit_object_unary_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-go_bnk71
```

保留所有旧批次；当前阶段不 commit、push 或创建 PR。
