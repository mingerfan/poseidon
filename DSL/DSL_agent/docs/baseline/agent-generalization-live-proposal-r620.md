# 新入口真实 Agent 评测提案

状态：已准备，未启动。提案绑定：`207f317b36f217482f095c1b97c26520bdf901f12d1b617f4975a98087dfa850`。

| 组 | 案例 |
|---|---|
| 参数化 PyTorch，6 项 | MLP、Conv→Pool、残差、BatchNorm、矩阵融合、形状变换→Linear |
| 非重复随机权重，8 项 | Linear、MLP、残差、通道 Linear、批量 Linear、双输入融合、共享中间值多输出、Conv→Pool |

逐项 ID、输入输出、请求哈希、来源、代码和依赖绑定见 [JSON 提案](agent-generalization-live-proposal-r620.json)。两组均已有真实 Poseidon CPU 离线数值验收；这次使用 Application 的实际生成、修复和交付路径。

外发内容为公开图、固定权重或其描述符、布局、DSL 规则和受控诊断。测试输入、reference、人工或规则 DSL 答案及凭据留在执行端。

- Provider：DeepSeek deepseek-flash/high/stream，explicit-v10，Poseidon w45。
- 每项最多 3 轮修复；每次生成最多 3 次传输重试。
- 全批最多 56 次生成、224 次 HTTP。
- 单次 1200 秒、384000 tokens；API 和密态串行，编译最多 2 jobs、链接 1。
- 全批 12 小时，每项 4 小时；单项 4 GiB，全局产物 32 GiB，磁盘余量 4 GiB。
- 没有明确货币上限；超时重试可能重复计费，token 账本不等于账单。

成功项验收真实编译、产物、四组密态数值、准确生成来源、程序包重载。失败分层记录；完整性、环境、资源或取消故障停止相关批次。未启动项不算通过；不自动补发不确定调用。

取得该绑定的新付费授权后，在配置好的 ARM Ubuntu 执行：

```bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/run_application_matrix.py \
  --execute docs/baseline/agent-generalization-live-proposal-r620.json \
  --approved-binding 207f317b36f217482f095c1b97c26520bdf901f12d1b617f4975a98087dfa850 \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generalization-live-r623
```

此提案没有安装、下载、后端或参数修改、提交与推送权限。
