# PyTorch 来源扩展与真实 Agent 验收准备

ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。本批完成来源配置和验收入口，真实 Agent 测试待批准。

## 来源配置

`prepare_pytorch_source.py --source-format latest` 选择当前 v21，并将具体版本写入清单。默认仍为 v18，显式 v12–v21 按原版本处理。八个已有 v21 模型由 `latest` 重建的清单与原定义一致，源码、权重和数学计算保持。

```bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:1,4 --output logits \
  --source-format latest --resource-profile general-tensors-v1 \
  --manifest ./source.json
```

## 本批结果

来源配置 67 项、执行入口及账本 76 项，共 **143 passed / 0 failed / 0 errors / 0 skipped**。检查包括 Linux 子进程清理、未批准执行拒绝、来源与工具哈希、调用上限、未知 HTTP 结果停止和既有数值验收等级。

36 项已有来源完成隔离捕获及独立明文参考检查，耗时 672.83 秒，原 900 秒窗口内完成。36 个请求、数学图签名与带算子拓扑签名均各自去重。准备任务已关闭，未来付费执行创建新任务与账本。

| 数量 | 来源版本 | 内容 |
|---:|---|---|
| 8 | v18 | 矩阵、分头、广播收缩、多操作数、公开权重及视图组合 |
| 6 | v17 | chunk/unbind、尾块、共享写入、公开权重与多输出 |
| 6 | v19 | 对角线、trace、标量因子与共享视图写入 |
| 6 | v20 | 一维 unbind、标量视图、快照、stack 与公开权重 |
| 8 | v21 | 平方、四次方、公开标量除法与别名组合 |
| 2 | v16 | self Attention＋Linear＋残差；投影后的 cross Attention |

本批 **新编译、密态执行、解密、Agent 生成与生成 HTTP 均为 0**。此前各版本的规则候选数值验收见 r1339、r1340、r1346、r1347、r1348；具体通过范围按各自绑定查阅。

当前 SDK 身份为 `67fc349e857bff911cc66c6a889d033853ba69b86d6df3b6114a9aa477e1539f`。654 个受保护的后端与数学参考文件、两份历史付费请求及 r1341 原方案核验保持。r1341 仍为旧失效方案，旧 cross 未知调用保留，未自动重试。

## 真实 Agent 批次

[36 项方案](agent-source-expansions-live-proposal-r1349.md)及[精确绑定](agent-source-expansions-live-proposal-r1349.json)已准备，尚未批准或执行。复用 Application、Provider 与程序包接口，要求真实编译、Poseidon CPU 数值验收、包重载、新输入执行和缓存复用。

```bash
python3 -B scripts/baseline/benchmarks/tools/source_live_v2/run.py --plan
```

一次无凭据 HTTPS 探针正常退出，鉴权和长流尚未测试。来源、后端、参数或工具变化会使此次计划失效。

## 后续工作

当前来源的真实生成及自动修复评测之后，继续较大图和权重、应用精度、bootstrap、密文交接、CUDA 实机和整网目标。上述项目按各自标准验收。

[机器报告](agent-source-expansions-readiness-r1349.json) · [当前源码验收](agent-pytorch-exact-arithmetic-acceptance-r1348.md)
