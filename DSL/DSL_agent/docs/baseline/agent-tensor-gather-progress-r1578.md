# PyTorch gather 接入

显式来源版本 `poseidon-trusted-pytorch-v37` 已接入 `torch.gather` 和 `Tensor.gather`，复用已有静态切片、拼接和 reshape。Dacapo、CKKS Runtime、Poseidon 和安全参数保持原样。

## 输入范围

- 固定 shape、静态整数轴，索引为固定公开 int64 张量。
- 支持各行不同索引、重复索引、负轴表示和非索引轴上的较小输出尺寸。
- 索引 rank 与输入一致；每项索引位于对应轴的有效范围。
- 按资源配置检查 rank、元素、图展开和存储预算。rank32 已有数学与 FX 上下文检查；本次真实密态输入覆盖 rank1–3。
- 空索引、原负索引、秘密索引、动态轴、原地 out 和 sparse_grad=True 保持明确拒绝。

索引和原始 dtype、源 FX、规范化 FX、版本与哈希进入来源记录。生成模型接收规范化模型与固定绑定，未提供规则 DSL 答案或私有测试数据。

## 验收

| 层 | 结果 |
|---|---|
| 组件与旧接口 | 117 passed，failed/skipped 0 |
| 隔离 PyTorch 捕获 | 3/3，48 组独立 reference，来源重载通过 |
| 规则候选真实编译 | 3/3 |
| Poseidon CPU 数值 | 3/3，9 次密态调用，24 组／112 个输出值 |
| 最大绝对误差 | 3.6870751446e-7 |
| 程序包重载／新输入／重复／缓存 | 3/3 |
| 当前真实 Agent | r1577 已启动，结果待结算 |

模型分别为向量重排、每行不同索引、负中轴及较小非索引维度。上述离线通过不计为当前 Agent 生成通过。

主批最多 12 次生成、48 次 HTTP、12 次编译、18 次密态调用；每例 4 小时、全批 12 小时。API 与密态串行，编译≤2／链接≤1，新增≤4 GiB／全局≤48 GiB／余量≥4 GiB，native 限制保持。使用既有计划内总测试授权。

当前 SDK：900393362406e0d051d1ecca04fe619d89344ea916963012e4549bf797348c68。
主 PID960964/start27458191；后置 PID961570/start27470965。共享源码冻结到两者终态及调用结算。

## 使用

通过 `scripts/prepare_pytorch_source.py --source-format poseidon-trusted-pytorch-v37` 创建清单，再按已有 PyTorch 模型导入和 Agent 入口执行。默认 v18 与 latest=v34 保持，旧显式版本与请求哈希继续兼容。

## 记录

- [精确实时测试范围](agent-tensor-gather-live-proposal-r1577.json)
- [机器可读进展](agent-tensor-gather-progress-r1578.json)
- [独立离线审计](../../scripts/baseline/benchmarks/tools/tensor_gather_acceptance_audit_v1/report.json)

下一条：

```sh
python3 -B -m json.tool /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-tensor-gather-live-r1577/report.json
```

通用算子与组合、容量、原模型应用精度、bootstrap、CUDA 实机和 ResNet20 继续分别推进。
