# Softmax / LogSoftmax 源码校准

源码准备接口 v3 支持模块、函数和 Tensor 方法中的 Softmax / LogSoftmax。每个算子验证了三个不同上下文，覆盖 rank 1–3 和不同轴。

## 接口

请求使用 poseidon-source-preparation-request-v3，保留 inspect、observe、freeze 三个动作。观测选择使用 poseidon-source-fx-calibration-selection-v3。旧 v1/v2 继续兼容。

运行入口：

    python3 -B scripts/agent_calibration.py inspect \
      --request /absolute/request.json --output /absolute/results/inventory \
      --trust-local-code

inspect 返回含 dim、dtype 及模块路径的绑定图。observe 分别选择原输入和声明的 exp 近似之和；freeze 为 exp 与 reciprocal / log 核对各自区间及配置优先级。

分母投影示例：

    {
      "kind": "softmax_exp_sum",
      "exp": {"method": "taylor", "degree": 2, "domain": [-1.0, 1.0]},
      "arithmetic": "reference-float64"
    }

投影按已声明的 Taylor 多项式计算分母，原模型 forward 保持其原生 dtype 和执行结果。轴必须为静态整数，dtype 接受 float32、float64 或默认值；动态参数返回明确错误。

## 本批验收

| 验收 | 结果 |
|---|---:|
| 控制及旧接口检查 | 106/106 |
| 正常隔离校准、配置冻结和 PyTorch 捕获 | 6/6 |
| 规则候选真实编译、Poseidon 数值验收 | 6/6 |
| 程序包重载、新输入与重复执行 | 6/6 |
| 编译 / 密态调用 | 6 / 18 |
| API 调用 | 0 |

资格验收为 24 组输入、80 值，最大绝对误差 5.573e-9；复用为 24 组、80 值，最大绝对误差 1.633e-8。门限保持 1e-5 + 1e-4*abs(reference)。

源校准与实际捕获采用同一配置选择记录。148 个编译产物和 160 个程序包文件已核验哈希。

首批六项在隔离 worker 中因 v3 白名单遗漏而失败，原记录保留；修正 Agent 接线后，同六项续验通过。Poseidon、Runtime、Dacapo、参数和 reference 保持。

这些结果验证声明的近似计算；原函数应用精度尚未认证。新模型未加入冻结的 1,200 模型计数。

## 真实 Agent 测试

同六项当前请求已冻结，采用 explicit-v21、DeepSeek deepseek-flash/high/stream、P4/w45。每例最多三轮修复、每次最多三次传输重试；全批最多 24 次生成 / 96 次 HTTP，API 和密态串行、编译 2 / 链接 1。

单次 1200 秒 / 384000 tokens；全批 12 小时、每例 4 小时；新增产物 512 MiB、总量 48 GiB、余量 4 GiB，native 4 GiB / CPU150 秒。授权来自当前计划内全部测试的统一批准。货币上限未设置，重试可能重复计费。新生成结果按单独批次记录。

后续接入 GroupNorm / InstanceNorm 等复合统计，并继续原函数精度、容量、更多算子组合、bootstrap、CUDA 和整模型工作。

[机器可读报告](agent-compound-source-calibration-r1686.json) · [真实测试范围](agent-compound-source-live-scope-r1687.json)

## 真实生成结算

本批真实 Agent 6/6通过，5首轮/1修复；[完整结果](agent-compound-source-live-results-r1688.md)。
