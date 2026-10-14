# Python 导入与 Poseidon CPU 验收

开发目录：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。保留现有模型、参数、误差门限和历史记录。

## 改动

- 模型导入入口统一进入 Poseidon CPU Nix 环境。
- 可信本地 PyTorch 捕获在隔离进程中检查原始 `nn.Module`、高层图和分解图。使用 CPU float64 eval 语义及 16 组固定输入，记录分解误差和近似误差。
- 文件清单、文件数量、累计大小和哈希在读取阶段检查。
- DSL 程序包保存源码清单、来源哈希和参考报告。重载校验来源后直接执行已编译的 RuntimePlan。

## 验收

| 检查 | 结果 |
| --- | --- |
| 组件、导入、程序包、旧接口和平台测试 | 124 passed / 0 failed / 3 skipped |
| 受限多文件 Python 平方 MLP | 编译、Poseidon CPU 密态及数值通过 |
| 可信 PyTorch nn.Module 平方 MLP | 编译、Poseidon CPU 密态及数值通过 |
| 两个程序包重新加载 | 密态及数值通过，生成与编译次数均为 0 |

三个 skipped 是历史 SEAL 物理证据测试，单列保留。

两个示例均为 4→2→2 平方激活 MLP。导入验证共 32 组双参考输入；可信模块另与原 `nn.Module` 比较 16 组。首次密态验收各 4 组输入，共 16 个输出值；重载使用相同输入再比较 16 个值。

| 示例 | 首次最大绝对误差 | 重载最大绝对误差 |
| --- | ---: | ---: |
| 受限多文件 Python | 4.37862e-8 | 2.04368e-8 |
| 可信 PyTorch 模块 | 8.60622e-9 | 1.55691e-7 |

验收门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。实际检查包括 frontend tracing、IR、RuntimePlan、OperatorSpec、常量 bundle、Poseidon CPU 执行、解密及逐项比较。

候选来自现有 self-test，属于离线后端验收。真实 Agent 成功数、付费调用、安装、下载、commit、push 均为 0。该结果限定于两个小型 MLP。

## 使用

详见[Python 与高层算子输入](operator-decomposition-and-python.md)。可选受限源码静态解析，或显式授权可信本地 PyTorch 捕获。两者输出统一模型 JSON，随后使用现有 Agent 和程序包接口。

原始结果：

- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-python-import-r287/tests.log`
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-import-e2e-r287/report.json`

[机器可读报告](poseidon-python-import-migration-r288.json)仅含汇总指标和证据位置。中间版本 r282/r283/r284 及 r281 源码快照继续保留，最终结果绑定 r287 实际运行源码。

## 后续

剩余 12 个原多项式模型仍受编译容量限制；bootstrap 的有符号输入误差仍未达标，r235 参数试验待批准。

真实 Agent 三整模型提案更新为 [r286](poseidon-whole-models-live-proposal-r286.json)，取代未执行的 r274/r284。三个模型、公开请求、后端二进制和调用预算保持一致；新应用核心绑定为 514 文件。MLP、RMSNorm、Attention 的新 Poseidon Agent 评测均未执行。
