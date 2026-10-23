# 生成来源核验与公共文件读取（r789）

日期：2026-10-05。ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。

## 本批改动

生成来源核验提取到 `generation_receipt.py`，配置及 usage 校验提取到 `generation_protocol.py`。
`application_generation.capture` 继续由生成控制层调用；程序包核验使用数据接口。
`deepseek_provider.Config`、`ProviderError`、usage 接口和旧生成记录格式保留。

`strict_json_files.py` 提供原有的受限 JSON 读取和原子报告写入。
应用导入、程序包和模型导入入口共用它；Benchmark 原入口保留同一函数别名。
提取的配置、usage、来源验证、读取和写入逻辑与原实现 AST 一致。

## 验收

- 276 项静态／组件／旧接口／真实隔离捕获测试：273 passed、0 failed/errors、3 skipped。跳过项依赖旧 SEAL 证据目录。
- 14 份历史真实 Agent 生成记录完整性及来源核验通过；42 项改源码身份、HTTP 账本或加入私有字段的拒绝检查通过。过程中未导入 Provider、生成适配器或 Benchmark 运行器，没有编译或密态执行。
- 当前三个规则候选和一个历史 Agent MLP 完成真实 tracing、编译、RuntimePlan／OperatorSpec／bundle 核验、Poseidon CPU 密态及数值比较。
- 四份包均完成重载、三组新输入各重复两次，以及零生成缓存复用；显式 compiled 等级和旧 Linear CLI 故障流程另行通过。

当前数值通过范围为 60 组输入、400 个输出值，最大绝对误差 `2.7474946274e-07`。
门限保持 `1e-5 + 1e-4*abs(reference)`；CLI 故意数值失败另有 4 组输入，单列负例。

| 当前验收来源 | 数量 | 范围 |
|---|---:|---|
| 原规则候选 | 3 | `axes_2`、`duplicate_input`、`duplicate_square` |
| 历史真实 Agent 原源码 | 1 | `weights_mlp`，原模型、请求、权重、源码保持 |
| 当前新生成真实 Agent | 0 | 本批没有付费调用 |
| scripted CLI 故障检查 | 1 | 解析失败 → 数值失败 → 通过 |
| 显式 compiled | 1 | 编译与产物通过；密态／数值声明保持 false |

64 组双参考、36 个结构化输出值、105 个程序包文件及129 个执行产物哈希核验通过。
新绑定为 `f63553e3476a467e2ad1862ddb9e196eedc5f984320ac603fb792087d0c3b5f9`。
历史生成及编译绑定不改写；新 MLP 包保存当前离线验收来源，原 Agent 生成来源仍保存在原包和本批来源记录中。

## 历史包的兼容范围

14 份生成记录核验通过，不等于14份包在当前 SDK 重新执行通过。
当前数据／来源检查中，8 份声明式权重包通过；6 份旧 PyTorch 包因 `Trusted reference identity` 拒绝。
其 reference 绑定的捕获源码与当前版本不一致。旧记录保留原身份，没有重写源码哈希或放宽检查。

本批新包与来源使用当前绑定。后端验收覆盖以上四个代表例；其他历史能力各保留原范围。

## 仍待拆分的边界

应用和程序包的数据读取、生成来源验证已脱离 Provider 和 Benchmark 运行器的实际导入。
整个模块的保守源码闭包仍保留相关路径：生成657个文件，其余阶段655个，共享655个。

主要路径是 `sandbox_policy → upstream_helper_probe → semantic_benchmark_execution → deepseek_provider`；
旧 CLI／规则 lowering 也仍由 helper 清单带入。下一批优先提取 helper 的可信契约与文件清单，保留全部动态授权、文件和哈希检查。
一套 SDK 对应一个绑定 worker，底层 WORK／SDK 全局路径继续保留。

新增输入真实 Agent、原模型应用精度、深 Attention、bootstrap、密文分区、CUDA、x86 原生／空白安装及 ResNet20 继续单列。

## 记录与资源

结果目录：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-receipt-preflight-r777`。
当前包：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-receipt-backend-r786`。

- `tests-r779.json`：最终测试。
- `history-audit-r781.json`：旧生成记录、完整性和拒绝检查。
- `dependency-audit-r782.json`：实际源码依赖路径。
- `audit-r788.json`：独立数值与产物审计。
- `final-check-r789.json`：最终来源、后端、reference、Git及资源检查。

r780 首次完整旧包检查的来源拒绝保留。r783／r784 在执行前分别因验收工具误用导入目录和传入内部选项而停止，未编译／密态执行；修正工具后建立 r786 新绑定。
r787 的来源标签把“提供了 provenance”误当作“执行过原 PyTorch module”；r788 按实际字段重新核对同一产物，密态没有重跑。

663 个冻结 Agent 文件、648 个后端文件、6 个 reference／门限／检查器及91 个历史文件核验一致。
原1380条修改路径完整保留；最终检查时新增产物约 36.60 MiB，全局 22.69 GiB，余量 46.06 GiB，已记录 worker 约 222.50 秒，包含失败尝试。
编译≤2、链接≤1、密态串行；本批新增API／HTTP、安装、下载、SDK重建、后端／参数／reference 修改和Git发布为0。

[机器报告](agent-receipt-acceptance-r789.json) · [应用接口](application-agent-component-v1.md)。
