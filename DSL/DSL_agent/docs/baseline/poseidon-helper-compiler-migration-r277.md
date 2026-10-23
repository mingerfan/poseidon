# Poseidon helper 与产物审计入口迁移

## 本轮结果

`scripts/benchmark.py helper-directed` 已接入 Poseidon CPU，保留冻结任务、原模型和原 helper 源码。`compiler-evidence` 已改为审计 RuntimePlan、OperatorSpec、常量 bundle 和 Poseidon 数值记录；旧 HEVM 审计模块及其历史报告保留。

| 验收 | 结果 |
|---|---|
| 静态、兼容与审计测试 | 33 passed，0 failed，0 skipped |
| 原 SiLU / BatchNorm / Concat helper | 3/3 真实编译、密态执行及数值通过 |
| helper 实际贡献 | 三例各有真实 frontend 调用及有限返回干预证据 |
| 完成任务续跑 | 3 例重新审计后复用原报告；没有新增生成、编译或密态执行 |
| 编译产物审计 | 12/12 原数值通过记录核验成功；审计本身不执行密态计算 |

三个 helper 完成 48 组双参考、12 组密态输入、36 个输出值；最大绝对误差 8.43019e-8。门限仍为 `1e-5 + 1e-4*abs(reference)`。候选为冻结的人工源码，不计真实 Agent 成功，也不新增模型集数量。

最终源码前的 r276 结果和精确工具源码已保留。随后补充了续跑记录与原审计结果的逐字段核验，并在新绑定 r277 下重新验收三个原例。

## 产物审计范围

新审计器复用现有候选、布局和 RuntimePlan 检查器，并核对固定 Q/P、degree、tc128、执行端私钥边界及数值结果。12 份产物中，modswitch 出现于 12 份，rescale 8 份，relinearization 6 份，rotation 8 份。

这里的数量是有对应指令的已执行程序数，不能直接当作不同拓扑数量。指令、scale/level 声明和 rescale 素数经静态传播核验；当前没有逐指令 native trace。加法、乘法或旋转的单条运行时状态不据此宣称已观测。密钥要求和指令对应关系已核对，没有外部轮转密钥文件或逐步密钥使用观察。

## 使用

~~~bash
python3 -B scripts/benchmark.py helper-directed \
  --suite scripts/baseline/benchmarks/semantic-v2-ledger-r38 \
  --frozen-suite-index-sha256 ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2 \
  --task-id upstream_silu_frozen_0 --limit 1
~~~

默认显示计划。执行时增加 `--execute --output <新的结果目录>`；相同绑定通过 `--resume` 核验并复用完成任务。原任务的 w40/w45 映射到 Poseidon 配置；可显式选择 CPU 配置。分片最多 48，native 串行，30 分钟／512 MiB，本机总产物 32 GiB、磁盘余量 4 GiB。

~~~bash
python3 -B scripts/benchmark.py compiler-evidence \
  --suite scripts/baseline/benchmarks/semantic-v2-ledger-r38 \
  --frozen-suite-index-sha256 ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2 \
  --batch-report /absolute/path/to/poseidon-batch/report.json \
  --output /absolute/path/to/results/new-compiler-audit.json
~~~

该入口审计规则/人工数值记录。真实 Agent 记录仍通过已有 provenance 审计接口核验，不能冒充人工记录。当前没有新增模型 API 调用。

## 后续与边界

应用核心 513 文件保持 r274 绑定一致，三个整模型付费提案未失效，也未执行。此次只改 Benchmark 分派、开发适配器和文档；Dacapo、Runtime、Poseidon 库、参数、reference 和门限未改。

旧静态 rejection CLI 仍待迁移。数学上下文剩余 12 个编译容量失败及真实 bootstrap 误差问题未解决，见 [r271](poseidon-dsl-migration-r271.md)。r235 独立参数试验与 r274 真实 Agent 三例仍分别等待批准。x86 原生、空白安装和 CUDA 继续暂缓。

[机器可读报告](poseidon-helper-compiler-migration-r277.json)。原始结果：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-benchmark-tools-r277`。本轮 API、安装、下载、commit 和 push 均为 0。
