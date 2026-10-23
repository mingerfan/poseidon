# 小模型 Benchmark

开发工具用于检查模型转换、DSL 构造和后端执行。普通应用直接使用 [Agent 入口](../../scripts/README.md)，不需要运行 Benchmark。

## 模型与任务

冻结主集有 1200 个去重小模型、779 个拓扑组。包含仿射、Linear、平方激活 MLP、卷积/池化/BN、残差、多输入融合、张量组合与多项式近似。

另有 96 个历史兼容配置、92 个补充模型和 30 个补充上下文。定向构造、helper 任务不另算新增模型。

| 任务 | 数量 |
|---|---:|
| 主集自由生成 | 1200 |
| 补充模型 / 补充上下文 | 92 / 30 |
| 定向 DSL 构造 | 627 |
| 定向 helper | 60 |
| 定向固定多项式 | 21 |

主集按拓扑组分为开发 720、验证 240、留出 240。每例有四组基础输入和最多十二组额外探针；独立数学解释器与 CPU float64 PyTorch 计算参考。

自由生成只给数学模型和规则。定向任务另指定 helper、数组、循环、参数展开等要求，并检查实际使用情况。

## 命令

在已配置平台的 Ubuntu 仓库根目录运行：

```bash
python3 -B scripts/benchmark.py check
python3 -B scripts/benchmark.py plan
python3 -B scripts/benchmark.py baseline --limit 48
python3 -B scripts/benchmark.py free --limit 48
```

这些命令检查或规划，不启动付费生成。`baseline`、`directed` 的 `--execute` 才运行离线候选：

```bash
python3 -B scripts/benchmark.py baseline --limit 48 \
  --execute --output "$HOME/poseidon-work/benchmark-run"

python3 -B scripts/benchmark.py summary \
  --output "$HOME/poseidon-work/benchmark-run"
```

输出目录必须新建。分片最多 48 项；续跑要求模型、源码、配置与预算身份匹配。详细参数用 `python3 -B scripts/benchmark.py --help` 查看。

## 状态

- 主集 1200 个模型全部通过双参考检查。
- 401 个语义分区已有分类与验收状态；627 项定向构造对应 209 个构造分区。
- 第二阶段计划的真实 Agent 任务已经评测并审计，仍有失败；不等于全部语义生成成功。
- 最近剩余 26 项为 Max、MaxPad、ReLU 相关模型及两个补充上下文。真实 bootstrap 仍不可用。

历史结果按原源码版本保存，不拼接为最新版本的成功率。通过计数、编译失败、数值失败、服务异常和未完成项分开报告。

[阶段验收](stage2-acceptance-r152.md) · [剩余问题](stage2-remaining-closure-r211.md) · [组件验收](application-component-acceptance-r216.md)

模型与覆盖清单位于 [benchmarks](../../scripts/baseline/benchmarks/)，人工探针和批次审计位于 [benchmarks/tools](../../scripts/baseline/benchmarks/tools/)。
