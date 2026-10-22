# R8e4：可行性验证改用留出数据，批量运行可接续，然后重跑 R8e 的真实运行

## 0. 诊断（Claude 于 2026-10-22 复核 R8e 的中断结果）

- ResNet20 的 agent 运行完成了。精调后冻结的程序预测 238 次自举，编译实测也是 238 次；验证集上一致率 994/1000。**但测试集只有 75.1%**（已跑 8300 张，其中 5 张发散）；R7c B 是 91.68%。
- Claude 在 500 张测试图上诊断：

  | 程序 | 与精确网络一致 | top-1 | 最大 max_u |
  |---|---|---|---|
  | R7c B（全部 cheb 63，定义域较宽） | 493 | 91.0% | 1.01 |
  | R8e 精调后冻结的程序 | 386 | 71.4% | 5e94（发散） |

  原因：现在的"验证集"取自**训练集**，网络见过这些图。测试图在各激活上的范围最多是它的 **1.33 倍**（各激活为 0.90–1.33 倍）。精调把定义域收紧到"验证集观测范围 / 0.8"，测试图一越界，31 阶多项式就在定义域外爆炸。R7c B 的 2 张发散也是同一个原因。
- **用户决定（2026-10-22）**：可行性验证改用网络没见过的数据。
  - CIFAR 测试集第 9000–9999 张为**留出验证集**，Agent 和精调可以在循环中使用；
  - 第 0–8999 张为**报告测试集**，只用于最终报告。M1 第 5 条使用的 40 张加密测试图（索引 0–67）属于报告测试集。
  - 训练集中的图像只用于校准。

  AGENTS.md 已经同步。
- 中断：SiLU 第 3 轮时执行会话丢失，原因未知；ResNet20 的整个 run 撞上 10800 s 上限（精调约 2.2 h，测试只跑到 8300 张）。所以批量运行要能接续，测试阶段要单独限时并保存进度。

## 1. 修改

路径相对 `DSL/DSL_agent/scripts/`。

### 1.1 `pt_agent/data/cifar.py`

```python
HELDOUT = range(9000, 10000)
REPORT = range(0, 9000)

def heldout_set(n: int = 1000, normalization=(MEAN, STD)):
    """测试集第 9000 张起的前 n 张（n ≤ 1000）。"""

def report_test_set(normalization=(MEAN, STD)):
    """测试集第 0–8999 张。"""
```

`validation_set` 和 `test_set` 保留，现有调用者的行为不变，只是 agent 不再使用它们。

### 1.2 `pt_agent/planners/agent_tools.py`

`build_context` 改为调用 `heldout_set(n_val, normalization=...)`，n_val 默认仍为 1000，normalization 取模型自己的归一化。needed_margin 的计算方式不变：它现在度量的是"留出数据相对校准集"的放大倍数，正好是我们需要的量。AgentContext 记录 `val_split="test[9000:10000]"`。

### 1.3 `pt_agent/run.py` 的 test 阶段

- 改用 `report_test_set`（9000 张）；REPORT 记录 `test_split="test[0:9000]"`。
- 新增 `torch_top1_report`：torch float32 原模型在这 9000 张上的 top-1。
- M1 第 3 条的判定改为 `top1_test >= torch_top1_report - 1.0`。
- **进度保存**：每完成 500 张，把 `correct`、`diverged_indices`、`images_done` 写入 `<out>/test_progress.json`。progress 文件中同时保存冻结程序的完整源码文本 `program_text`。如果启动时该文件存在，并且 `program_text` 与当前冻结程序的源码完全相同（直接比较字符串，不用哈希），就从断点继续；否则从头开始，并把旧文件改名为 `test_progress.stale.json` 保留。
- test 阶段单独限时 7200 s。超时的话，保存进度，以退出码 124 结束。

### 1.4 `pt_agent/experiments/r8b_dev.py`

- 新增 `--resume <run_dir>`：每个模型都要满足"REPORT 中 complete=true 且 stage=complete"才跳过；不满足的模型从头运行。例外：如果它的冻结程序已经存在、只剩 test 没做完，就只接续 test。
- 每个模型的总限时改为 21600 s，test 阶段另按 §1.3 计时。
- 外层启动方式：用 `setsid nohup` 启动，并写 `<run_dir>/runner.pid`。汇报中写明如何接续：`--resume <run_dir>`。
- REPORT 顶层增加 `val_split` 和 `test_split`。

### 1.5 基线在新口径下重算（不调用 API，不重新编译）

新增 `pt_agent/experiments/r8e4_baselines.py`。在报告测试集（9000 张）上，对以下两个冻结程序运行 test 阶段，记录 top-1、n_diverged 和 torch_top1_report：
- R6c 规则程序：`run-resnet20-rule-20261020T165817/program.ptdsl`
- R7c B：`run-resnet20-agent-20261021T033036/program.ptdsl`

同时在留出验证集上运行 validate，记录 agree 和 max_u。

## 2. 测试（test_r8e4_split.py，用 nixrun，离线）

1. heldout_set 和 report_test_set 的索引互不相交，并集为 0–9999；heldout_set(1000) 的第一张就是测试集第 9000 张（逐像素比较）。
2. frozen-selections 中 encrypted_test 的 40 个索引全部属于 REPORT。
3. build_context(n_val=50) 使用的是留出数据：图像等于 `heldout_set(50)`，`val_split` 为 "test[9000:10000]"。
4. test 阶段的接续：用一个 2 张图的假 test 函数模拟"跑完 1 张后中断"，再次启动后从第 2 张继续，并且 correct 正确累加。
5. `--resume`：构造一个临时 run_dir，其中 1 个模型的 REPORT 为 complete，另 1 个为 incomplete；验证只重跑后者（mock 掉内部的 run 调用）。
6. R1–R8e3 的测试全部通过。

## 3. 运行

1. 运行 `r8e4_baselines`（不调用 API）。
2. 用新的 run_dir 运行 `r8b_dev --round R8e4 --tuner on`：4 个开发模型，LLM 调用合计不超过 24 次。R8e 原 run 作废，但它的 9 次调用照实计入累计用量。中断时用 `--resume` 接续，已用的调用照实累计，接续本身不增加预算。
3. 汇报内容同 R8e §5，另外增加：
   - 每个模型的"留出验证 agree / 报告测试集 top-1 / torch_top1_report / n_diverged"；
   - 精调前后的对照；
   - 基线（R6c、R7c B）在新口径下的结果。

STATUS.md：新增 R8e4 一行；R8e 按本轮结果标为完成；M1 第 1–4 条以 R8e4 中 resnet20 的结果为准，测试口径改为 9000 张。

## 4. 不要做

- 不把报告测试集的任何结果放进 Agent 的反馈或精调；不调整精调的 TARGET_U 和门槛（先看留出数据下的真实效果）。
- 不删除 R8e 原 run 的目录和报告。
- 不打冻结标签。
