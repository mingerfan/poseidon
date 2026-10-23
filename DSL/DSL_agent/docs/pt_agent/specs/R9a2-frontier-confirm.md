# R9a2：在 2000 张未见图上复核 E1 的快速可行配置；修正 E6 与 E3 的统计口径

前置：R9a 已完成，结果目录为 `pt_result("r9a-20261022T152208")`。本轮不调用 LLM，不读取报告集，不修改 pt_agent 已有模块，只修改 R9a 新增的两个实验脚本和对应测试。预计约 1 小时。

## 0. 诊断（Claude 于 2026-10-23 复核 R9a）

1. **E1 的 feasible_like 只用了 200 张图**，噪声大。R8e5/R8e6 已经表明，31 阶程序在小样本上看不出问题，到报告集上却掉了约 1.3pp。现在 E1 显示 cnn_plain 有 73%、mlp 有 67% 的 feasible_like 配置比冻结程序更快，这需要在更大的未见集上复核，才能作为论文中的前沿，也才能用来解读 R9b 的"只用搜索"组。
2. **E6 有 2 对"不一致"**（`existing/r6b/1`、`existing/r6b/3`）。这两个程序的 header 是 `refresh="agent"`（Agent 自己放置自举，用 pars 编译），对应的正确预测是 lazy 模型，历史上与实测完全一致（461/461、382/382）。E6 却对所有程序都使用了 depth-dp 模型。应当按程序 header 自动选择，即 `cost.estimate` 不传 placement_model 时的默认行为。
3. **E3 中 MLP 的人工决定数为 0**：Dacapo 的 MLP 专家程序使用平方激活 `h1 * h1`，两层全连接是手写循环，不调用 `HE_*` 函数，所以正则一个都没有数到。

## 1. 修改

### 1.1 `experiments/r9a_motivation.py`

1. **新子命令 `e1confirm`**：读取 R9a 的 `e1/rows.jsonl`。对每个模型，选出以下配置，按 key 去重：
   - 所有 feasible_like 且 `latency_pred_us < 冻结程序的 latency_pred_us` 的配置；
   - E1 的 Pareto 前沿上 feasible_like 的配置；
   - 冻结程序本身（kind="reference"）；
   - 所有 feasible_like 的 anchor 配置。

   对每个选中的配置，按 rows 中记录的 config 重新构造程序（`apply_config`），在**未见集 2000 张**（测试集第 8000–9999 张）上执行 `validate(..., stop_on_divergence=False)`。结果按 key 追加写入 `e1confirm/rows.jsonl`，支持断点续跑。

   汇总 `e1confirm/summary.json`，每个模型记录：
   - 复核的配置数，以及其中在 2000 张上仍满足"无发散且 acc_drop_pp ≤ 0.5"的数量和比例（称为 robust_feasible）；
   - robust_feasible 中最快的延迟，以及它与冻结程序延迟的比值；
   - 冻结程序自身在 2000 张上的 acc_drop_pp 和 n_diverged；
   - 200 张与 2000 张两种结果的判定混淆表：两者都可行、只有 200 张可行、只有 2000 张可行、两者都不可行。
2. **E6 汇总口径**：重算 summary 时，每个程序的预测用 `cost.estimate(program, weights)` 的默认行为（refresh="compiler" 用 depth-dp，refresh="agent" 用 lazy），记为 `matched`；lazy 和 depth-dp 两列保持不变，作为对照。summary 增加 `matched` 一节，指标与另外两节相同。不需要重新编译。
3. **E3**：增加一个人工核对表 `MANUAL_DECISIONS`，只针对正则无法识别的专家程序，并在输出中注明 `source="manual"`：
   ```python
   MANUAL_DECISIONS = {"MLP": dict(activations=1, linear_layers=2,
                                   note="square activation h1*h1; two hand-written diagonal-style linear loops")}
   ```
   MLP 的 decisions 改为 `3*0 + 2 = 2`：平方激活的方法是固定的，不产生近似决策，计 0；两层手写全连接各计 1 个布局与算法决策。另外，在 loc.csv 中增加一列 `decisions_source`，取值为 "regex" 或 "manual"。

### 1.2 测试（在 `tests/test_r9a_motivation.py` 中新增）

1. e1confirm 的选择规则：构造一份 rows，断言选中集合等于"比冻结程序快的 feasible_like ∪ Pareto 上的 feasible_like ∪ reference ∪ feasible_like 的 anchor"，并且已经去重。
2. e1confirm 只读取测试集第 8000–9999 张：patch `report_test_set`，被调用就抛出异常。
3. E6 的 matched 口径：refresh="agent" 的程序使用 lazy 预测，refresh="compiler" 的程序使用 depth-dp 预测。
4. E3：MLP 一行的 decisions 为 2，decisions_source 为 "manual"。
5. R1–R9a 的全部测试通过。

## 2. 运行

`r9a_motivation e1confirm --source pt_result("r9a-20261022T152208")`，然后执行 `e6` 和 `e3` 的汇总重算（不重新编译）。结果写入新目录 `pt_result("r9a2-<时间戳>")`，R9a 原目录保持不变。

## 3. 汇报

- 每个模型：复核的配置数、robust_feasible 的比例、robust_feasible 中最快延迟与冻结程序之比、冻结程序在 2000 张上的结果、200 张与 2000 张的混淆表。
- E6：matched 口径下自举数完全一致的比例、延迟误差的中位数和最大值。
- E3：修正后的表格。

STATUS.md：新增 R9a2 一行。

## 4. 不要做

- 不修改冻结的 Agent 模块，不调用 LLM，不读取报告集，不重新编译。
- 不修改 R9a 原结果目录中的任何文件。
