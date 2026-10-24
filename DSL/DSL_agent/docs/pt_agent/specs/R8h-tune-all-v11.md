# R8h：对每个可行的 LLM 候选都做精调，并把 4 个模型重跑为冻结 v1.1

用户决定（2026-10-23）：修改精调的触发规则；在 S0 之后、U1 之前，按冻结协议重跑 4 个模型，作为冻结 v1.1。

## 0. 背景

- 现在的规则（`planners/agent.py`）是 `if tuner and eligible and (new_llm_best or not has_tuned)`：只有当 LLM 给出新的最好结果时才精调，第一次可行的候选除外。这条规则是在精调需要 16–24 分钟时定下的。
- R8f 之后，一次精调用 tensor 引擎约 1–2 分钟，加上最终 Mock 复核（ResNet20 约 133 s），总共约 3–4 分钟。
- R8g 暴露了这条规则的代价：MLP 第 2 轮 LLM 提出了可行的 cheb 255（8.64M µs），它比当时最好的结果慢，所以没有触发精调；而精调把它从 255 降到 127，就能得到 5.29M µs、20 次自举、精度下降 0，比最终选出的 minimax（6.29M µs）快 16%。

## 1. 修改 `planners/agent.py`（冻结模块，本 spec 明确授权修改）

1. **触发条件**改为 `if tuner and eligible`：每个通过资格检查的 LLM 候选都做精调。资格检查即现有的 `eligible`：精确检查通过、验证完成且无发散、stage 为 complete、满足自举输入约束。
2. **去重**：同一次 `run_agent` 中，对规范化后程序文本（`print_program(parse(source))`）相同的候选，不重复精调，直接复用上一次的 TuneResult。trace 中记录 `tuned_reused_from=<iteration>`。
3. **反馈**：
   - 每个做了精调的迭代，都按现有的 `feedback_message(..., tuned=tuned)` 把精调结果发给 LLM；
   - 没有做精调的迭代（不可行，或资格检查未通过）与现在相同；
   - `has_tuned`、`new_llm_best` 和 `llm_min` 只用于记录，不再控制触发。
4. **候选池**：精调后可行的程序照旧加入候选池（`origin='tuned'`），LLM 原样的可行候选也照旧加入；`confirm_select` 不变。
5. **提前停止**规则不变：最好结果连续 2 轮改进不到 5% 就停止。
6. 其他一切不变：提示词 r8-v4、`max_iters=6`、`tuner_budget_s=2400`、`max_sweeps=1`、LADDER、阈值、数据划分、confirm 参数。

`prompts.py` 中的系统提示如果写着"只有新的最好结果才会被精调"这类描述，改为"每个可行的候选都会被精调"，并在汇报中列出改动的原文和新文。如果没有这类描述，提示词不改。

## 2. 测试（新增 `tests/test_r8h_tune_all.py`）

使用假的 LLM 客户端和 patch 掉的 evaluate/tune，不调用真实 LLM：

1. 3 轮回复都可行，延迟依次为 100、120、90：三轮都调用了 tune（旧规则下第 2 轮不会调用）。
2. 第 2 轮回复的程序与第 1 轮的规范化文本相同：tune 只调用一次，第 2 轮记录 `tuned_reused_from=1`。
3. 不可行或资格检查未通过的候选不调用 tune。
4. 第 2 轮的 LLM 候选较慢，但精调后变快，成为新的最好结果：它进入候选池并被 confirm 选中，提前停止的计数相应重置。
5. 既有 358 项测试全部通过。只有直接断言旧触发规则的旧测试，允许改为新规则，并在汇报中逐项列出。

## 3. 重跑 4 个模型（冻结 v1.1）

- 模型：resnet20、resnet20_silu、cnn_plain、mlp。协议与 R8e5、R8e7、R8g 相同：
  - 循环内用留出集 test[9000:10000]；
  - 冻结时在确认集 test[8000:9000] 上执行 `confirm_select`（`MAX_CONFIRM_EVALS=6`）；
  - 选出的程序走 P2 路径编译；
  - 用 Mock 在报告集 test[0:8000] 上评测，结果不回传。
- 编排：在 `experiments/r8g_mlp_rerun.py` 的基础上推广为 `experiments/r8h_rerun.py`（只编排，支持 `--model`，逐模型断点续跑）。
- LLM 调用上限：本轮合计 **30 次**（累计不超过 78/300），超过就停下汇报。
- 运行顺序：mlp → cnn_plain → resnet20_silu → resnet20，先跑小模型，尽早发现问题。

结果目录：`pt_result("r8h-v11-<时间戳>")`。

## 4. 汇报

每个模型和冻结 v1（R8e7；MLP 另列 R8g）并列成一张表，内容包括：

- 最终程序：每个 act 的方法、阶数、定义域；
- 预测自举数和编译自举数，预测延迟；
- 相对规则基线的自举数减少比例；
- 确认集的 acc_drop，报告集的 top-1 和 acc_drop；
- LLM 调用数、迭代数、精调次数（其中复用的次数）、总耗时（其中精调耗时）；
- 选中程序的来源：哪一轮，是 LLM 原样还是精调后的结果。

另外逐轮列出：LLM 候选是否可行，精调前后的预测延迟。

STATUS.md 新增 R8h 一行。

## 5. 冻结标签

本轮不打 tag。结果确认后，由用户决定是否打 `pt-agent-freeze-v1.1`，指向包含本轮改动的提交，以本轮 4 个模型的结果作为 v1.1 的冻结结果。R9b 之后以 v1.1 为准。

## 6. 不要做

- 不修改 §1 以外的冻结模块，不修改精调参数和 confirm 参数。
- 报告集结果不用于任何选择，也不回传给 Agent。
- LLM 调用不超过 30 次。
