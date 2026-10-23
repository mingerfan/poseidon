# R8g：用修复后的代码重跑 MLP 的 Agent（冻结 v1.1 的 MLP 部分）

用户决定（2026-10-23）：flat 尾槽缺陷（R8f2）修复后，重跑 MLP 的 Agent。其余 3 个模型（ResNet20、SiLU、cnn_plain）都是 mux 布局，不受该缺陷影响，冻结结果保持不变。

## 0. 背景

- 冻结的 MLP 程序（R8e7，minimax_comp，24 次自举，预测延迟 6 286 530 µs）是在带缺陷的验证语义下选出来的：flat 布局激活后尾槽为 p(0)，Chebyshev-MLP 的 logits 被加上常数偏移，很多 Chebyshev 候选被误判为不可行。R8f2 的重放中，`mlp/iteration-2`（cheb 255 阶）原记录的 acc_drop 为 1.4pp，修复后为 0。
- R8f2 的历史重算显示：E1 中 MLP 的可行配置从 98 个增加到 137 个，最快可行程序的预测延迟为 4 244 994 µs，比冻结程序快 33%。
- 当前代码：精调用 tensor 引擎，最终统计经 Mock 复核；evaluate、confirm、精确检查和报告用 Mock；编译走 P2 的包路径。

## 1. 运行

与 R8e5、R8e7 的冻结协议完全相同，只是代码换成当前版本，并且只跑 mlp：

- 提示词 r8-v4，LLM 设置与冻结 Agent 相同；`--agent-iters 6`，`--tuner on`，精调预算和 `max_sweeps` 不变。
- 数据划分：循环内用留出集 test[9000:10000]；冻结时在确认集 test[8000:9000] 上做一次 `confirm_select`（`MAX_CONFIRM_EVALS=6`）；最终报告用 test[0:8000]，结果不回传。
- 冻结选出的程序走 P2 路径（export 加 compile），读取自举数和 plan_latency。
- LLM 调用上限 8 次（累计约 53/300）。

入口优先用 `experiments/r8b_dev.py --model mlp --tuner on --agent-iters 6` 加上确认和报告阶段。如果当前入口不能一次完成"循环 → 确认 → 编译 → 报告"，可以参照 `experiments/r8e6_reconfirm.py` 的确认和报告流程，新增一个只编排、不改逻辑的脚本 `experiments/r8g_mlp_rerun.py`。不修改任何冻结模块。

结果目录：`pt_result("r8g-mlp-rerun-<时间戳>")`。

## 2. 汇报

和冻结的 R8e7 MLP 并列成一张表：

- 最终程序（每个 act 的方法、阶数、定义域）；
- 预测自举数和实际编译自举数、预测延迟；
- 确认集的 acc_drop 和是否通过；
- 报告集的 top-1 和 acc_drop；
- LLM 调用数、总耗时（其中精调耗时）。

另外报告：
- 新程序的预测延迟与 R8f2 历史重算中"最快可行程序"（4 244 994 µs）的比值；
- 每一轮的候选是否可行，以及被拒绝的原因。

STATUS.md 新增 R8g 一行。

## 3. 冻结标签

本轮**不打 tag**。结果确认后，由用户决定是否打 `pt-agent-freeze-v1.1`：内容为当前代码，MLP 换成 R8g 的结果，其余 3 个模型沿用 R8e7。

## 4. 不要做

- 不修改冻结模块和精调参数；不重跑其余 3 个模型。
- 报告集结果不用于任何选择，也不回传给 Agent。
- LLM 调用不超过 8 次；预算用完就停下汇报。
