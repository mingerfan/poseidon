# R8h3：把精调触发规则改为参数，默认恢复 v1 行为

用户决定（2026-10-24）：冻结仍用 v1（方案 A，tag `pt-agent-freeze-v1`），R8h/R8h2 作为"对每个可行候选都精调"的消融。

问题：R8h 已经把 `planners/agent.py` 的触发条件写死为 `if tuner and eligible`。这样一来，后续 R9b 等实验用默认参数运行时，执行的就不是冻结 v1 的行为。本轮不调用 LLM，不读取报告集。

## 1. 修改 `planners/agent.py`（本 spec 明确授权）

- `run_agent(...)` 增加关键字参数 `tune_policy: str = "new_best"`，可选值为 `"new_best"` 和 `"all"`，其他值报错。
- 触发条件：
  - `"new_best"`：`tuner and eligible and (new_llm_best or not has_tuned)`，与 048f737c 的行为相同；
  - `"all"`：`tuner and eligible`，即 R8h 的行为。
- 规范化去重的精调缓存（`tune_cache`、`tuned_reused_from`）在两种策略下都保留。`"new_best"` 下同一程序不会再次成为新的最好结果，所以缓存不会改变 v1 的行为。
- trace 和 REPORT 中记录所用的 `tune_policy`。
- `confirm.py` 的去重修复保留，不回退。

实验入口（`r8b_dev.py`、`r8h_rerun.py`、`r8h2_reconfirm.py` 等）增加 `--tune-policy` 参数，默认 `new_best`。R8h、R8h2 的编排脚本显式传入 `all`，以便能够复现。

## 2. 测试

1. R8h 的 `tests/test_r8h_tune_all.py` 显式传入 `tune_policy="all"`，断言不变。这属于为参数化做的口径调整，在汇报中列出。
2. 新增测试：`tune_policy` 取默认值时，3 轮回复延迟依次为 100、120、90，只有第 1 轮和第 3 轮调用了 tune（第 2 轮不是新的最好结果），与 v1 的规则相同。
3. 非法的 `tune_policy` 报错。
4. 冻结 v1 行为等价：用 R8e5 的 4 个 trace 作为 LLM 回复，patch 掉 evaluate 和 tune，只比较"哪些迭代调用了 tune"。默认策略下的调用序列必须与 trace 中记录的 tuned 出现位置完全相同。
5. 既有 366 项测试全部通过。

## 3. 汇报

汇报以下内容，并在 STATUS.md 中新增 R8h3 一行：
- diff 摘要；
- 测试结果；
- 第 2.4 条中每个模型的触发序列对比。

## 4. 不要做

- 不改动其他冻结模块，不重跑任何模型，不调用 LLM。
