# U3b：Agent 用 r8-v5 提示词为 9 个 DaCapo 模型变体生成程序

前置：U3a 已完成（本地提交 4292a9b7）。9 个变体都能导入、生成程序、check 通过，Mock 与张量参考一致。规则规划器的结果作为基线（U3a 报告）：

| 变体 | 规则基线自举数 | 留出集精度下降 |
|---|---|---|
| resnet20_relu | 383 | 0.4pp |
| squeezenet_silu_avgpool | 282 | 0.0pp |
| squeezenet_relu_maxpool | 1122 | 0.5pp |
| mobilenet_silu | 789 | −0.2pp |
| mobilenet_relu | 789 | 0.4pp |
| alexnet_silu_avgpool | 352 | 0.0pp |
| alexnet_relu_maxpool | 1276 | **6.9pp** |
| vgg16_silu_avgpool | 470 | 0.0pp |
| vgg16_relu_maxpool | 849 | **1.6pp** |

本轮让冻结 Agent 的方法（r8-v4 之后的唯一变化是提示词版本 r8-v5，加入新的运算和 maxpool 的近似参数）在这些**未见模型**上运行，得到论文中的通用性结果。

## 1. 协议（与冻结 v1 相同的部分）

- **入口**：U1 的 `--model pt_agent/models/dacapo_builders.py:<name>`。
- **数据**：项目固定的 CIFAR 划分，由 `legacy_cifar` 提供。校准用训练集的冻结选图；循环内用留出集 test[9000:10000]；冻结时在确认集 test[8000:9000] 上确认；最终报告用 test[0:8000]，结果不回传。
- **Agent**：
  - `tune_policy="new_best"`（v1 默认），`max_iters=6`；
  - 提前停止规则不变；
  - 精调参数（LADDER、阈值、预算、`max_sweeps=1`）不变；
  - 确认判据不变（acc_drop ≤ 0.5pp，发散按错误计入），`MAX_CONFIRM_EVALS=6`，并使用 R8h2 的去重；
  - LLM 设置与冻结 Agent 相同（模型、思考模式、max_tokens）。
- **提示词**：r8-v5，由注册表生成，包含 depthwise、avgpool、maxpool（及其近似参数）、concat、展平后接 linear、独立 BN。r8-v4 不改动。

## 2. 引擎配置（用户 10/24 同意报告集用张量引擎评测；本 spec 把同一理由延伸到 r8-v5 的全部评估）

**理由**：
- 张量参考与 Mock 的一致性已经在 R8f 的 46 个程序、U2 的 6 个变体和 U3a 的 9 个变体上验证到 ≤ 1e-9，发散判定口径相同；
- 大模型（AlexNet、VGG16、MobileNet）在 1000 至 8000 张图上做槽级 Mock，耗时以小时计。

**规则**：只对 **r8-v5 的运行**新增 `engine_profile="tensor"`：
- `evaluate`（面向 LLM 的反馈）、精调（包括 R8f2 §9.2 的最终复核）、`confirm_select` 和报告集评测，一律使用张量引擎。
- 张量引擎没有 `boot_input_max_abs`，反馈中该字段为 null。当前 profile 的 `boot_input_bound` 为 None，不影响判据。
- **r8-v4 的冻结路径完全不变**：4 个开发模型和 R9b 仍然使用 Mock，默认值也不变。
- **Mock 抽查**（每个模型的最终程序和规则基线程序都做）：在留出集前 20 张和报告集前 100 张上，Mock 与张量引擎的 logits 满足 `|Δ| ≤ 1e-9·(1+|ref|)`，argmax 和发散集合相同。任何不一致都停下来汇报。抽查结果只用于核对，不回传给 Agent，也不参与选择。

## 3. 运行

- **顺序**：从小到大，依次为 resnet20_relu → squeezenet_silu_avgpool → squeezenet_relu_maxpool → mobilenet_silu → mobilenet_relu → vgg16_silu_avgpool → vgg16_relu_maxpool → alexnet_silu_avgpool → alexnet_relu_maxpool。支持逐模型断点续跑。
- **LLM 调用上限**：本轮合计 50 次（累计不超过 113/300），用完就停下汇报。
- **确认失败**：如果某个模型的确认失败（去重后检查过的程序都没有通过），记为"确认失败"，**不重跑、不换程序**，在汇报中列出被检查的候选及原因。
- **编译**：选中的程序走 P2 路径。常量不超过 1.5 GiB、并且能通过 trusted 运行器资源限制的，本机编译，比较预测与编译的自举数（要求相等），编译后立即 prune；超出的生成服务器作业，标注"本机未编译"。
- **规则基线的报告集评测**：同样用张量引擎在 test[0:8000] 上评测 U3a 的规则程序，以便并列比较。只读，不参与任何选择。
- 编排脚本：在 `experiments/r8h_rerun.py` 的基础上新增 `experiments/u3b_agent_cnn.py`，只编排，不修改冻结模块的逻辑。引擎配置作为参数传入。

## 4. 测试（新增 `tests/test_u3b_*.py`）

1. `engine_profile="tensor"` 时，evaluate、tune、confirm 和报告评测实际调用的都是张量引擎（patch 记录调用）；默认配置下仍然是 Mock，r8-v4 路径不变。
2. r8-v5 提示词包含全部新运算及 maxpool 的近似参数说明；r8-v4 文本逐字不变。
3. 张量引擎下，反馈中的 `boot_input_max_abs` 为 null，其他字段与 Mock 下的字段名相同。
4. 用假 LLM 客户端，在一个小 CNN（含 maxpool 和 concat）上完整跑通一次 r8-v5 的循环：evaluate → tune → confirm → 报告。
5. 既有 455 项测试全部通过。

## 5. 汇报

每个模型一行，与规则基线并列：
- Agent 最终程序摘要：每个近似点（act 和 maxpool）的方法、阶数和定义域的分布；
- 自举数（预测和编译）、旋转数、预测延迟，相对规则基线自举数和延迟的变化；
- 确认集的 acc_drop、报告集的 top-1 和 acc_drop（Agent 和规则基线各一列）；
- LLM 调用数、迭代数、精调次数和耗时；
- 编译与否；
- Mock 抽查结果。

另外汇总：
- 9 个模型中满足"报告集精度下降 ≤ 0.5pp"的个数；
- 自举数比规则基线少的个数及中位数降幅；
- maxpool 模型上 Agent 对 maxpool 近似参数的选择方式，与规则基线的差别。

STATUS.md 新增 U3b 一行；PLAN 中 U3 一行由 Claude 更新。

## 6. 不要做

- 不修改冻结模块的逻辑、精调与确认参数、r8-v4 提示词和默认引擎。
- 不重跑确认失败的模型，不事后换程序；报告集结果不用于任何选择。
- 不放宽资源限制，不下载、不安装。
- LLM 调用不超过 50 次。
