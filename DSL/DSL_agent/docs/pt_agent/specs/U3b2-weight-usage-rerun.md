# U3b2：check 阶段新增"图中权重未被使用"的诊断，并重跑 3 个 ReLU 模型（方案 A）

前置：U3b 已运行（结果目录 `pt_result("u3b-agent-20261024T170907")`），LLM 调用累计 113/300。用户 2026-10-25 选择方案 A。

## 0. 诊断（Claude 于 2026-10-25 复核 U3b）

- MobileNet-ReLU、VGG16-ReLU、AlexNet-ReLU 的大多数迭代都停在 `exact` 阶段，`first_mismatch` 都在第一个 act（act_index 0），偏差为 2.3e-5、1.3e-2、2.4e-4，act 的个数与图一致。
- 原因是 **LLM 漏写了卷积的 bias**：例如 VGG16-ReLU 的程序写成 `conv2d(x, w=W["conv_1_1.Conv2d.weight"], bn="conv_1_1.bn", ...)`，缺少 `b=W["conv_1_1.Conv2d.bias"]`。DaCapo 的卷积在 BN 之前，但 `nn.Conv2d` 默认带 bias。MobileNet-SiLU 通过，是因为那次 LLM 写了 `b=`。偏差小，是因为这些 bias 本身很小。
- 精确检查正确地发现了错误，但反馈只给出了"第 5 行 act 的输入偏差多少"。LLM 连续多轮都没有想到是缺少 bias。**问题在于 Agent 反馈的诊断能力**，与测试工具和 kernel 无关。

## 1. 新增检查：图中权重未被使用（`dsl/typecheck.py`，以及 `planners/agent_tools.py` 中的接线）

- **规则**：程序通过各种方式引用到的全部权重名，与 PT-Graph 的 `graph.weights` 中应当被使用的权重相比，找出缺少的那些。程序中引用权重的方式包括 `W[...]`、`bn="prefix"` 展开出的 4 个张量，以及 `b=`。
  - "应当被使用"的权重，是指 `graph.weights` 中被某个图节点消费的权重。BN 的 `num_batches_tracked` 等不参与前向计算的缓冲区不算在内。规则按"是否被图中节点消费"来判定，不按名字后缀判定。
- **报错**：错误码为 `E_WEIGHT_UNUSED`。每个未使用的权重给出：权重名、消费它的图节点（节点名和类型），以及它在该节点中的角色（例如 conv 的 bias、linear 的 bias、BN 的 running_mean）。如果能定位到程序中对应的语句（同一个节点映射出的 PT-DSL 运算），也给出行号。
  - 示例："`conv_1_1.Conv2d.bias` 未被使用：它是图节点 conv2d `convolution` 的 bias，对应程序第 4 行的 `conv2d`；请加上 `b=W["conv_1_1.Conv2d.bias"]`。"
- **生效范围**：只对 **r8-v5 的运行**（`engine_profile="tensor"` 或提示词版本为 r8-v5）生效，并且作为 typecheck 阶段的错误，先于 exact 检查报告。r8-v4 的冻结路径**不启用**，反馈文本逐字不变。
- 提示词 r8-v5 的文本不改动。这个错误码通过反馈传给 LLM，与其他 typecheck 错误相同。

## 2. 测试（新增 `tests/test_u3b2_weight_usage.py`）

1. **规则程序全部通过**：U3a 的 9 个规则程序、4 个冻结开发程序和 P2 的 7 个程序，在启用检查时都没有 `E_WEIGHT_UNUSED`。这说明一个忠实的程序确实用到了全部被消费的权重。如果有例外，停下来汇报。
2. **能抓住 U3b 的真实失败**：对 U3b 中 MobileNet-ReLU、VGG16-ReLU、AlexNet-ReLU 停在 exact 阶段的 LLM 程序（从 trace 中重放，不调用 LLM），启用检查后在 typecheck 阶段报出 `E_WEIGHT_UNUSED`，并且列出的权重中包含相应的 conv bias。逐个列出每个程序报出的权重。
3. **不影响 r8-v4**：r8-v4 路径上，`evaluate` 的反馈与修改前逐字相同（复用 R8f2 的反馈文本测试）。
4. 构造一个漏写 linear bias 的小程序，以及一个漏写 `bn=` 的小程序，都报出正确的权重和角色。
5. 既有 469 项测试全部通过。

## 3. 重跑（不调用 LLM 的部分先做）

1. **补完 U3b 未完成的部分**：用 U3a 的规则程序，对 AlexNet-ReLU 做确认、Mock 抽查和报告集评测（张量引擎），与其他 8 个模型的规则基线口径相同。
2. **重跑 3 个 ReLU 模型**：MobileNet-ReLU → VGG16-ReLU → AlexNet-ReLU。协议与 U3b 完全相同：
   - r8-v5，`engine_profile="tensor"`，`tune_policy="new_best"`，`max_iters=6`；
   - 确认去重、CIFAR 固定划分；
   - 每个最终程序和规则基线都做 Mock 抽查；
   - 本机能编译的核对自举数后 prune，其余打包服务器作业。
   
   唯一的区别是启用了 §1 的检查。确认失败的模型不再重跑。
3. **不重跑**其余 6 个模型。ResNet20-ReLU 的确认失败、VGG16-SiLU 的报告集未达标，都如实保留。
4. LLM 调用上限：本轮 20 次（累计不超过 133/300）。

## 4. 汇报

- §2 的测试结果，以及第 2 项中每个程序报出的未使用权重；
- **9 个模型的完整总表**：其中 3 个 ReLU 模型标注"U3b2 重跑（加入权重检查后）"，其余 6 个沿用 U3b 的结果。列与 U3b §5 相同：Agent 与规则的自举数、延迟、确认与报告集精度下降、LLM 调用数、编译与否、Mock 抽查；
- 重新计算的汇总：报告集精度达标的个数（Agent 与规则）、自举数少于规则的个数及中位数降幅；
- 3 个重跑模型中，`E_WEIGHT_UNUSED` 被触发的次数，以及 LLM 在下一轮是否修正了；
- STATUS.md 新增 U3b2 一行。

## 5. 不要做

- 不修改提示词 r8-v4/r8-v5 的文本，不修改冻结模块的其他逻辑和精调、确认参数。
- 不重跑其余 6 个模型，不事后换程序；报告集结果不用于任何选择。
- 不放宽资源限制，不下载、不安装；LLM 调用不超过 20 次。

## 6. 补充：张量引擎的内存（Claude 于 2026-10-25 复核 AlexNet-ReLU 规则确认的 MemoryError）

**现象**：AlexNet-ReLU 的规则程序在张量引擎上做确认（1000 张确认图，VAL_BATCH=20）时，在 4 GiB 的限制内申请 2.5 MiB 失败，峰值 RSS 为 3.80 GiB。

**判定**：这是张量引擎的内存占用问题，不放宽资源。**4 GiB 的限制保持不变。** 原因有两处叠加：
- R8f 的前缀激活缓存上限固定为 1.5 GiB；
- 多项式求值（act 和 maxpool 内部的 relu）要同时保留全部 baby-step 的 T_j 以及 giant-step 的中间数组。对 AlexNet 第一层这样的大张量，3×3 maxpool 每个窗口要做 4 次 relu 近似，峰值可达数百 MB。

按 AGENTS.md 的自写错误例外，做以下**通用**修正：

1. **缓存上限随资源限制调整**：前缀缓存上限取 `min(1.5 GiB, 0.2 × 进程地址空间上限)`。4 GiB 时为 0.8 GiB；没有设置上限时仍是 1.5 GiB。缓存逐位相同的保证不变，命中率下降只影响速度。
2. **多项式求值分块**：张量引擎在近似点（act 和 maxpool 内部的 relu）上，沿着张量的展平维度分块求值，使每次调用的中间数组总量不超过 256 MiB。多项式是逐元素运算，分块后与不分块的结果**逐位相同**。卷积、linear、池化求和等非逐元素运算不分块，batch 的划分也保持不变，因此张量引擎与 Mock 的对应关系和 R8f 的结论不受影响。
3. **测试**：
   - 用一个大激活张量对比分块与不分块，`np.array_equal`；
   - 4 个冻结模型的 validate（tensor）与修改前逐项相同；
   - R8f 的缓存测试（6a–6d）照常通过；
   - 记录 AlexNet-ReLU 规则确认在修改前后的峰值 RSS。

修正后继续 §3：先补完 AlexNet-ReLU 的规则基线，再重跑 3 个 ReLU 模型。如果在 4 GiB 内仍然无法完成，停下来汇报峰值和瓶颈所在的运算，不放宽限制。
