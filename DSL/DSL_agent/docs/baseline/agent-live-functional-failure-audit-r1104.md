# 本轮真实 Agent 验收与失败定位

截至 2026-10-07T13:21:16.872141+08:00，原 76 项批准批次仍在运行。该文档记录已关闭的 15 项组和当前快照，不作为全部 76 项的完成报告。

| 已关闭的模型组 | 项数 | 数值通过 | 失败 |
| --- | ---: | ---: | ---: |
| 权重包接入：MLP、Conv→Pool、残差、BatchNorm、矩阵融合、形状→Linear | 6 | 6 | 0 |
| functional LayerNorm | 3 | 0 | 3 |
| functional GroupNorm | 3 | 2 | 1 |
| functional variance | 3 | 2 | 1 |
| 合计 | 15 | 10 | 5 |

10 项是本轮 DeepSeek 返回的原始源码：8 项首次通过，2 项修复后通过。全部通过真实 Hecate tracing、Dacapo-modified 编译、RuntimePlan 及常量检查、Poseidon CPU 密态执行与独立 reference 比较。10 份程序包均重载并执行 3 组新输入，复用未调用模型或编译器。

通过项共 70 组原始及新输入、378 个输出值，最大绝对误差 1.9341682254425674e-7。门限保持 `1e-5 + 1e-4*abs(reference)`。程序包 263 处和所有相关编译产物 706 处哈希核验一致。15 项组新用 33 次生成／33 次 HTTP；此前 TLS 失败 1／1 保留，总计 34／34，没有未决 HTTP。

## 失败定位

| 原案例 | 最终生成源码中的问题 |
| --- | --- |
| `pytorch_functional_layer_0` | 将单槽 `mask1` 当成全元素权重、将 `mask2` 当成零偏置，导致只保留一槽并在另一槽加 1。 |
| `pytorch_functional_layer_1` | 行内归约后的广播位置错误；最终轮还重复广播已广播的因子。 |
| `pytorch_functional_layer_2` | 前三轮使用未绑定的模型权重名称；最终轮直接返回输入，遗漏归一化、归约和 stack。 |
| `pytorch_functional_group_0` | 将 `pv0` 的单槽归约权重误当成全槽标量，又把 `mask0/mask1` 当成模型增益与偏置。 |
| `pytorch_functional_variance_1` | 跨块、跨轴的均值广播与输出排列不符合模型；最终输出第二个有效槽接近零。 |

这 5 项每例均用尽原定 4 次生成，不自动使用其他案例剩余额度补跑。20 轮候选中，5 轮静态拒绝、15 轮真实编译并执行后数值失败。静态拒绝复核为 4 次未定义名称和 1 次 `np.array(..., dtype=np.float64)`；当前契约只允许 `dtype=object` 的 Expr 容器。

为区分生成与后端问题，开发审计器对原候选做受限 AST 数学解释，仅支持直线算术、公开常量和 rotation，不执行上传的 Python。15 轮全部在原输入上证明：源码数学结果不满足模型 reference，而真实密态输出满足该源码数学结果的冻结门限。源码相对模型的最大偏差为 0.2926092487–1.5813776741；密态相对源码数学结果的最大差异为 1.0717025710e-7。

因此，这些案例的主要故障是生成的模型计算不正确。此结论限定于保留的测试输入，不宣称编译器在所有程序上正确。该数学审计是开发诊断，不计为新的编译、密态执行或 Agent 成功。

## 下一批修改

当前共享生产源码、后端与 reference 保持冻结。批次关闭后，优先在 Agent 层处理：

1. 在公开生成上下文中明确区分模型常量名称、可访问的 DSL 常量与 packing 掩码，按实际形状提供归约后广播和输出排列约束。
2. 将静态拒绝中的未绑定名称、受限构造和稳定 AST 位置转成受控反馈；当前泛化的“违反契约”不足以让模型修复具体错误。
3. 保留数值失败的受控反馈；增加可确认的重复候选识别，不向模型发送测试数组、reference 或人工修复源码。
4. 在新源码绑定下先做相关单元与无收费兼容检查，再为原 5 项准备新的有界真实 Agent 提案。当前 76 项已批准范围继续运行。

## 当前运行快照

矩阵与 InstanceNorm 60 项组：27 通过、0 失败、1 项运行、32 项未启动。Attention decode 1 项尚未启动。整体为 37 通过、5 失败、1 运行、33 未启动；累计预留 68 次生成／68 次 HTTP，包含此前 TLS 失败。

绑定：`81c396aeb46344fdce88d746ea781e784afd0647b58d37ff848027028d57c01e`。695 份生产源码、3 份执行工具及 698 份保留证据均匹配。API／密态串行、编译 2／链接 1，原时间与空间边界保持。

本轮仅新增开发审计工具和文档；没有安装、下载、参数调整、后端修改、提交或推送。Bootstrap、CUDA、完整 PyTorch 任意输入、应用精度与规模扩展仍分别待验收。

原始证据位于 ARM Ubuntu：

- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-pending-continuation-r1099`
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-functional-source-audit-r1102/report.json`
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-functional-source-audit-r1102/delivery-audit.json`

审计入口：`scripts/baseline/benchmarks/tools/closed_live_source_audit_v1/`。8 项诊断器正反例通过，0 failed/skipped。审计新增 API、编译、密态执行均为 0。
