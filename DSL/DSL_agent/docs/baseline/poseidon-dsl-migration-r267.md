# Poseidon 数学上下文迁移

## 结果

ARM Ubuntu 的 Poseidon CPU 执行链已完成本轮补测。冻结模型与数值门限保持原样，候选来自规则生成或人工等价表达；本轮 API 调用为 0。

| 范围 | 结果 |
|---|---|
| 既有 DSL 构造 | 627 任务、209 分区，每分区三个上下文；沿用 r261 的明确来源绑定 |
| 冻结数学上下文 | 268/280 个不同模型、442/459 个任务有 CPU 数值证据 |
| 数学分区 | 146/153 完成三个原定上下文；其余 7 个分区仍有受阻任务 |
| 本轮新增模型 | 196/208 通过；2 个编译失败、10 个规则预检阻塞 |
| 相关静态与组件测试 | 110 passed，0 failed，0 skipped |
| 应用与程序包 | 6/6；5 项数值验收、1 项编译验收，均通过重载及缓存复用 |

208 个原定模型先经过 3328 组双参考检查。196 个通过模型完成 784 组真实密态输入、4732 个输出值比较；最大绝对误差 1.7037987e-06，加权 MAE 1.1706624e-08。每例仍执行四组输入、逐项解密比较，门限为 `1e-5 + 1e-4*abs(reference)`。

这份统计汇总多个明确源码绑定，不把旧结果改写成当前绑定。规则/人工候选的通过数不计入真实 Agent 生成结果。原 1200 模型、92 补充模型与全部失败记录保留，未重新生成模型集。

## 本地修正

无明文常量的 RuntimePlan 原先被布局适配器附加空 bundle 声明，Runtime 按协议拒绝。现在仅在存在 bundle Encode 时声明 `plaintext_bundle`；空辅助 manifest 保留用于路径与程序包兼容。检查器同步核验声明与使用关系。

修改发生于所有运行批次结束后，原 512 文件源码快照保存在 `poseidon-bundle-adapter-boundary-r264/core-source-before.zip`。5 个原失败模型在新绑定下全部通过。另通过紧凑旋转、归约及精确平衡 Chebyshev 表达解决 4 个原模型；没有更改系数或提升 AST/工作量限制。

应用回归包括 MLP、RMSNorm、Attention decode、native 返回构造、编译等级 MLP 和无常量模型。每个包重新加载时生成与编译调用均为 0，缓存 Provider 调用为 0。临时测试驱动的选项传递与请求文件哈希比较错误已经修正，失败目录仍保留。

## 剩余问题

- `bench_helper_0040`、`bench_helper_0044`：原近似最大池化的三个高阶 Chebyshev 组合。平衡表达在 w45、w40 下仍被 Earth 的消耗层数/scale 容量规则拒绝，没有密态执行。
- `bench_helper_0032/0035/0036/0039/0042/0043/0046/0047`、`semantic_context_331124fd5981bd48a0c2971f`、`semantic_context_89458ebf15d877d81d6b8b2d`：规则候选超过 AST 或工作量边界。两套数学参考已通过，尚需紧凑表达后再判断编译与执行。不能据此宣称 slice 或 Agent 本身不支持。
- Poly_Default 的三个补充 w45 用例继续保留编译失败。此前“输入 scale 不一致”的表述不准确：Earth Mul 没有要求两个输入 scale 相等，实际检查为 `level*60 + lhs.scale <= 12*60`。Earth level 11 映射到物理 level 1，只有两个 Q 模数，输入 scale 90/75 的乘积 scale 165 超出约 120 位容量。编译器、Runtime 及参数均未修改。
- 真实 bootstrap：r233 批准已经执行为 r234，有符号输入刷新误差超限。r235 的独立 ratio128 试验仍待批准，未运行。
- Poseidon 上的真实 Agent 三整模型评测尚未执行。适配器修正后已重新冻结 r268，替代未执行的 r256；模型、公开请求与后端二进制未变。

完整逐项状态见 [JSON](poseidon-dsl-migration-r267.json)，401 分区叠加账本见 [覆盖账本](poseidon-semantic-migration-overlay-r267.json)。数学原始证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-math-aggregate-r267.json`。

## 下一步

先验收有界真实 Agent 的 MLP、RMSNorm、Attention decode，或获准后执行 r235 独立 bootstrap 试验。新付费提案为 [r268](poseidon-whole-models-live-proposal-r268.json)。高阶最大池化的表达/编译深度问题单列，不放宽模型、参数和门限。

本轮没有安装、下载、付费调用、提交或推送；x86 原生执行、空白机器安装和 CUDA 继续暂缓。
