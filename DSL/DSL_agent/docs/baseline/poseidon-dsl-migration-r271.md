# Poseidon 数学上下文后续验收

## 本轮结果

剩余 12 个原模型的紧凑 native helper 表达全部通过静态预检及独立参考。真实 frontend tracing 后，12 项均在 Earth 乘法类型推导处失败，没有密态执行。此前 10 项规则规模阻塞已消除，当前剩余项统一归入编译层。

| 验收 | 结果 |
|---|---|
| 紧凑候选静态检查 | 12/12；AST 2704–2902，展开工作量 562–608，原限制不变 |
| 独立参考探针 | 192 组，每组分别与数学解释器、Torch 比较，最大差异 1.66534e-16 |
| 原模型真实编译 | 0 通过、12 失败、0 skip、0 未运行 |
| 固定参数 RuntimePlan 正例 | 2/2 真实密态、解密与数值比较通过 |
| 固定参数 RuntimePlan 反例 | 检查器拒绝；固定 Runtime 亦拒绝 |
| 累计冻结数学上下文 | 268/280 不同模型、442/459 任务、146/153 三上下文分区 |

候选为人工等价表达，不计真实 Agent 成功。原模型、常量系数、reference、1200 模型集、92 个补充模型及历史记录保持不变。没有修改 Dacapo、Runtime、Poseidon 库或参数。

## 容量结论

十项失败乘法的输入 scale 为 90/60，两项为 90/45。Earth 的消耗 level 11 对应物理 level 1，只有两个既有 60 位 Q 模数；乘积 scale 150 或 135 超出约 120 位容量。此前未密态执行，不能归类为密态数值失败。

最小 RuntimePlan 使用同一固定参数与当前检查器、Runtime、Poseidon CPU，直接验证容量：

| 物理 level | 数据模数数量 | 乘积 log₂(scale) | 结果 |
|---:|---:|---:|---|
| 2 | 3，实际乘积位数 180 | 150 | 通过 |
| 3 | 4，实际乘积位数 240 | 210 | 通过 |
| 1 | 2，实际乘积位数 120 | 150 | 检查器与 Runtime 拒绝 |

两项正例共 8 组输入、16 个输出值；最大绝对误差 8.68871e-10，门限仍为 `1e-5 + 1e-4*abs(reference)`。Runtime 反例退出码 1，诊断为 `value scale exceeds modulus budget`。

这些是手工 RuntimePlan 的后端容量检查，编译调用与 Agent 调用均为 0，不算 DSL 编译通过。它们支持容量诊断，也说明不应保留固定 180 上限；不证明所有等价 DSL 表达都不可行。后续可审查调度或接入经验证的真实 bootstrap，涉及编译器、核心语义或参数的改动须另行批准。

## 剩余案例

~~~text
bench_helper_0032 / 0035 / 0036 / 0039
bench_helper_0040 / 0042 / 0043 / 0044 / 0046 / 0047
semantic_context_331124fd5981bd48a0c2971f
semantic_context_89458ebf15d877d81d6b8b2d
~~~

这些模型含高阶近似最大池化/最大值组合；覆盖账本将部分任务映射到 slice 参数，失败不能直接解释为 slice 不支持。

Poly_Default 的三个补充 w45 编译失败仍保留。真实 bootstrap 的 r234 有符号刷新误差失败未变，r235 独立试验未获新批准。三个小整模型的真实 Agent 提案 r268 仍待批准，共享 512 文件源码未改动。

## 文件与下一步

[逐项报告](poseidon-dsl-migration-r271.json) · [401 分区账本](poseidon-semantic-migration-overlay-r271.json)

原始结果：
- ~/poseidon-work/platforms/aarch64-linux/results/poseidon-math-native-prepare-r269
- ~/poseidon-work/platforms/aarch64-linux/results/poseidon-math-native-r269
- ~/poseidon-work/platforms/aarch64-linux/results/poseidon-scale-capacity-r270
- ~/poseidon-work/platforms/aarch64-linux/results/poseidon-math-aggregate-r271.json

相关 110 项组件测试沿用 r267 的未变化源码证据，本轮没有重复运行。开发探针缺少 WORK 导入的启动错误已修正并单独保留；它发生在后端执行前，不计为 Agent 或 Runtime 失败。

下一步按既有有界提案申请批准后执行真实 Agent 三例或独立 bootstrap 试验。本轮安装、下载、付费调用、commit 和 push 均为 0。x86 原生、空白安装和 CUDA 继续暂缓。
