# Poseidon 算子迁移回归 r245

本轮补测此前没有 Poseidon 证据的 30 个原模型；规则候选通过真实 Hecate tracing、Dacapo 编译、RuntimePlan/spec/bundle 检查、Poseidon CPU 密态执行、解密和逐项比较。

| 本轮验收 | 结果 |
|---|---:|
| 通过／失败／跳过／未运行 | 30／0／0／0 |
| 高层参考与两套明文解释器对照 | 480 组通过 |
| 密态输入组／输出值 | 120／756 |
| 最大密态绝对误差 | 5.00092e-7 |
| 执行批次耗时 | 126.91 秒 |
| 批次及候选产物 | 约 44.8 MiB |
| 付费调用／真实 Agent 生成 | 0／0 |

误差条件仍为 `1e-5 + 1e-4*abs(reference)`。没有修改模型、权重、reference、参数、库、Runtime 或编译器。

## 覆盖范围

| 分区 | 本轮新验收 | r233 历史验收 | 合计模型 |
|---|---:|---:|---:|
| 广播、公开索引、自适应平均池化 | 各2 | 各1 | 各3 |
| LeakyReLU、PReLU、LogSoftmax | 各2 | 各1 | 各3 |
| constant、reflect、replicate padding | 各2 | 各1 | 各3 |
| maximum、minimum、max、min | 各2 | 各1 | 各3 |
| MaxPool1d、MaxPool2d | 各2 | 各1 | 各3 |

15 个分区各有三个模型的 Poseidon CPU 证据，共45个模型；其中15个保留r233原绑定，30个使用本轮绑定。没有将历史结果改标为当前版本通过。清单按模型哈希去重，也不把它们加入原冻结的1,200模型集。

非线性与比较类使用原显式近似。相对原函数的误差单独记录：LeakyReLU/PReLU 最高0.15、LogSoftmax约1.69251e-4、比较与池化最高0.5。这些是小型构造回归，不能替代整网近似精度验收。

## 分解绑定

首次 r244 预检遇到旧分解绑定：注册表版本和实现哈希已更新，尚未编译或密态执行即停止。逐项核对30个模型后，确认新旧数学图、来源映射、能力需求和近似配置一致；r245另存新分解绑定并执行，旧文件和首次停止记录保留。没有放宽完整性检查来接受旧绑定。

本轮未修改共享Python源码，r243三个模型的付费提案源码绑定仍匹配。旧开发脚本 check_operator_regression.py 仍硬编码SEAL配置；其Poseidon改造待已冻结批次的安全边界处理，本轮使用现有Poseidon单例入口。

结果在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-operators-gap-r245`；[机器报告](poseidon-operator-migration-r245.json)包含模型、分区、来源与证据哈希。

下一步：r243真实Agent生成与r235独立bootstrap分别等待批准；全部DSL分区的迁移验收仍未完成。x86原生、空白安装和CUDA继续暂缓。
