# GELU 来源版本适配草稿

已实现默认 GELU 与 `approximate="tanh"` 的静态调用映射、FX 识别和独立数学参考。生产入口仍为当前版本。

| 验收 | 结果 |
|---|---|
| 单元、签名、拒绝及完整性测试 | 20 passed，0 failed/errors/skipped |
| 数学上下文 | 6 个；none、tanh 各 3 个 |
| 明文参考 | 96 组输入、704 个输出值 |
| 原函数参考与固定 Torch 最大差异 | 1.11e-16 |
| 多项式分解最大差异 | 2.22e-16 |
| Hecate、编译、密态执行、API | 0 |

## 来源与近似

默认 GELU 使用 erf；Torch 的 tanh 版本使用包含 `0.044715·x³` 的固定公式。本批两种原函数最大差异为 **0.0001527555**，部分输入已经超过当前密态数值门限。把 tanh 来源改写为默认 GELU 会改变 reference。

来源函数、近似多项式和分解图分别保存并绑定哈希。相同多项式可以用于两个来源，但来源身份及近似误差保持独立。数学参考放在 `gelu_source_reference.py`，没有调用图 planner 计算期望答案。

测试覆盖 functional、模块及 input 关键字写法，三个形状上下文和两种多项式基。多项式在 [-1,1] 上的本批最大近似误差为 **0.008893**。这是测试用显式近似，原模型应用精度尚未认证。

## 实现范围

目录：`scripts/baseline/benchmarks/tools/static_gelu_source_draft_v1/`。

- 仅匹配固定 Torch 函数对象及精确 GELU 模块类，拒绝名称冒充和行为不同的子类。
- 保留 none/tanh、原调用参数、近似区间、系数、图和来源绑定。
- 按实际 Torch 2.0.1 签名检查；拒绝位置参数 approximate、非法版本、out 修改、错误 dtype、超容量和范围外输入。
- 继续使用既有 polynomial 图与分解；没有改后端库或隐藏改写 IR。

这是输入适配草稿，尚未接入生产捕获、生成契约、编译、密态执行和程序包。旧 GELU reference 和旧接口保持。

## 执行与后续

待批准的 r1042 仍为151项及25包复用，5952冻结文件匹配，没有批准收据及后端输出目录。上一轮公开除法/MaxPool草稿的源码与结果保持；本批未重跑原24项或冻结 Benchmark。

下一步在冻结后端的安全边界接入新的来源版本，将 GELU 版本信息贯通到原函数参考、近似精度与程序包接口，再补受影响的后端及 Agent 验收。MHA、原地别名、容量、bootstrap、密文交接与 CUDA 独立推进。

[机器记录](agent-gelu-source-draft-acceptance-r1060.json) · [公开除法与 MaxPool](agent-static-source-ops-draft-acceptance-r1057.md) · [待批准范围](agent-common-pytorch-offline-proposal-r1042.md)

结果目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-gelu-source-draft-r1059`。
报告绑定：`c6182f3b8b06c55ae643f5ee0f1127e5893e16370409f41455ca6b1f3883bf95`。
