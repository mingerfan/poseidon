# 公开除法与 functional MaxPool 适配草稿

已完成 Agent 层的调用映射和明文分解检查。生产入口保持当前版本，草稿尚未接入真实编译和密态执行。

| 项目 | 结果 |
|---|---|
| 单元、拒绝、签名与完整性测试 | 37 passed，0 failed/errors/skipped |
| 数学上下文 | 9 个；公开除法、MaxPool1d、MaxPool2d 各 3 个 |
| 独立参考 | 144 组输入、1,408 个输出值 |
| 图分解最大差异 | 2.776e-17 |
| 固定 Torch 夹具的真实 FX 图 | 9 个；未经过生产导入入口 |
| Hecate / 编译 / 密态执行 / 付费 API | 0 |

## 实现

源码位于 `scripts/baseline/benchmarks/tools/static_source_ops_draft_v1/`。

- 公开除法：识别 Python、Torch 函数与 Tensor 方法；保留原分母，使用可信公开倒数和现有 broadcast 分解。各别名按 Torch 2.0.1 实际签名检查。
- functional MaxPool：按固定函数对象识别 FX 的 boolean_dispatch 目标，处理静态 kernel、stride、padding、dilation；调用既有近似分解。
- 参数与身份：固定错误码及参数位置；拒绝未知调用、重复参数、加密分母、out 修改和篡改绑定。FX 元数据采用受限复制，拒绝对象钩子和循环数据。
- 两套图参考与固定 Torch 原函数分别比较；来源、夹具、参考和资源配置都有哈希绑定。

MaxPool 本轮使用显式测试多项式 `abs(x)≈x²`，相对原始 MaxPool 的最大差异为 **0.312281**。这验证了声明的近似分解，没有认证原模型精度。

## 保留与范围

r1050 的广播失败、r1052 的非法签名误接受、r1054 的 FX 不可变容器拒绝均保留了原日志和源码快照。最终结果属于新绑定，没有覆盖旧记录。

待批准的 r1042 共 151 项、25 份程序包。5952 个冻结文件逐项匹配；648 个后端文件、6 个 reference 文件及历史保护清单保持。当前没有批准收据和后端输出目录。

r870 的 24 项已在 r878 完成：24/24 真实 Poseidon CPU 数值通过，原授权已关闭，本轮没有重跑。

## 下一步

先在当前 r1042 获准范围内完成离线后端回归，再在安全边界接入新输入版本。草稿需要补生产来源捕获、真实编译、密态数值和程序包验收；新功能的真实 Agent 测试另行冻结预算。MHA、GELU tanh 和原地别名语义仍是独立输入缺口。

唯一开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。Linux x86_64 配置保留。框架库、SDK、参数、reference、误差门限及 Git 发布保持。

[机器记录](agent-static-source-ops-draft-acceptance-r1057.json) · [待批准范围](agent-common-pytorch-offline-proposal-r1042.md)

结果目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-source-ops-draft-r1056`。
报告绑定：`2cbcf2e2dc84498acea16d7d3faab81fd84fa5a9b034a9399dddff5afcf2ee1a`。
