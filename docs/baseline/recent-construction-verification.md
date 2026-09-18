# 最近新增构造语法：逐项真实 Agent 覆盖汇总

## 已确认结果

四轮冻结构造清单已全部取得有效的真实付费Agent候选，并通过编译、真实密态执行及差分。
本次重新离线审计63个最终有效候选：252组输入、1008个输出值，
最大绝对误差3.3573417601573396e-8，加权MAE5.272248406714349e-9。
未重复付费重跑已成功案例，也不把历史补跑后的最终通过冒称首次通过。

| 契约批次 | 构造范围 | 有效案例 | 逐项观察覆盖 | 详细矩阵 |
|---|---|---:|---:|---|
| v19 | closure/nonlocal、参数绑定、推导式、迭代、循环/else、字符串、公开数值/多项式、对象数组、字典及live views | 30/30 | 117/117 | [构造清单](construction-agent-coverage.md) |
| v20 | 对象数组算术、broadcast、overlap、augassign、密文乘法 | 10/10 | 10/10 | [对象算术](object-arithmetic-agent.md) |
| v21 | float/int、size-one、item及索引、截断、公开对象提取 | 15/15 | 19/19 | [标量提取](scalar-conversion-agent.md) |
| v22 | 一元正负号、np入口、0-D、view/fresh、Plain与混合类型 | 8/8 | 12/12 | [一元运算](object-unary-agent.md) |

这些分母是各版本冻结的观察项，不是上游全部语法项，也不是FHE opcode数量。
不同版本不能简单相加为“完整DSL覆盖率”。v19有34项输出扰动敏感证据，
83项为执行结构证据加源码复核；其余三轮typed清单还检查有限输出影响。
任何有限探测都不是全输入形式化等价证明。

## 最新v22的12项具体落点

| 观察项 | 对应真实生成案例 |
|---|---|
| operator.USub | ou-neg、ou-zero、ou-view、ou-plain、ou-mixed |
| operator.UAdd | ou-positive |
| call.USub | ou-negative-call |
| call.UAdd | ou-positive-call |
| cipher_cells | ou-neg、ou-zero、ou-view、ou-negative-call、ou-mixed |
| plain_cells | ou-plain |
| public_cells | ou-positive、ou-positive-call、ou-mixed |
| boolean_cells | ou-mixed |
| zero_dim | ou-zero |
| input_view | ou-view |
| fresh.USub | ou-view |
| fresh.UAdd | ou-positive |

这里要求构造真的执行、相关类型单元真的影响输出；AST里出现拼写并不足够。
view与fresh、混合类型还要求证据来自同一次操作，避免拼凑不相关事件。
原始失败和修复保留在版本报告中，不人工替换LLM输出。

## 这次新付费部分

额外完成三个尚未有真实Agent证据的模型组合：reshape→BN、Conv1D→reshape→BN、
Conv2D→reshape→BN，首次3/3，3次API，0修复/传输重试，57295 tokens。
详见[逻辑reshape组合](logical-reshape-agent.md)。这部分与上表构造覆盖分开统计：
模型的逻辑reshape可以在lowering中消去，不要求LLM源码里出现reshape拼写。
确定性frontend仍准备布局和公开常量，不声称Agent独立选择任意packing。

所有真实执行证据目前属于Hecate/Dacapo→upstream SEAL HEVM CPU，
不是Poseidon GPU端到端通过，也不是完整DSL目标已完成。

## 可重验的总报告

新增工具 `scripts/baseline/audit_recent_construction.py` 重用四套独立审计器，
重算模型/reference、原请求/候选/产物哈希、执行记录、覆盖矩阵和解密差分。
不会读取API key或发起新的API/FHE运行。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 660s python3 scripts/baseline/audit_recent_construction.py
```

本次生成：
`/home/lhy/poseidon-work/results/recent-construction-audit-_6wfey9q/report.json`

SHA256：`55ec49b375c80d7159ff92ade9f536bb39b3d4696864a8694eb931aaebb7f549`。
cohorts内保留每个版本的feature_matrix、具体有效候选、源码证据、历史失败及原始路径。
该审计只覆盖明确列出的冻结版本，未来新语法不能自动继承这些证据。

新回归 `test_logical_reshape_paid.RecentConstructionPaidTests` 重新执行全部审计
并与保存报告比较；须显式设置POSEIDON_RECENT_CONSTRUCTION_AUDIT，否则skip。
本轮相关回归448项：426通过、22条件跳过、0失败。跳过不算通过。

不改.env、不commit/push、不覆盖原批次。下一步需要扩展边界/组合与不同模型结构，
并继续单独解决高层helper、packing、FHE管理语义和Poseidon GPU执行门禁。
