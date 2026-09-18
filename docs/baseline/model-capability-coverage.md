# 用户模型能力审计：不要把 DSL 写法、预置模型和自由图混为一谈

## 当前结论

更新：用户批准外发公开模型结构和权重后的 [12 例线上批次](advanced-agent-12-results.md)
已通过，累计独立复核为 126 例（78 用户图、48 历史 catalog），13/13 图算子已有
输出可达的线上证据。下文 114 例报告、旧权限阻塞和回归数字保留为历史记录；
它们没有被改写。当前批次不再处于等待批准状态。

截至本次审计，原有累计 114 个成功 Agent 案例中，**66 个是完整 schema-2/3
用户图描述，48 个是旧 schema-1 catalog 描述**。旧结果仍有效，但只含模型族和
configuration 的目录项不能直接算作“用户提供任意图及权重”的实验证据。

66 个用户图的输出依赖包含 13 类受限图算子中的 12 类，缺少显式 `power`。
这与 Hecate AST 的 29/32 结构分区是不同维度：比如 graph power 可以被正确
翻译为两次密文乘法，不能因为 AST 中出现乘法就宣称已经验证了 graph power。

所有这些数字均为有限输入上的工程证据，不是任意权重、任意深度或全部 DSL
组合的形式化证明。GPU 仍不在当前任务范围。

后续已提供 [完整 96 例显式图清单](self-contained-model-suite.md)，其中旧 48 例
展开后通过新的本地规则 CKKS 批次（48/48）。这解决测试输入对旧模型目录的
依赖，但不改写上面的历史 Agent 66/48 分类，也不增加在线生成成功计数。

完整审计：`/home/lhy/poseidon-work/results/model-capability-audit-zxrvrtlh/report.json`。
SHA256：`a1986a39223f53649d65500e3c6038211ecb6b51b81a5627f228029941ce5366`。审计只读取原来的四条结果 lineage，
不调用 API，也不重新执行 FHE；实际数值和编译产物逐项复核。

## 审计器做了什么

`scripts/baseline/audit_model_capabilities.py` 先复用已有 lineage、冻结输入、
真实编译产物、解密数值及 trace payload 审计，再读取与实际 request 一致的模型：

1. schema-1 原始目录证据单列，保留来源，不重建或假定其用户图定义。
2. schema-2/3 按输出逆向遍历，只有输出可达的模型算子进入能力统计。
3. 记录输入个数/shape、Linear 输入输出宽度与串联层数、broadcast 形式、
   Conv groups/dilation/stride/padding、AvgPool 边界除数规则等。
4. 每个受限图算子关联 DSL 语义清单的支持状态和真实测试入口；代码有新算子
   而映射未更新时测试失败。列出测试入口不等于该审计已经执行这些测试。
5. 保存分析器源码哈希；不修改原模型、权重、门限或历史实验。

模型 dead node 不提高输出可达覆盖；但整个模型仍先过静态检查，不能在死分支
隐藏不支持的算子。输出可达也不保证数值影响，例如乘公开零仍可能消去输入。

## 找出的明确缺口

| 能力 | 人工/规则密态证据 | 当前在线 Agent 用户图证据 |
|---|---|---|
| 1..4 个逻辑输入、六种当前输入 shape | 已有 | 已有 |
| Linear 隐藏宽度 5..8 | 已有并重新审计通过 | 待十案例付费批次 |
| Conv groups=2、dilation=2/3 | 已有并重新审计通过 | 待十案例付费批次 |
| 显式 graph power(2/4) | 本次补全 | 尚缺，不能用 square/multiply 替代计数 |

十个高级案例的固定清单在 `scripts/baseline/cases/advanced-shapes-manifest.json`。
它复用已有人为核验的四个较宽 MLP 和六个分组/膨胀卷积，不更改权重或 reference。
模型 id 只是标识；构造与 lowering 根据数据图和算子进行，不按名称选择实现。

本次启动付费批次被工具安全审批拦截，原因是需要针对这十个模型图结构及固定
权重的数据外发确认。**没有启动新批次，也没有产生 API 调用或新的付费结果。**
准备配置仍为 DeepSeek `deepseek-flash` / high / 384000 tokens / 1200s，API 并发
10、本地 native 并发 2、命令级代理 6478；确认前不尝试其他执行方式绕过审批。

## 本次补全的显式幂次本地闭环

两模型分别是 `0.5*x^2 + 0.125` 和 `0.5*x^4 + 0.125`。
图模型允许的指数仍是 2/4；Hecate 候选中不允许任意 `pow()` 或 Python `**`，
人工程序用密文乘法表达平方和再次平方。

- 独立标量公式、图 reference、Torch float64、规则 DSL、人工 DSL 交叉检查。
- 2/2 正确 golden 通过真实编译/密态执行，2/2 错误幂次程序在解密比较时拒绝。
- 2/2 规则转换器对应程序也通过。
- 人工批次为 16 组输入、64 个输出值，正确程序最大绝对误差
  `1.6808609948348874e-08`，逐项门限保持 `1e-5 + 1e-4*abs(reference)`。
- 错误 x^4→x^2 在 0、±1 上相同，必须依赖固定非整数/随机输入检出，不能只测边界。
- 测试不改模型、不进行 ReLU/SiLU 近似，误差只是针对原明确多项式的实现残差。

人工证据：`/home/lhy/poseidon-work/results/explicit-power-goldens-a3ovzyp7/report.json`，
SHA256 `58cee079ef167d56eed5f75ec96f7396e4bb89554608e2317478f2a13e45b9bd`。
规则证据：`/home/lhy/poseidon-work/results/fx-batch-e_r7dpt6/report.json`，
SHA256 `47291c774a352958b9030f71e75bef59ce50ec24b25424b30c0cad7bc18aa152`。
两批清理临时密钥共 1486415240 bytes；随机密钥原字节不可恢复，可以重新生成
新密钥重跑。输入、权重、DSL、编译产物、解密结果和诊断报告保留。

从 WSL 源码根目录复现本地测试（不调用模型 API）：

```bash
timeout -k 3s 1280s python3 scripts/baseline/run_power_goldens.py
timeout -k 3s 940s python3 scripts/baseline/run_model_batch.py --case-files \
  scripts/baseline/cases/explicit-power-2.json scripts/baseline/cases/explicit-power-4.json
```

`test_explicit_power.py` 用 `POSEIDON_POWER_GOLDEN_REPORT` 和
`POSEIDON_POWER_RULE_REPORT` 审计实际结果；未设置时相应检查跳过，不算通过。

## 最后回归

- 普通 WSL Python 全量：406 项，319 通过、87 按条件跳过。
- 固定 Nix/Torch 环境，启用模型能力报告、power 人工/规则报告、较宽 MLP 和
  grouped/dilated 人工/规则报告：30 项，29 通过、1 跳过。
- 唯一跳过的是尚未启动的十例在线 Agent 测试，不是已完成的密态结果审计。
- `git diff --check` 通过；没有安装依赖、调整安全参数或提交/推送代码。

全量命令（单独执行，避免与其他 Nix 作业争用 launcher.lock）：

```bash
timeout -k 3s 150s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -q
```
