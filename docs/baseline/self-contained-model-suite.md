# 完整模型图测试集：不再依赖预置模型编号

## 输入能力与证据边界

新测试集 `scripts/baseline/cases/user-graph-suite-v2.json` 包含 96 个完整
schema-2/3 模型图，覆盖既有 16 个模型族、每族 6 个配置。所有计算连接、
输入 shape 和公开固定权重都保存在 JSON 中；模型 id 只是标识。
使用者可以复制为自己的新图文件，在受限算子、shape、数值范围内修改数据，不需要修改
`CatalogModel` 或增加新的 family/configuration 分支。

这不是新增 16 个模型族，也不是把旧 Agent 报告改名为新实验。
旧 `restricted-model-suite-v1`、48 个 schema-1 描述和所有历史报告保持原样。
新输入接口不扩大单个图的安全参数、数值门限或底层支持范围。

三份源码数据文件的职责：

| 文件（位于 scripts/baseline/cases） | 内容 |
|---|---|
| `user-graph-suite-v2.json` | 96 个可直接交给规则或 Agent 批量入口的显式图 |
| `legacy48-user-graphs.json` | 原八类各六例展开后的 48 个图；用于单独验证迁移 |
| `user-graph-suite-v2.provenance.json` | 模型族、原描述、原描述哈希、展开图哈希与整个图集合哈希 |

来源文件中的 family 是测试分类元数据，不进入模型构造或 lowering 的选择逻辑。
原有后 48 个 schema-2/3 图保持内容不变；前 48 个使用新的 `usergraph-` id，
避免把不同输入描述当成同一个历史运行。

## 转换是如何实现的

`scripts/baseline/catalog_graph_migration.py` 是一次性兼容导出器，不是 Agent，
也不是接受任意用户 Python 代码的新 frontend。它读取原可信 PyTorch 模型的
实际 tensor 数值，输出明确的算子节点和常量数组，不独立重新猜测随机权重。
只有这个兼容导出器按旧 family 分派；导出结果执行时不查询旧模型目录。

例如 Linear 使用实际矩阵和偏置；MLP 保留 Linear 的连接和中间 square；
fan-out 保留两支计算与合并；residual 保留输入旁路。
不存在把跨元素 Linear 偷换为逐元素乘法、自动近似 ReLU 或缩短深度的操作。

离线测试会禁用 `CatalogModel` 后构造新图、改名、执行 reference 和规则 lowering，
并用每例四组原输入及四组新固定种子输入检查原模型与新图的一致性。
另有改变用户矩阵元素、添加 negate 节点的测试，确认图数据决定实际计算。

## 同输入的规则与 Agent 对照

两个入口现在接受同一份 `--case-manifest`。必须从 WSL 源码根执行。

仅验证 Agent 批次计划，不读取凭据、不调用 API：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan \
  --case-manifest scripts/baseline/cases/user-graph-suite-v2.json
```

本地确定性迁移验证（会运行真实 CKKS，不收费）：

```bash
timeout -k 5s 3700s python3 scripts/baseline/run_model_batch.py \
  --case-manifest scripts/baseline/cases/legacy48-user-graphs.json
```

完整 96 例也可以把同一个 manifest 交给规则入口；不必为比较修改输入结构。
付费 Agent 运行需要明确授权公开模型结构及权重外发，不能把 `--plan` 通过
当成已经生成程序。后续默认 DeepSeek `deepseek-flash`、high、384000 tokens、
1200 秒、API 并发 10，本地 native 并发 2；命令级代理端口 6478。
本次迁移不触发任何付费调用。

规则入口验证原始清单哈希，在结果目录保存独立 JSON 快照并冻结其哈希；
快照是相同数据的重新序列化，不宣称与源文件逐字节相同。
模型、权重、reference、DSL 和编译产物继续单独冻结。
清单不能与 `--smoke`、`--unit-tests` 或 runtime worker 参数混用。
源文件在隔离入口前改变、快照被改动/删除/无法读取均判失败，不能静默少测案例。

## 验证范围

`test_catalog_graph_migration.py` 和 `test_custom_batch_manifest.py` 检查：

- 完整图的静态允许/拒绝规则、重复字段、重复 id、非法数值和不支持的算子。
- 96 个图及来源文件可由当前固定依赖重建；旧清单不变。
- 原 PyTorch、独立图 reference、新 PyTorch 与规则 DSL 明文解释结果一致。
- 计划模式读取精确的显式图，不访问预置目录或凭据，API 调用数为零。
- 设置 `POSEIDON_MIGRATED_RULE_REPORT` 后检查真实迁移批次的冻结输入、
  编译产物、原始模型 reference、解密数组、逐项误差和密钥清理。

未设置真实报告的条件测试会跳过，不能算成密态通过。
本地规则测试也不能替代新的在线 Agent 证据：原有累计 Agent 报告中的
66 个用户图和 48 个旧目录描述仍需分开统计，不能重新标注历史生成成功率。

所有激活保持原明确的 square 多项式，所以迁移没有引入模型近似误差；
CKKS 执行误差只与原模型的独立 reference 比较。另行批准的激活近似仍需
分别保存原激活、近似多项式和密态执行三者结果。

## 本轮真实密态结果

报告：`/home/lhy/poseidon-work/results/fx-batch-t0alnl8z/report.json`。
SHA256：`868bebe4148a12cb8e6467867692ae52302883c21f90098283520e52fb6cc6a4`。

- 迁移的 48/48 显式图全部通过，八类各六例，无遗漏或失败。
- 每例四组输入：零、有符号非整数、固定种子随机、声明范围边界；
  共 192 组密态输入、640 个逐项输出。
- 最大绝对误差 `3.555069133520661e-08`；全部输出加权 MAE
  `4.6578292264333e-09`；最大非零 reference 相对误差 `8.372780840982043e-06`。
- 最小逐案例 cosine similarity `0.9999999999999482`。
- 门限保持 `1e-5 + 1e-4*abs(reference)`，没有因为测试结果而放宽。
- SEAL 4.0.0、N=32768、16384 slots、tc128、14 个 60-bit 模数（含特殊模数），
  初始 data modulus 数 13；rotation keys 为 1 和 2；不执行 bootstrap。
- 保存的运行源文件哈希全部与当前对应实现相符。编译使用 `--verify-each`，
  真实产物通过 HEVM/CST 门禁；解密比较不是明文模拟。
- API 调用数为 0。这是新图描述的规则转换基线，不是新增 48 个 Agent 成功。
- 已清理本批次五个可再生密钥文件，释放 297283048 bytes，保留清理审计。
  原随机密钥字节不可恢复，未来重跑会重新生成密钥；全部模型和结果证据保留。

这证明指定输入与模型上的迁移及执行一致，不是对任意权重/深度的形式化证明。
在线 Agent 的较宽 MLP、分组/膨胀卷积等新覆盖仍需独立付费实验证据。

回归记录：

- 普通 WSL Python 全量：413 项，322 通过、91 按依赖/证据条件跳过。
- 固定 Nix/Torch 环境，显式启用迁移批次、既有 power 人工/规则和模型能力
  审计报告：62 项，61 通过、1 跳过。唯一跳过项是待授权的高级模型在线 Agent。
- 新 48 例实际报告的条件测试已经启用并通过，不包含在跳过项中。
- 拒绝测试打印的非法 `--jobs 11` / `--max-repairs 4` 信息和模拟 transport
  retry 信息是预期反例，不是本批次网络或编译失败。
- `git diff --check` 通过；不安装依赖，不调整数值/安全参数，不提交或推送。
