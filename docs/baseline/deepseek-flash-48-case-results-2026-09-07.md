# DeepSeek V4 Flash / high：完整48案例重跑结果

日期：2026-09-07。独立新批次，不复用历史候选，不拼接失败子集结果。

## 结论

**Confirmed fact：48/48 最终通过，45/48（93.75%）首次通过；独立证据审计通过，0个审计错误。**

执行后端为 Dacapo 原有 SEAL HEVM CPU，使用真实 CKKS 加密、求值、解密和数值比较。
**Poseidon GPU 未在本批得到验证**。没有 bootstrap、明文模拟或 decrypt-and-reencrypt 替代。

## 配置与复现

- 服务：DeepSeek 官方 API；模型请求及返回身份：`deepseek-v4-flash`。
- 思考强度：`high`；每次请求 `max_tokens=384000`；硬超时900秒。
- 两路并发；每例初次生成后最多3轮候选反馈修复；服务失败不自动重试。
- 使用既有本地凭据，未修改或复制真实密钥文件，未调用两个订阅账号。
- LLVM/MLIR 18.1.2、SEAL 4.0.0；现有固定编译 profile/runtime；门限不变。
- N=32768，16384 slots，14个60-bit key-context模数、13个data-context模数，tc128检查；输入scale为2^40。
- 源码分支 `feat/agent-dsl-correctness`，HEAD `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`。
- upstream `origin/gs/feat-application`；Dacapo固定commit `4616402710f39df3e5f5bd7930a6c036025aaac3`。

实际运行命令（会产生新API费用，不要为了查看报告重复运行）：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_agent_batch.py --live --provider deepseek \
  --model deepseek-v4-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 900 --jobs 2
```

批次证据：`/home/lhy/poseidon-work/results/agent-batch-b1cv_1rd`。
其中有 `report.json`、`audit.json`、`cases.csv`、`summary.md`，以及逐案例证据目录的路径。
不要分享包含 private-keys 的完整结果目录。

独立审计（不调用API）：

```bash
python3 scripts/baseline/audit_agent_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-b1cv_1rd \
  --baseline /home/lhy/poseidon-work/results/fx-batch-yw9qpdtx
```

## 成功率：分母始终保留全部48例

| 指标 | 首次候选 | 修复后的最终结果 |
|---|---:|---:|
| 响应JSON合法 | 45/48 | 48/48 |
| 源码parse | 45/48 | 48/48 |
| 静态检查 | 45/48 | 48/48 |
| 编译通过 | 45/48 | 48/48 |
| 密态执行通过 | 45/48 | 48/48 |
| 数值正确 | 45/48 | 48/48 |

共52次真实API请求，3个案例需要修复，共4轮修复。每例平均修复轮数为0.08333；
仅在需要修复的3例中平均为1.3333轮。没有最终失败、provider超时、token截断、
编译失败或数值失败。4次未通过的候选均在执行前被拒绝：3次响应JSON解析错误、1次静态类型错误。

| 模型族 | 首次通过 | 最终通过 | 结构留出标记 |
|---|---:|---:|---|
| affine | 6/6 | 6/6 | 否 |
| polynomial | 6/6 | 6/6 | 否 |
| linear | 5/6 | 6/6 | 否 |
| mlp2 | 6/6 | 6/6 | 否 |
| mlp3 | 5/6 | 6/6 | 否 |
| fanout | 6/6 | 6/6 | 是 |
| residual | 5/6 | 6/6 | 是 |
| flatten_linear | 6/6 | 6/6 | 否 |

非留出组首次34/36（94.44%），最终36/36；fanout/residual组首次11/12（91.67%），最终12/12。
“留出”指不进入示例，不是模型预训练无污染证明。请求仍包含每例目标图，且固定catalog已被多轮使用；
不能由此宣称对任意未知模型有相同泛化率。

## 实际反馈修复

| 案例 | 失败尝试（从0编号） | 层 | 诊断 |
|---|---:|---|---|
| linear-0 | 0 | response_parse | `Extra data: line 1 column 348 (char 347)` |
| mlp3-1 | 0 | response_parse | `Extra data: line 1 column 30 (char 29)` |
| mlp3-1 | 1 | static_check | `Left operand must be ciphertext; precompute public-only arithmetic` |
| residual-1 | 0 | response_parse | `Extra data: line 2 column 1 (char 239)` |

这些错误没有通过放宽JSON/AST规则解决，而是把诊断反馈给同一模型后生成新候选。
规则转换器答案未作为失败fallback使用，reference、输入、权重、layout、门限和安全参数未改。

## 数值与实际开销

- 192组真实密态输入执行：48例 × 4组（零、有符号固定、固定种子随机、范围边界）。
- 640个输出值全部通过逐项门限 `|actual-reference| <= 1e-5 + 1e-4*|reference|`。
- 按输出元素加权MAE：3.5453965616123294e-9。
- 最大绝对误差：5.879186359969424e-8。
- 非零reference上的最大相对误差：0.000009723937184778592。
- 每例保存cosine similarity与逐项误差/通过标记。近零reference不适合单独按相对误差判定。
- batch记录墙钟：3788.074秒，约63.13分钟。
  包含模型API等待、编译和验证，不是FHE evaluator性能benchmark。

实际token用量（52/52请求均返回有效usage）：

| 项目 | Tokens |
|---|---:|
| Prompt | 66957 |
| Completion（包含reasoning） | 870962 |
| 其中reasoning | 840671 |
| 总计 | 937919 |
| 单次最大completion | 40328 |

单次最大completion来自 `mlp3-1` 的第0次尝试，结束原因为 `stop`，
但它本身因额外JSON内容未通过，说明充分token预算不等于正确答案。
41次请求的completion超过旧8192上限，0次超过65536；没有触及384000。
因此本次证明了完整输出确有超过8K的需求，却没有证明384K优于64K或high优于low。
与历史Pro批次比较时，模型、预算和超时共同改变，不能做单因素因果归因。
实际账单以服务商为准，不把上限乘以调用次数当作实际消耗，也不编造金额。

真实 `linear-1` 数值示例：

- 输入：`[0.5, -1, 0.25, -0.75]`。
- PyTorch reference：`[0.96875, -0.71875]`。
- 解密输出：`[0.9687500008925795, -0.718750001782741]`。
- 两项绝对误差：`8.925794547920418e-10`、`1.7827409548587525e-9`。
- 证据：`/home/lhy/poseidon-work/results/agent-deepseek-ku5nb5sf`。

## 全部案例

| 案例 | 首次 | 最终 | API请求数 | 最大绝对误差 |
|---|---|---|---:|---:|
| affine-0 | 通过 | 通过 | 1 | 3.33199757e-9 |
| affine-1 | 通过 | 通过 | 1 | 5.71868448e-9 |
| affine-2 | 通过 | 通过 | 1 | 5.60435003e-9 |
| affine-3 | 通过 | 通过 | 1 | 6.49626508e-9 |
| affine-4 | 通过 | 通过 | 1 | 3.64228259e-9 |
| affine-5 | 通过 | 通过 | 1 | 2.82295666e-9 |
| polynomial-0 | 通过 | 通过 | 1 | 1.51293950e-9 |
| polynomial-1 | 通过 | 通过 | 1 | 6.01947114e-9 |
| polynomial-2 | 通过 | 通过 | 1 | 1.29116559e-8 |
| polynomial-3 | 通过 | 通过 | 1 | 8.63681873e-9 |
| polynomial-4 | 通过 | 通过 | 1 | 4.25872049e-9 |
| polynomial-5 | 通过 | 通过 | 1 | 2.37827533e-8 |
| linear-0 | 未通过 | 通过 | 2 | 6.78468848e-9 |
| linear-1 | 通过 | 通过 | 1 | 3.20017690e-9 |
| linear-2 | 通过 | 通过 | 1 | 1.02495591e-8 |
| linear-3 | 通过 | 通过 | 1 | 5.11121523e-9 |
| linear-4 | 通过 | 通过 | 1 | 8.19368590e-9 |
| linear-5 | 通过 | 通过 | 1 | 8.15144355e-9 |
| mlp2-0 | 通过 | 通过 | 1 | 1.34057315e-9 |
| mlp2-1 | 通过 | 通过 | 1 | 3.49756579e-9 |
| mlp2-2 | 通过 | 通过 | 1 | 5.95378900e-9 |
| mlp2-3 | 通过 | 通过 | 1 | 2.03589370e-8 |
| mlp2-4 | 通过 | 通过 | 1 | 3.12172244e-8 |
| mlp2-5 | 通过 | 通过 | 1 | 1.30526931e-8 |
| mlp3-0 | 通过 | 通过 | 1 | 1.17298310e-9 |
| mlp3-1 | 未通过 | 通过 | 3 | 2.56896049e-9 |
| mlp3-2 | 通过 | 通过 | 1 | 1.20303639e-8 |
| mlp3-3 | 通过 | 通过 | 1 | 5.25408972e-9 |
| mlp3-4 | 通过 | 通过 | 1 | 1.36320566e-8 |
| mlp3-5 | 通过 | 通过 | 1 | 5.87918636e-8 |
| fanout-0 | 通过 | 通过 | 1 | 2.01302975e-8 |
| fanout-1 | 通过 | 通过 | 1 | 2.07561328e-8 |
| fanout-2 | 通过 | 通过 | 1 | 1.20143576e-8 |
| fanout-3 | 通过 | 通过 | 1 | 2.23013441e-8 |
| fanout-4 | 通过 | 通过 | 1 | 1.42282952e-8 |
| fanout-5 | 通过 | 通过 | 1 | 2.84157039e-8 |
| residual-0 | 通过 | 通过 | 1 | 1.03253641e-8 |
| residual-1 | 未通过 | 通过 | 2 | 6.81776710e-9 |
| residual-2 | 通过 | 通过 | 1 | 1.94164015e-8 |
| residual-3 | 通过 | 通过 | 1 | 1.56011473e-8 |
| residual-4 | 通过 | 通过 | 1 | 1.08652960e-8 |
| residual-5 | 通过 | 通过 | 1 | 8.49359539e-9 |
| flatten_linear-0 | 通过 | 通过 | 1 | 2.84311520e-9 |
| flatten_linear-1 | 通过 | 通过 | 1 | 9.01227018e-9 |
| flatten_linear-2 | 通过 | 通过 | 1 | 6.43002066e-9 |
| flatten_linear-3 | 通过 | 通过 | 1 | 7.28886812e-9 |
| flatten_linear-4 | 通过 | 通过 | 1 | 8.08652290e-9 |
| flatten_linear-5 | 通过 | 通过 | 1 | 9.39815070e-9 |

## 审计确认与测试

审计核对全部48例的模型身份、实际provider配置、冻结文件哈希、逐轮artifact哈希、
权重/输入/reference数组与历史确定性基线的内容一致性、compiler profile、runtime二进制、
门限和汇总指标一致性。每例源码哈希相同，没有运行中修改Agent源码。
这是离线证据一致性检查，不是重新进行52次模型调用，也不是形式化语义等价证明。

- 全WSL回归：173项，164通过、9项Torch依赖测试在普通环境跳过。
- 固定Nix/PyTorch环境：上述9项全部通过。
- 新增订阅provider测试：5项离线通过，覆盖独立凭据、路由、模型前缀、请求参数、header、
  redirect拒绝与响应限额。这不是OpenCode Go或Command Code GOAT账号实测。
- Windows DeepSeek定向回归：42项，40通过、2项Linux专用跳过。
- 当前变更文件无行末空白；tracked diff检查无补丁错误。

Windows侧UNC只读访问被权限边界拒绝，Windows Git submodule辅助脚本也缺少其shell工具；
相同只读检查改用WSL完成。没有为这些诊断修改系统、Git全局配置或项目路径，亦未影响本批执行。

## 本轮本地修改与Git状态

生产脚本：`deepseek_provider.py`、`deepseek_http_worker.py`、`agent_credentials.py`、
`run_candidate.py`、`run_agent_batch.py`、`audit_agent_batch.py`。
测试：`test_subscription_providers.py`（新增）、`test_deepseek_provider.py`、
`test_deepseek_expanded.py`、`test_agent_entry.py`。
文档：provider订阅说明、项目/Agent详细架构、quickstart更新和本报告。

整体working tree还有此前留下的修改；下面不是“本轮新增修改清单”：

```text
 M .gitignore
 M examples/ckks/CMakeLists.txt
 M src/poseidon/frontends/dacapo/hevm_plaintext_encoding.cpp
 M src/poseidon/frontends/dacapo/hevm_plaintext_encoding.h
 M src/poseidon/tests/frontends/dacapo/hevm_plaintext_encoding_test.cpp
 M src/poseidon/tools/dacapo/dacapo-shell.nix
?? .env.example
?? docs/
?? examples/ckks/test_ckks_deterministic.cpp
?? scripts/
?? src/poseidon/tools/dacapo/dacapo-dependencies.nix
?? src/poseidon/tools/dacapo/dependency-lock.json
?? src/poseidon/tools/dacapo/python-wheels.lock.json
```

分支、sparse checkout与Zone.Identifier文件的skip-worktree保持不变。
未commit/push/创建PR：遵循用户本地工作树策略。若之后授权，可将本轮拆成
“provider/配置与测试”和“架构及实验文档”两个commit；不要把此前全部未跟踪目录盲目一起提交。

## 解释边界与下一门禁

**Confirmed fact**：真实模型生成 → Hecate → Dacapo/Earth/CKKS/HEVM → SEAL CPU
密态执行 → 解密差分这条受限路径在本批48例上最终全部通过。

**Evidence-based inference**：对于当前固定小模型范围，程序生成与诊断修复闭环具备
可运行的工程基础。扩大输出预算容纳了超过旧8K上限的完整响应，但没有排除格式/类型错误。

**Unconfirmed**：任意用户PyTorch文件、不同shape和更大模型、自由packing选择、
重复采样后的稳定率、真正未知模型族泛化、两个订阅账号的实际兼容性，以及Poseidon GPU全链路。

确定性规则基线也已48/48通过；本批不能单独证明Agent比规则转换器更好，亦不是研究贡献证明。
下一步应分别推进真实用户模型输入接口/独立留出集，以及Poseidon GPU指令与参数语义对齐。
不扩大为完整ResNet20、bootstrap或multi-GPU性能实验。

相关说明：[Provider订阅接入](provider-subscriptions.md)、
[项目与Agent详细架构](project-and-agent-architecture-2026-09-07.md)、
[使用入口](agent-quickstart.md)。
