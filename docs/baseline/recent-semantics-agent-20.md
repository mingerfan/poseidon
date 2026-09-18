# 本轮真实 Agent 生成测试：20 个模型

## 结论与范围

Confirmed fact：20/20 模型经过真实 DeepSeek API 生成、受限 AST 检查、真实 Hecate tracing、Dacapo 编译、SEAL CPU 密态执行和独立 reference 差分比较通过。19/20 首次通过；1/20 经模型反馈修复一次后通过。

本批是一次新的独立生成实验，不是 golden 源码回放；部分模型沿用既有定义，不能称为全新的 unseen 测试集。规则转换器生成的源码保存在本地证据目录，但不提供给 Agent 作为答案。本文不把它作为本批额外执行过的规则转换器对照实验。

本批**不证明全部 DSL 语义覆盖**，也**不验证 Poseidon GPU**。正式任务契约为 `hecate-function-synthesis-v18`，AST 契约为 `hecate-function-v17`；object-array 能力在允许集合内，但本批模型没有选用它。上一轮 public mappings 扩展尚未接入此正式入口，本批未测试该能力。

## 配置与安全边界

- 服务商：DeepSeek 官方接口；请求模型及响应允许身份：`deepseek-flash`。
- reasoning：high；单次输出预算上限：384000 tokens。
- API 并发 10；密态编译/执行并发 2；单次请求超时 1200 秒。
- 每例最多 3 轮反馈修复；瞬时传输失败最多 3 次重试。本次实际网络重试 0。
- 凭据仅从项目本地 `.env` 加载，或遵循已有 loader 的显式环境变量优先规则；不记录密钥值。
- 出站内容：公开合成模型图、固定公开权重、DSL 规则、布局和白名单诊断。修复时可发送汇总误差，不发送 reference/输入/解密输出向量或测试密钥。
- LLVM/MLIR 18.1.2；SEAL 4.0.0；N=32768；14 个 60-bit 模数；tc128 参数检查。
- 无 bootstrap、无 GPU；不改变安全参数或算法语义。
- 判定固定为逐项 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 20 个候选配置完全一致，测试期间 producer source hashes 只有一组。

## 汇总结果

| 指标 | 结果 |
|---|---:|
| 计划 / 完成模型 | 20 / 20 |
| 首次 JSON / Python 解析 | 20/20 / 20/20 |
| 首次静态检查 / 编译 / 密态执行 / 数值正确 | 各 19/20 |
| 修复后数值正确 | 20/20 |
| 实际 API 调用 | 21 |
| 平均修复轮数 | 0.05 |
| 传输重试 / provider 错误 | 0 / 0 |
| 密态输入执行组数 | 80 |
| 输出标量比较数 | 244 |
| 按输出数加权 MAE | 2.052615618e-7 |
| 最大绝对误差 | 6.823009582e-6 |
| prompt / completion tokens | 128836 / 127461 |
| total tokens | 256297 |
| 批次 elapsed | 182.589 秒 |

21 次调用均有 usage 回执；金额以服务商账单为准，未估算费用。时长包含本批生成/执行编排，不是 FHE 性能基准。

## 逐模型结果

| 模型 ID | 结果 | MAE | 最大绝对误差 | 输出数 |
|---|---|---:|---:|---:|
| custom-wide-mlp-5 | 首次通过 | 2.657959e-9 | 5.038410e-9 | 8 |
| custom-wide-mlp-6 | 首次通过 | 3.193824e-9 | 9.353146e-9 | 8 |
| custom-wide-mlp-7 | 首次通过 | 7.188672e-9 | 1.483766e-8 | 8 |
| custom-wide-mlp-8 | 修复一次通过 | 2.790616e-9 | 5.185407e-9 | 8 |
| conv1d-dilation2 | 首次通过 | 4.641543e-10 | 1.401461e-9 | 8 |
| conv1d-dilation3 | 首次通过 | 4.053838e-9 | 5.544508e-9 | 4 |
| conv2d-dilation2 | 首次通过 | 2.329040e-9 | 5.248169e-9 | 8 |
| conv1d-groups2 | 首次通过 | 1.260786e-9 | 4.062320e-9 | 16 |
| conv2d-depthwise-multiplier | 首次通过 | 3.061992e-9 | 9.406765e-9 | 16 |
| conv2d-grouped-dilated-pad | 首次通过 | 8.110882e-10 | 1.736238e-9 | 8 |
| explicit-power-2 | 首次通过 | 2.013375e-9 | 9.267592e-9 | 16 |
| explicit-power-4 | 首次通过 | 5.079008e-9 | 2.424691e-8 | 16 |
| arithmetic-alias-chain | 首次通过 | 3.609936e-9 | 9.972734e-9 | 16 |
| construction-linear | 首次通过 | 4.187444e-9 | 9.931446e-9 | 8 |
| construction-sumslots3 | 首次通过 | 2.093674e-6 | 6.823010e-6 | 16 |
| construction-sumslots4 | 首次通过 | 9.702889e-7 | 1.527086e-6 | 16 |
| construction-genpoly3 | 首次通过 | 8.462419e-9 | 3.791683e-8 | 16 |
| construction-genpoly5 | 首次通过 | 9.336488e-9 | 2.829112e-8 | 16 |
| construction-genpoly7 | 首次通过 | 1.351022e-8 | 4.422880e-8 | 16 |
| construction-genpoly-even | 首次通过 | 7.116897e-9 | 2.254740e-8 | 16 |

四组输入沿用固定 reference 测试规范，未回填解密中间值。旋转求和的误差明显大于其余模型；本次均通过门限，但不能从单次实验推断长期稳定性或其精确误差成因。

## 唯一失败与真实反馈修复

模型：`custom-wide-mlp-8`。

- Symptom：第一次生成在 static_check 层被拒绝。
- Evidence：`Public arrays are read-only; augmented array writes are not implemented`。
- Cause：候选先令 `out0 = b_out0`，公开 bias 是只读数组；随后 `out0 += h[j] * w_out0[j]` 触发公开数组增强赋值，而不是合法的密文累加。
- Repair：DeepSeek 接收到诊断后重新生成：从 `acc0 = sq[0] * w0[0]` 的密文值起步，后续普通加法累加，最后加 bias。
- Result：第二次候选编译、密态执行、数值比较全部通过；最大绝对误差 5.185407298e-9。

失败、诊断与修复源码完整保留在：
`/home/lhy/poseidon-work/results/agent-deepseek-rr5p53_k/attempt-00` 和 `attempt-01`。

这说明本例诊断反馈修复实际工作，但不是对任意错误的修复保证。

## 实际新语法覆盖：不能把允许集合当作已测试集合

从冻结 trace payload 重新检查得到以下非零构造计数：

- `function_definitions`：3/20，custom-wide-mlp-5、custom-wide-mlp-7、custom-wide-mlp-8。
- `helper_calls`：3/20，custom-wide-mlp-5、custom-wide-mlp-7、custom-wide-mlp-8。
- `closure_instances`：2/20，custom-wide-mlp-5、custom-wide-mlp-8。
- `loop_iterations`：1/20，custom-wide-mlp-8。
- `container_writes`：1/20，custom-wide-mlp-8。
- `iterator_advances`：1/20，custom-wide-mlp-8。
- `public_numeric_operations`：1/20，conv1d-dilation3。

这些是构造阶段观察，不是全语义证明；闭包实例计数不证明覆盖 nonlocal 或 late-binding 边界。上述 public_numeric_operations 的出现也不等于覆盖全部数组计算。

以下能力在本批 20 个最终通过候选中未实际观察到：nonlocal、默认/关键字/星号参数展开、comprehension、lambda/sorted、序列切片、公开数组运算、while/break/continue、membership/short-circuit、字符串方法、Chebyshev 对象运算、object-array/Empty 操作。模型是多项式，**不代表生成代码使用了 Chebyshev 或 GenPoly helper**；本批对应候选直接构建乘加表达式。

现有基础 SSA 覆盖审计记录 18/32 个基础分区；这 32 个分区是旧的基础集合，**不是当前 v18 全语义分母**。完整新语法 Agent 覆盖需要独立的定向生成实验与输出使用证据，不能把本批成功率改写成“全部语义通过”。

## 证据与审计

- 原始批次：`/home/lhy/poseidon-work/results/agent-batch-__nl6y0c/report.json`
- 批次 SHA-256：`eda6fd4a1916ecb1c1bedcd3e2eb2d9283ba4ca9a557e35c15bd04508406855d`
- 离线数值/语法审计：`/home/lhy/poseidon-work/results/dsl-grammar-audit-6_7zfo96/report.json`
- 审计 SHA-256：`f8525016717785cbcbfabc63d2ebf26dbf2346a78423c4f84662927bc69c5701`
- 审计对成功候选重算差分，核验 frozen inputs、trace payload、compiler artifacts、SEAL 参数、真实密态执行、模型身份及密钥检查证据；审计新增 API 调用和密态执行均为 0。
- 所有 candidate DSL、Earth/CKKS IR、HEVM/CST、输入/reference、解密输出和日志保留在各行 evidence 指向的目录。
- 20/20 运行均有完整密钥清理回执，共删除 13579600200 字节（约 12.65 GiB）的随机临时测试密钥。原随机密钥不可恢复，可重新生成新的测试密钥；没有删除上述结果证据或旧批次。

## 复现命令

以下 live 命令会再次产生 API 费用，并建立新的结果目录；不要为了阅读结果而运行。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 15s 48000s env PYTHONDONTWRITEBYTECODE=1 python3 \
  scripts/baseline/run_agent_batch.py \
  --deepseek \
  --case-manifest scripts/baseline/cases/recent-semantics-agent-20-manifest.json \
  --object-arrays --provider deepseek --model deepseek-flash \
  --reasoning-effort high --max-tokens 384000 --api-timeout 1200 \
  --provider-retries 3 --loopback-proxy-port 6478 --jobs 10
```

只审计已保存结果（不付费）：

```bash
timeout -k 5s 280s env PYTHONDONTWRITEBYTECODE=1 python3 \
  scripts/baseline/audit_dsl_coverage.py \
  /home/lhy/poseidon-work/results/agent-batch-__nl6y0c/report.json
```

离线清单回归：

```bash
timeout -k 3s 60s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest test_recent_agent_manifest test_custom_batch_manifest -q
```

结果：13/13 通过。清单测试仅证明样本描述保持一致、入口参数正确，不能代替上面的真实 Agent/FHE 证据。

## 本轮本地变更与下一门禁

仅新增 20 例数据清单、对应离线清单测试、本报告；未修改 provider、compiler、runtime 或现有用户修改。分支仍为 `feat/agent-dsl-correctness`，HEAD 仍为 `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`。不 commit、不 push。

下一门禁是对尚未观察到的新增构造语法做定向 Agent 生成覆盖（正式契约冻结、输出使用检查和同一密态差分链）；public mappings 先补齐正式入口，再纳入该类测试。Poseidon GPU 及完整 DSL 支持仍未因此完成。
