# Logical reshape：Conv/BatchNorm 组合与真实 Agent 验证

## 结论与证据边界

冻结的三个新增模型首次生成全部通过：真实 DeepSeek API → Hecate Python →
Dacapo Earth/CKKS → HEVM/CST → upstream SEAL HEVM CPU 加密执行 → 解密差分。
不是 Poseidon GPU 结果；不是所有输入上的等价证明，也不是任意 packing 的证明。

本次模型图支持 C-order 逻辑 reshape：静态 rank 1..4、总元素数最多8，
允许一个 -1 推断维度，元素总数必须保持。FX 支持 torch.reshape 和 Tensor.reshape
的静态 tuple/list/多位置参数形式。它仅更新逻辑 shape，保留 packed 或 scalar-neuron
表示与线性元素顺序，不引入密态指令。transpose、permute、view 别名语义、
数据相关 shape、重新 packing 不在这次支持范围。

必须区分三个对象：PyTorch 模型的逻辑 reshape、公开对象数组的构造 reshape、
密文 slot 重排。本次验证第一个，不能冒充后两个的完整支持。
Agent 的请求仍由确定性 frontend 准备公开常量和布局；不声称 Agent 独立完成
布局选择。本次生成源码可能没有 reshape 字样，因为它已在逻辑布局处理中消去。

## 固定模型与逐项结果

| 模型 | 输入及组合 | 最大绝对误差 | 生成次数 |
|---|---|---:|---:|
| reshape-bn | [4] → [1,2,-1] → 固定统计 BN → flatten | 1.1981366565549934e-8 | 1 |
| reshape-conv1-bn | [1,4] → stride2 Conv1D → [1,2,-1] → BN → flatten | 2.51501224290962e-8 | 1 |
| reshape-conv2-bn | [1,2,2] → Conv2D → [1,1,2,2] → BN → flatten | 5.0747147040119955e-9 | 1 |

每例四组输入：零、有符号、固定种子随机、边界；合计12组、48个输出值。
加权 MAE 4.679983389247837e-9；逐项门限保持
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
API 3次，修复0次，传输重试0次；prompt 21741、completion 35554、total 57295 tokens，
usage缺失0次；批次耗时85.17578101158142秒。账单金额未查询。

使用 deepseek-flash / high，API并发上限10（本批最多3例并行）、native并发2，
max_tokens 384000、API硬超时1200s、传输重试最多3次、反馈修复最多3轮。
预算和超时仍为有限上限，不承诺无限输出。三轮无凭据TLS检查均返回401、verify0；
命令级代理127.0.0.1:6478。凭据从原共享.env加载，未修改或打印。

固定 LLVM/MLIR18.1.2、SEAL4.0.0、NumPy1.25.2、Torch2.0.1+cpu。
degree32768、modulus_bits=[60]*14、tc128；没有bootstrap或decrypt-and-reencrypt。
没有安装、下载新依赖或更改驱动。

## 人工前置验证及保留的失败

人工正确程序三例最终通过；两个错误布局程序均在 numerical_comparison 被拒绝。
原人工批次 `/home/lhy/poseidon-work/results/logical-reshape-goldens-gzgkhlqi/report.json`
仍保留 failed：Conv2测试程序用了当前契约不支持的np.zeros，在static_check即失败，
没有编译执行。仅把该人工掩码改为公开列表乘法加np.array，并单独补跑Conv2，
没有扩大候选权限或改动模型、reference、安全参数和门限。
补跑证据 `/home/lhy/poseidon-work/results/logical-reshape-conv2-k1i5z9hf/report.json` 为passed。
证据测试同时检查原失败及独立补跑；从不覆盖原报告。

付费候选没有使用人工答案回填。reshape-bn生成公开仿射表达式；
Conv1生成helper和rotation归约；Conv2生成公开列表/循环与slot对齐。
这三例属于模型组合验证，不并入“指定构造形式”的覆盖计数。

## 证据文件

- 付费批次：`/home/lhy/poseidon-work/results/agent-batch-yox1_hs8/report.json`
  SHA256 `a29ccf3a2bfa6d3643704c20a43ad94d31053d1e3a0a27fadcf2124288358a8e`。
- 独立审计：`/home/lhy/poseidon-work/results/logical-reshape-agent-audit-h86hsu9k/report.json`
  SHA256 `06c6b17f8ccd61580eed4a6cf5e37bbc9857e233a2739ef8a2a3579fe14a47e2`。
- 冻结manifest：`scripts/baseline/cases/logical-reshape-agent-3-manifest.json`
  SHA256 `67c25994a875553becfa4173602ff46f9b97eaec8d66060ad38abb12e4e2e86c`。
- 累计模型域审计：`/home/lhy/poseidon-work/results/model-capability-audit-nqn9hf4f/report.json`
  SHA256 `ece1101ce5b5efea4e5c1738832a625fcb99658d0f07a48de730af8e9e68936c`。
  135例=87个graph案例+48个历史catalog案例；当前15个受限graph算子均有
  output-reachable的实际生成/数值结果，不代表完整上游DSL。

以上审计重读模型、reference、源码、静态规则、产物和解密值，0新增API/密态执行。
原132/126/114模型报告不改写；旧分析源码按哈希归档并由历史回归验证。
本次付费批自动删除2,036,940,030 bytes（约1.90GiB）可再生测试密钥，
旧密钥不可恢复，可重新生成。请求、候选、IR、HEVM/CST、解密结果和诊断保留。

## 复现与回归

仅离线检查，不会再次付费：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 480s python3 scripts/baseline/audit_logical_reshape_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-yox1_hs8
```

新回归入口：test_logical_reshape、test_logical_reshape_evidence、test_logical_reshape_paid。
实际证据环境变量：

```text
POSEIDON_LOGICAL_RESHAPE_GOLDENS=/home/lhy/poseidon-work/results/logical-reshape-goldens-gzgkhlqi/report.json
POSEIDON_LOGICAL_RESHAPE_CONV2=/home/lhy/poseidon-work/results/logical-reshape-conv2-k1i5z9hf/report.json
POSEIDON_LOGICAL_RESHAPE_AGENT=/home/lhy/poseidon-work/results/agent-batch-yox1_hs8
POSEIDON_RECENT_CONSTRUCTION_AUDIT=/home/lhy/poseidon-work/results/recent-construction-audit-_6wfey9q/report.json
```

固定Nix/venv内44模块回归448项：426通过、22条件跳过、0失败。
显式启用新增人工/付费证据、四轮构造付费证据及旧BN/模型能力历史报告。
日志Provider retry来自mock负例，不是额外付费调用。
模块清单：

`test_logical_reshape test_logical_reshape_evidence test_logical_reshape_paid test_batch_norm test_batch_norm_evidence test_batch_norm_paid test_model_graph test_fx_to_hecate test_model_semantic_coverage test_advanced_agent_manifest test_dsl_semantic_inventory test_object_unary test_object_unary_exercises test_object_unary_evidence test_object_unary_paid_evidence test_scalar_conversion test_scalar_conversion_exercises test_scalar_conversion_evidence test_object_arithmetic test_object_arithmetic_exercises test_object_arrays test_public_mappings test_function_construction test_construction_exercises test_construction_calls test_closure_construction test_public_polynomial test_public_strings test_public_control test_public_numbers test_public_sequences test_function_literals test_public_iteration test_public_construction test_extended_arithmetic test_agent_batch test_custom_batch_manifest test_subscription_providers test_rotation_contract test_provider_retries test_hecate_contract test_deepseek_provider test_candidate_pipeline test_agent_credentials`

执行前把上述变量及旧报告变量通过`env K=V`放在Nix内部命令中；
外层环境不自动透传。使用`hecate_python_env.enter_nix`进入既有固定环境，
工作目录scripts/baseline，以VENV/bin/python -m unittest执行。

源码改动集中在logical_reshape、model_graph、fx_to_hecate、model_semantic_coverage、
语义清单、冻结三例/人工程序、离线审计与测试。没有修改Poseidon GPU运行时。
分支feat/agent-dsl-correctness、HEAD4995e7cadedf2bfb9104658b5638662ecf6a1d0a保持。
已有无关修改和sparse checkout保持，未commit/push。建议之后经单独批准拆为
逻辑shape语义、测试/审计、实验文档三个本地commit。

完整DSL目标仍未完成：任意布局/高层helpers、其余FHE管理语义、真实bootstrap，
以及Poseidon GPU编译产物执行仍需各自门禁；不由本批CPU通过替代。
