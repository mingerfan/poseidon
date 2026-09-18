# 12 例扩展 Agent 实验及受限 CPU 目标验收

## 结论与范围

用户明确批准本批付费测试后，12/12 个完整用户图由 DeepSeek 生成 Hecate 程序，
均首次通过 AST/类型/布局检查、Dacapo 编译、实际 SEAL HEVM CKKS 执行和解密差分。
没有规则答案回退，没有反馈修复，没有新增 GPU 对接，也没有降低安全参数。

本批补齐了较宽 MLP、分组/膨胀卷积及显式 graph power 的在线生成证据。
这里的“通过”是固定正常/有符号/随机/边界输入上的工程验证，不是所有允许输入、
任意权重或全部 DSL 组合的形式化证明。完整 Hecate upstream helpers 不因此获准调用。

## 固定设置及真实结果

- Provider/model：DeepSeek 官方 API / `deepseek-flash`，high。
- 输出上限 384000 tokens/请求，1200 秒/请求；API workers=10，native workers=2。
- 命令级代理 127.0.0.1:6478。无凭据 TLS 预检 CONNECT=200、verify=0、HTTP=401。
- LLVM/MLIR 18.1.2，SEAL 4.0.0，tc128，N=32768，16384 slots，14×60-bit key context。
- 不执行 bootstrap，不以 decrypt-and-reencrypt 替代。
- 12 次真实 API 调用；0 次传输重试；0 轮语义修复；0 个失败或未完成案例。
- 首次 parse/type/compile/execution/numerical success 均为 12/12。
- 48 组加密输入、124 个逐项比较输出；最大绝对误差 `1.3196602477449915e-08`。
- 冻结门限仍为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- API usage：prompt=21270，completion=108644，total=129914；12 次均有 usage。
  这是 token 统计，货币费用以服务商账单为准，不从本报告猜测价格。
- 批次内部墙钟计时 103.393 秒，包含 API 等待及本地执行；不是 FHE 性能基准。
- 12 例均已清理 temporary private-keys，共 3567396576 bytes（约 3.32 GiB）。
  原随机密钥字节不可恢复，可生成新密钥重跑。证据和解密数组保留。

| 案例 | 首次通过 | 最大绝对误差 |
|---|---|---:|
| custom-wide-mlp-5 | 是 | 7.770208081e-09 |
| custom-wide-mlp-6 | 是 | 7.824695400e-09 |
| custom-wide-mlp-7 | 是 | 1.319660248e-08 |
| custom-wide-mlp-8 | 是 | 3.661286163e-09 |
| conv1d-dilation2 | 是 | 8.351781641e-09 |
| conv1d-dilation3 | 是 | 3.305544949e-09 |
| conv2d-dilation2 | 是 | 5.563485617e-09 |
| conv1d-groups2 | 是 | 2.752823386e-09 |
| conv2d-depthwise-multiplier | 是 | 6.748631209e-09 |
| conv2d-grouped-dilated-pad | 是 | 3.005476784e-09 |
| explicit-power-2 | 是 | 6.612193015e-09 |
| explicit-power-4 | 是 | 1.313002218e-08 |

## 不依赖绿色报告标记的复核

`audit_model_capabilities.py` 与 `audit_dsl_coverage.py` 重新核对冻结的模型、权重、
输入、trace payload、实际 Hecate 源码、HEVM/CST、实际解密数组以及逐项门限。
新的 opt-in 测试把这次结果与固定 12 例清单绑定，并复核临时密钥目录已经清理。

累计五条结果 lineage 为 **126 个成功案例：78 个 schema-2/3 用户图，
48 个历史 catalog 描述**，共 504 组输入、1428 个输出值。旧报告不改写，不把
48 个旧 catalog 结果重新归类为自定义图，也不把累计集合当成同配置批次成功率。

- 输出可达的受限图算子：13/13。原缺失的 power 已有真实 Agent 证据。
- 模型参数新增线上覆盖：Linear 隐藏宽度 5..8，groups=2、dilation=2/3，
  bounded 2D grouped/depthwise 卷积，power(2/4)。
- DSL AST 分区仍为 29/32。未在线观察到 subtract.scalar、subtract.length1、
  subtract.length4；它们有人工 DSL 密态正反例，而不是未实现的算子。
- 61 份人工证据独立重算：42 个正确 golden、19 个数值反例，244 组输入、
  716 个输出。不是本次重新执行了 61 个密态程序。
- 最早 3 个 v1 报告缺少后来新增的 rotation-key-file probe，明确保留此历史缺口；
  不补写伪造字段。六个 signed-rotation 正例和实际 missing-key 拒绝另有证据。

## 对原三个目标的逐项验收

| 目标 | 已验证的交付 | 仍保留的边界 |
|---|---|---|
| 用户自由提供受限模型与权重；范围至少翻倍 | schema-2/3 数据图输入，不按模型名字选择实现；独立修改名称、权重和图连接的测试；算子 6→13，模型族 8→16；96 例完整图清单；迁移 48 例规则密态和新 12 例 Agent 密态 | 不是任意 PyTorch/Python 或任意动态模型；模型族是明确定义的结构/隐私组合，不冒充新神经网络架构 |
| DSL 语义—状态—测试对应 | 从 pinned Python opcode、Earth/CKKS TableGen、poly helper 枚举源名称；每个名称映射到支持/拒绝/未验证状态和真实测试位置；另有模型算子与实际输出、产物矩阵 | “完整清单”不等于“全部语义已实现”；bootstrap、任意控制流等继续明确拒绝 |
| 算术/旋转/密钥/多输入/shape/broadcast/Linear/MLP/Conv/Pool 增量 | 独立手算/图/PyTorch reference、人工 golden、AST 允许/拒绝、实际编译产物和密态数组、正常/边界/错误程序测试分别验证；164 项定向测试全部通过无跳过 | 输入 packing 仍受限；更大图、范围或深度需重新验证，不默许扩大 |
| 模型近似误差与 CKKS 误差分离 | ReLU 与 p(x)=(x+x²)/2、实际解密值三方独立比较；近似最大误差 0.125，正确 CKKS 残差约 6.94719e-09；错误程序残差约 1.0 | 不自动将 ReLU/SiLU 替换成多项式，不将近似后的模型宣称与原模型等价 |

当前输入边界直接来自 `model_graph.py`：1..4 个逻辑密文输入；每个输入 4 个元素，
允许形状 [4]、[2,2]、[1,4]、[1,2,2]、[2,1,2]、[1,1,4]；Linear 隐藏宽度
1..8，最终 1..4 输出；公开数组有限实数、shape 与总大小受检查；旋转 ±1/±2/±3；
broadcast 是 scalar/length1/精确向量。不是任意维度 broadcasting/repacking。

Poseidon GPU 按用户“不要再对接”的指示不在当前三个目标内，历史 GPU 记录保留，
不作为本轮通过条件。此处验收的是受限 CPU Agent 工程能力，不是完整 FHE compiler
语义证明或新的研究贡献。审计器保留 `full_model_domain_verified=false`，不越权作全域证明。

## 报告和完整性

以下路径均位于 `/home/lhy/poseidon-work/results/`：

| 报告 | SHA256 |
|---|---|
| agent-batch-0j5c90hj/report.json | c4056e98ba730dbccdd23f43f926cc7bfde29bdd672c1fda4a4a45334d266297 |
| model-capability-audit-58yddwhv/report.json | 234af9a8e98d6f19ce1e297c8d586df75427e6877cab2577794ee62f2a28e1a9 |
| dsl-grammar-audit-xs_s4cxq/report.json | 953a5c1acd43c286cb2db6e949c0357db66ac3e29a6f6e673aa7e195da1de0b6 |
| manual-semantics-audit-i_brftfr/report.json | ade92ae69b5cf9e026625ca517fa68787ed8082fa55911fa578e194c2469ed5b |

输入清单：`scripts/baseline/cases/advanced-agent-12-manifest.json`；
SHA256 `a398e1ec1ade2d1639d793b41cf8fe8523df4cb3a8b0c901b0b0e869cb4d5678`。
原模型、权重、输入、数值门限、编译器配置均未为通过而修改。

## 回归与失败定位

- 普通 WSL Python：429 项，326 通过、103 个条件跳过，不把跳过算成功。
- 固定 Nix/Torch/NumPy 环境，明确启用相关实际证据：164 项，全部通过，无跳过。
- 临时跨 PowerShell→WSL 测试命令曾提前展开动态库环境变量，导致 NumPy 报
  `libstdc++.so.6: cannot open shared object file`（28 错误、13 跳过）。
  修正命令使其只在 Nix shell 内展开后原测试通过。未安装依赖，未修改数值、
  编译器或批次结果。下方可复现命令用 chr(36) 避免该跨 shell 问题。

## 可复现命令

付费批次的实际启动命令（重新执行会再次收费，不是查看结果）：

```bash
python3 scripts/baseline/run_agent_batch.py --live --provider deepseek \
  --model deepseek-flash --reasoning-effort high --max-tokens 384000 \
  --api-timeout 1200 --jobs 10 --loopback-proxy-port 6478 \
  --case-manifest scripts/baseline/cases/advanced-agent-12-manifest.json
```

下列验收命令不调用 API、不重跑 FHE，仅检查已有证据和本地规则。应从 WSL 源码根
运行，勿与另一个持有 Nix launcher 锁的任务并行：

```bash
timeout -k 3s 410s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline python3 - <<'PY'
from hecate_python_env import enter_nix,VENV
import shlex
envs={
  "POSEIDON_ADVANCED_12_AGENT_REPORT": "/home/lhy/poseidon-work/results/agent-batch-0j5c90hj/report.json",
  "POSEIDON_EXTENDED_CAPABILITY_REPORT": "/home/lhy/poseidon-work/results/model-capability-audit-58yddwhv/report.json",
  "POSEIDON_MANUAL_SEMANTICS_REPORT": "/home/lhy/poseidon-work/results/manual-semantics-audit-i_brftfr/report.json",
  "POSEIDON_APPROXIMATION_RESULTS": "/home/lhy/poseidon-work/results/approximation-golden-v08aa_ux",
  "POSEIDON_MIGRATED_RULE_REPORT": "/home/lhy/poseidon-work/results/fx-batch-t0alnl8z/report.json",
  "POSEIDON_POWER_GOLDEN_REPORT": "/home/lhy/poseidon-work/results/explicit-power-goldens-a3ovzyp7/report.json",
  "POSEIDON_POWER_RULE_REPORT": "/home/lhy/poseidon-work/results/fx-batch-e_r7dpt6/report.json",
  "POSEIDON_WIDE_RULE_RESULTS": "/home/lhy/poseidon-work/results/fx-batch-97kbilk1",
  "POSEIDON_WIDE_GOLDEN_RESULTS": "/home/lhy/poseidon-work/results/wide-linear-goldens-02h0bwbh",
  "POSEIDON_GROUPED_SPATIAL_RESULTS": "/home/lhy/poseidon-work/results/grouped-spatial-goldens-ffbvf58_",
  "POSEIDON_GROUPED_SPATIAL_RULE_RESULTS": "/home/lhy/poseidon-work/results/fx-batch-3lsvho1l",
  "POSEIDON_MODEL_CAPABILITY_REPORT": "/home/lhy/poseidon-work/results/model-capability-audit-zxrvrtlh/report.json",
  "POSEIDON_EXPANDED_RULE_RESULTS": "/home/lhy/poseidon-work/results/fx-batch-0z3yioki",
  "POSEIDON_ROTATION_MISSING_KEY_REPORT": "/home/lhy/poseidon-work/results/rotation-batch-i9qaiev4/missing-key.json",
  "POSEIDON_ROTATION_BATCH_RESULTS": "/home/lhy/poseidon-work/results/rotation-batch-i9qaiev4",
  "POSEIDON_SCHEMA3_RESULTS": "/home/lhy/poseidon-work/results/schema3-golden-batch-ht5no6ir",
  "POSEIDON_NATIVE_NEGATE_RESULTS": "/home/lhy/poseidon-work/results/candidate-replay-gq2y3dwl",
  "POSEIDON_NATIVE_SUBTRACT_RESULTS": "/home/lhy/poseidon-work/results/candidate-replay-yovkx6a5",
  "POSEIDON_NATIVE_SUBTRACT_WRONG_RESULTS": "/home/lhy/poseidon-work/results/candidate-replay-8vm9gphy",
  "POSEIDON_ZERO_GOLDEN_REPORT": "/home/lhy/poseidon-work/results/zero-golden-batch-w9vuz53i/report.json",
  "POSEIDON_ZERO_RULE_REPORT": "/home/lhy/poseidon-work/results/fx-batch-nmqzon3l/report.json",
  "POSEIDON_ZERO_AGENT_REPORT": "/home/lhy/poseidon-work/results/agent-batch-gp73w2a1/report.json",
  "POSEIDON_BROADCAST_GOLDEN_REPORT": "/home/lhy/poseidon-work/results/broadcast-golden-batch-cq9e4lbd/report.json",
  "POSEIDON_BROADCAST_AGENT_LINEAGE": "/home/lhy/poseidon-work/results/agent-batch-d1dbuvl6/report.json",
  "POSEIDON_SPATIAL_RESULTS": "/home/lhy/poseidon-work/results/spatial-golden-batch-z8dy657u"
}
tests=[
  "test_manual_semantics",
  "test_advanced_agent_manifest",
  "test_dsl_semantic_inventory",
  "test_approximation_errors",
  "test_catalog_graph_migration",
  "test_explicit_power",
  "test_wide_linear",
  "test_grouped_spatial",
  "test_model_graph",
  "test_model_semantic_coverage.ModelSemanticTests",
  "test_model_semantic_coverage.ModelCapabilityEvidenceTests",
  "test_expanded_suite",
  "test_rotation_contract",
  "test_schema3_graph",
  "test_native_arithmetic",
  "test_zero_abi",
  "test_broadcast_semantics",
  "test_hecate_contract",
  "test_candidate_pipeline.CandidateContractTests",
  "test_candidate_pipeline.FeedbackLoopTests",
  "test_candidate_pipeline.SandboxCommandTests",
  "test_semantic_guidance",
  "test_spatial.SpatialTests",
  "test_spatial.SpatialTorchTests",
  "test_spatial.SpatialEvidenceTests.test_all_goldens_and_wrong_windows_are_real_and_immutable"
]
command = 'LD_LIBRARY_PATH='+chr(36)+'HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline ' + ' '.join(k+'='+shlex.quote(v) for k,v in envs.items()) + ' ' + shlex.join([str(VENV/'bin/python'),'-m','unittest',*tests,'-q'])
raise SystemExit(enter_nix(command,seconds=300))
PY
```

修改仅保存在本地 working tree；没有 commit、push、PR 或分支操作。

