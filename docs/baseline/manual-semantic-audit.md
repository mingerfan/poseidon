# 人工 DSL 语义证据：从实际数组重算，不只读取通过标记

## 结论

61 份已有人工程序证据重新审计通过：42 个正确 golden、19 个应在数值比较
中失败的反例，共 244 组输入、716 个输出值。正确程序最大绝对误差
`4.886002322956884e-06`，仍满足原门限 `1e-5 + 1e-4*abs(reference)`。

这是读取已有结果后重算，不是重新执行 61 个密态程序，也不是新增 Agent 成功。
本轮 API 调用数、新 FHE 执行数均为 0；没有改写原报告或接触私钥。

最终报告：`/home/lhy/poseidon-work/results/manual-semantics-audit-i_brftfr/report.json`。
SHA256：`ade92ae69b5cf9e026625ca517fa68787ed8082fa55911fa578e194c2469ed5b`。

## 本次补强的缺口

部分早期证据测试检查文件哈希和报告中的 comparison.passed，却没有重新加载
解密数组计算误差。新 `manual_semantic_evidence.py` 增加以下完整检查：

1. 原模型、request、实际送入 tracing 的冻结 payload 必须一致。
2. 人工 golden 源码、候选 sidecar 和冻结 payload 中的程序必须相同；重新运行
   AST/类型检查，不执行候选 Python。
3. 核对冻结输入、权重、IR、HEVM/CST，重新解析实际二进制编译产物。
4. runtime 的数组文件必须只有 inputs，且与原 inputs 完全相同，不能包含 reference。
5. 根据原输入重新计算独立图 reference，再核对保存的 reference。
6. 从实际 decrypted.npy 重新计算逐项误差及指标，与旧记录一致后才认可结果。
7. 正例必须数值通过；反例必须已经完成真实 tracing/compile/execution，最后在
   numerical_comparison 失败。成功检出反例不是一个正确模型。

数组篡改、把解密答案回填 reference、维持旧绿色报告但修改输出、交换输入顺序、
改变 dtype/shape、NaN、放宽误差门限都有新的审计器反例测试。

## 覆盖与边界

| 证据组 | 正确程序 | 故意错误程序 | 关键检查 |
|---|---:|---:|---|
| 原生算术 | 2 | 1 | 取负、公开/密文减法，错误减法顺序 |
| signed rotation | 6 | 1 | ±1/±2/±3、方向与当时的实际密钥检查记录 |
| 多输入 | 4 | 1 | 2..4 输入的名称、shape、顺序及冻结绑定 |
| Conv/AvgPool | 9 | 2 | 窗口、通道、步长、padding、边界除数 |
| 较宽 Linear/MLP | 1 | 1 | width8 人工程序与错误 neuron index |
| grouped/dilated Conv | 6 | 2 | 分组、dilation、错误 tap/group |
| broadcast | 5 | 5 | scalar/length1/vector、别名和错误常量语义 |
| encrypted zero | 6 | 3 | 显式受信 Enc(0) 绑定，不使用透明密文豁免 |
| 显式 power | 2 | 2 | x²/x⁴，错误指数不能只靠 0、±1 检出 |
| 激活近似示例 | 1 | 1 | 原激活、多项式、实际解密输出三方比较 |

13 类当前受限图算子都出现在正确人工程序所实现的输出依赖中。
这不是完整 Hecate DSL 的所有组合验证，也不是任意权重/深度的形式化证明。
某个错误程序包含某算子，不代表错误已定位到该算子；矩阵明确保留这个区别。
测试入口是可追踪链接，不会因为出现在矩阵里就被算作该审计已经执行的测试。

另有两点必须保留：

- 最早三个 v1 算术报告没有后来新增的 rotation-key-file probe 字段。
  新审计明确记录 `not_recorded_in_legacy_v1`，不伪造字段；较新契约缺失该
  字段仍失败。旋转要求另有六个 signed-rotation 正例的记录支撑。
- 较宽模型的人工密态证据是 width8 正/反两例，不是 width5..8 各一例；
  width5/6/7/8 四例的规则转换密态证据在 `fx-batch-97kbilk1`，两类证据不混计。

## 激活近似误差单独重算

示例多项式为 p(x)=(x+x²)/2，目标激活为 ReLU，输入范围 [-1,1]。
审计同时读取原输入和实际解密数组，重新计算：

- 模型近似误差：p(x) - ReLU(x)，本组最大绝对值 0.125。
- 实现/CKKS 残差：decrypted - p(x)，正确程序最大绝对值约 `6.94719e-09`。
- 相对原激活的总误差：decrypted - ReLU(x)。

正确密态执行并不消除 0.125 的模型近似误差，不能据此宣布原 ReLU 模型等价。
错误程序相对多项式的残差约 1.0，也不能笼统叫作 CKKS 噪声。
自动 ReLU/SiLU 改写和生产近似方案仍未启用。

首次新审计报告 `manual-semantics-audit-t4z73_8w/report.json` 保留了 59/61 的结果：
两项失败来自新审计器把既有 `ckks_execution_error` 错写为 `execution_error`。
这是审计读取层错误，不是 FHE 数值失败。修正字段名并添加回归测试后，最终
61/61 通过；没有修改旧数值、模型或门限。

## 当前目标逐项状态

后续更新：下表记录本人工审计完成时的剩余工作。用户批准后的
[12 例扩展 Agent 实验](advanced-agent-12-results.md) 已补齐较宽 MLP、
分组/膨胀 Conv、显式 power 的线上生成与密态证据；61 例人工审计仍是
独立的历史证据，不增加其计数，也不再因下述外发批准等待而阻塞。

| 要求 | 当前证据 | 未完成项 |
|---|---|---|
| 自由提供受限图/公开权重，范围至少翻倍 | 13 图算子，对比冻结基线 6；16 模型族，对比 8；完整 96 例 schema-2/3 清单、迁移 48 例真实规则执行 | 更广模型结构的在线 Agent 验证仍需独立结果 |
| DSL 语义—支持状态—测试对应 | 固定上游 Python/Earth/CKKS/poly 操作枚举与本地状态映射；新 61 例实际数组/产物复核 | 不将未支持控制流、bootstrap、上游 helper 等说成已支持 |
| 算术、rotation、输入/shape/broadcast、Linear/MLP、Conv/Pool | 独立 reference、人工 golden、静态拒绝、真实编译/密态、边界与反例分开记录 | 较宽 MLP、分组/膨胀 Conv、显式 power 的新在线生成结果 |
| 近似误差与执行误差区分 | 三方输出从原输入/解密数组重新计算，保留错误程序反例 | 不宣称自动近似能力或原激活等价 |
| Poseidon GPU | 用户已移出当前范围 | 不作为本目标门禁，不继续 GPU 工作 |

目前等待明确的数据外发批准，不能用更多离线模拟代替尚缺的真实 Agent 调用。
待运行的固定清单是 `scripts/baseline/cases/advanced-agent-12-manifest.json`，
包含原高级 10 例和显式幂次 2 例，未改任何现有权重或输入定义。
清单 SHA256：`a398e1ec1ade2d1639d793b41cf8fe8523df4cb3a8b0c901b0b0e869cb4d5678`。
其计划测试验证 DeepSeek Flash、API 并发 10、零调用；真实结果测试仅在
`POSEIDON_ADVANCED_12_AGENT_REPORT` 指向实际批次后启用，未运行时明确跳过。

## 可复现命令

从 WSL 源码根执行，以下均不调用 API：

```bash
timeout -k 3s 180s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/audit_manual_semantics.py

timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan \
  --case-manifest scripts/baseline/cases/advanced-agent-12-manifest.json
```

审计写一个新的小型报告目录，原密态结果保持只读。它不会生成大密钥或清理旧证据。
