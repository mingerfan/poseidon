# 原生函数11项：真实Agent生成与密态差分结果

2026-09-16，用户明确批准11项测试及付费调用上限后，使用现有批量入口完成
DeepSeek官方 `deepseek-flash` / `high` 的真实Agent生成。最终11/11通过，
首次10/11通过；不是人工golden、规则转换器或旧结果回放。

## 批次与配置

- 路径：`/home/lhy/poseidon-work/results/agent-batch-bg4htr49`。
- report SHA256：`9cbe3a0f9a1ad2904a6c669df5f0f865505be74084f84d88df667f5f13ff7f30`。
- API并发10，native编译/密态执行并发2；stream开启，默认连接路由。
- 每次API硬超时1200秒、输出上限384000 tokens；每例最多3轮语义修复，
  每次逻辑生成最多3次可重试传输失败重试。
- 批次允许上限44次逻辑生成、176次HTTP尝试；实际12次生成/HTTP，0次传输重试。
- 使用本地 `.env` 中的凭据，由现有加载器读取；未修改或输出凭据。
- 仅发送合成模型/公开常量、layout、DSL规则、构造要求和受限反馈，
  不发送人工golden、测试输入数组、逐项reference、仓库全量或FHE私钥。
- 批次耗时约287.87秒；21,200 prompt + 75,784 completion = 96,984 reported tokens。
  所有请求均返回usage；未查询服务商货币账单，不能将token数称作实际金额。

## 逐项覆盖

| ID | 必须实际使用的构造 | 首次 | 最终 | API调用 |
|---|---|---|---|---:|
| nf-scalar | helper返回单密文 | 通过 | 通过 | 1 |
| nf-pair | tuple多返回，每个单元影响输出 | 通过 | 通过 | 1 |
| nf-nested | helper内部调用另一个helper | 通过 | 通过 | 1 |
| nf-forward | 调用源码中后定义的helper | 未通过静态检查 | 通过 | 2 |
| nf-two-inputs | 两个密文形参均影响有序减法 | 通过 | 通过 | 1 |
| nf-public-argument | p参数参与密态结果 | 通过 | 通过 | 1 |
| nf-identity | 直接返回密文形参并使用 | 通过 | 通过 | 1 |
| nf-repeated | 同一helper两次调用分别影响输出 | 通过 | 通过 | 1 |
| nf-zero-input | 无参数helper返回公开系数 | 通过 | 通过 | 1 |
| nf-empty | 可达空list/tuple返回调用 | 通过 | 通过 | 1 |
| nf-plain-return | helper返回Plain并用于输出 | 通过 | 通过 | 1 |

`nf-forward`首次错误为 `Invalid or colliding function name`，位于候选静态检查层，
未进入编译。首次helper名为 `_residual`，不符合当前标识符允许规则；Agent收到诊断后
自行改为 `residual` 并通过。没有人工替换候选，没有放宽语法或数值门限。
无TLS错误、无编译/执行/数值失败。首次parse率11/11、首次compile/execution/数值正确率10/11，
最终数值正确率11/11；平均修复轮数1/11。

十项具有固定公开探针上的有限输出影响证据，并与真实Hecate native调用源码位置对齐。
空返回项只有可达调用和真实trace的结构证据；空返回本身不存在可扰动的输出单元。
这些程序使用两个数学图（十例仿射、一例双输入减法），不是11个模型家族或全输入形式化证明。

## 真实密态执行与独立审计

链路：模型描述 → 真实API生成Hecate候选 → AST/c-p类型/逐项构造检查 →
实际Hecate tracing → Dacapo编译 → HEVM/CST → upstream SEAL HEVM CPU → 解密差分。
后端不是Poseidon GPU；不新增CPU模拟器，不执行bootstrap。

- SEAL4.0.0，N32768，14×60-bit模数，tc128检查通过；waterline40。
- 四组固定输入/案例，共44组输入执行、176个输出值。
- 门限不变：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 最大绝对误差：`3.038591578352623e-8`。
- 加权MAE：`6.057505596756339e-9`。
- 非零reference最大相对误差：`9.516061001452754e-8`。

独立离线审计：`/home/lhy/poseidon-work/results/native-exercise-audit-s9vhta5d/report.json`。
SHA256：`7279990dcc5364b3d0d7b3d83b723b3e71924a049255b8645d58ca526d026c52`。
审计逐项复核provider响应与trace payload/源码的一致性、冻结输入与产物哈希、
重新计算的构造检查、真实native调用位置、独立明文公式、解密数组以及密钥清理。
审计本身0次API、0次新FHE执行。

本批按既有保留策略已删除7,468,780,110字节（约6.96GiB）可再生密钥，
每例清理完整，私钥文件不可恢复。保留模型、请求、响应、失败反馈、IR、HEVM/CST、
输入/reference、解密数组、日志和报告。没有删除旧批次。

## 最近原生数组扩展：仍不能算Agent覆盖

> 2026-09-17：数组逐项要求、真实frontend存储观察和独立审计现已接通并通过人工密态门禁，
> 但尚未启动数组付费批次。新进展见 [原生数组构造门禁](native-array-construction-coverage.md)。
> 本节其余内容保留2026-09-16批次完成时状态。

上一阶段人工批 `native-array-goldens-y_bgpaop` 的10个正确程序全部通过，
故意用错转置索引的程序经真实密态执行后在数值比较层被正确拒绝。
覆盖负步长切片、转置、rank-4 reshape、按行解包、混合Plain/密文、0-D提取/返回、
空数组和嵌套调用。160个正确输出值最大绝对误差 `2.843846358402402e-8`，
MAE `5.331592287799447e-9`。本次新增离线证据测试复核了这批结果。

这是独立native-array v3请求契约，尚未冻结逐项Agent构造要求，也未包含在本次11例付费批准范围内。
不能把这批人工代码改标成Agent结果，或把原生函数11项通过说成所有近期数组语法已付费覆盖。
下一步需建立数组逐项要求和防无效填充检查，再进行独立获批的真实Agent批次。
native core与旧v22构造的组合、上游高层helper和真实bootstrap等仍是未完成范围。

## 回归、复现与本地修改

本次针对native core、数组类型/证据、真实Agent证据、人工/Agent隔离和构造门禁的
42项定向离线回归全部通过。随后包含旧v22、native frontend、bootstrap前端边界和语义清单的
118项综合回归全部通过，无失败、无跳过；两组存在重叠，不相加计数。
这些回归不调用付费API，也不把mock调用算成付费请求。
初次试图在Nix外直接启动隔离venv失败（FileNotFoundError），按既有Nix入口启动后通过；
未安装或修改依赖。一次审批服务容量错误在离线进程创建前阻断，随后同一检查重试获准。

```bash
cd '/mnt/d/Code Space/Poseidon'
# 只审计已有付费结果；不会重新请求模型或执行FHE
timeout -k 5s 280s python3 scripts/baseline/audit_native_function_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-bg4htr49
```

新增 `test_native_function_paid_evidence.py` 和 `test_native_array_evidence.py`；
更新语义清单、清单单测和历史文档索引。生成/编译/执行器、候选源码、reference、安全参数
和既有provider配置没有因本批改动。所有修改留在本地working tree，不commit/push/PR。
建议把本次证据测试、清单更新和结果文档作为一个独立本地commit，待用户明确授权后再提交。
