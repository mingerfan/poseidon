# 最近新增构造语法：真实 Agent 定向覆盖验收

## 结论与范围

本轮冻结清单 **30/30 案例、117/117 去重观察项**通过真实 DeepSeek 生成、受限 AST 检查、Hecate tracing、Dacapo 编译、SEAL CPU 密态执行和数值差分。最终审计没有新增 API 调用或密态执行。

这是 **implementation-form coverage（指定构造写法的覆盖）**，不是全体上游 DSL 语义的证明。30 个案例都使用同一受限公开计算图：

`y = x * 0.5 + x + 0.375`，输入/输出 shape 为 `[4]`。

每例要求 Agent 使用不同构造形式实现这一数学目标；没有提供该案例的 golden 答案、测试输入或 reference 输出。每个最终候选运行零输入、有符号输入、固定种子随机输入和边界输入，共 4 组。这样隔离“公开构图语法是否真正被生成并使用”，不能据此宣称 30 种网络、unseen family 泛化、跨元素 Linear 或所有 packing 已验证。

本轮公共字典正式接入 **request v19 / AST v18**；v18 及更早请求规则保留。字符串、字典、控制流和对象数组在公开构图时运行/展开，不是在密文上执行 Python 字典或字符串操作。最后交给 Hecate 的仍是可检查的同态算术程序。

## 最终数值与付费结果

| 指标 | 结果 |
|---|---:|
| 最终唯一有效案例 | 30/30 |
| 冻结清单去重观察项 | 117/117 |
| 最终选定候选输入组 | 120 |
| 最终选定输出标量比较 | 480 |
| 加权 MAE | 5.892144535358616e-9 |
| 最大绝对误差 | 3.214195110068374e-8 |
| 本轮真实 API 调用 | 55 |
| prompt tokens | 336466 |
| completion tokens | 497165 |
| total tokens | 833631 |
| 实际传输层重试 | 0 |
| 全部批次实际密态输入组（含被替代结果） | 132 |
| 全部批次实际比较标量（含被替代结果） | 528 |

逐项数值门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`，没有改 reference、输入、权重、CKKS 安全参数或门限。相对误差和适用时 cosine 等明细保留在单例 comparison 中。

服务为 DeepSeek，模型 `deepseek-flash`，reasoning `high`，max output tokens 384000，单次 API timeout 1200s，传输重试上限 3，反馈修复上限 3（最多初始生成 + 3 次修复）。主批 API 并发 10、native 执行并发 2；小补批受案例数限制。显式代理端口 6478；凭据仍从现有本地共享环境文件加载，没有写入报告。账单金额未查询，不按不确定价格估算费用。

### 不合并或美化原始失败

| 批次 | 计划案例 | 首次数值通过 | 批次最终数值通过 | API 调用 | tokens |
|---|---:|---:|---:|---:|---:|
| 主批 `agent-batch-64es49_l` | 30 | 20/30 | 28/30 | 46 | 684557 |
| 补批 `construction-retry-m54nn1ba` | 4 | 2/4 | 4/4 | 7 | 105932 |
| live-view 补批 `construction-retry-vv10r7yj` | 1 | 0/1 | 1/1 | 2 | 43142 |

主批首次 JSON/Python parse 30/30；首次 static/compile/execution/numerical 均为 20/30。主批全部完成案例平均反馈修复 0.5333 次。这个数仅描述原始主批，不把补批混入同一个固定策略成功率。

主批两个最终未过案例为字符串 replace/join 和对象 transpose。后来源码复核又从主批的数值成功案例中排除 3 个**不足以证明有效构造覆盖**的程序：loop-else、mapping-pop-clear、mapping-live-views。它们的数值通过历史仍保留，但最终覆盖选择补测版本。

所有运行共 22 次 static_check 失败尝试、3 条 coverage_reaudit 排除记录；后者不是额外 API 失败。期间覆盖检查器也有误拒绝，因此不能把 22 次全部算作 Agent 语义错误。原始汇总分类不改写，解释如下：

- Agent 曾生成当前契约不接受的 `np.transpose(a)`，后续使用 `a.transpose()` / `a.T`。
- 原 loop-else 的两个 else 对同一个计数器产生冗余效果；去掉任一个仍正确，不作为有效覆盖。新版本每个 else 独立改变返回计算。
- 原 clear 操作针对已经清空的字典；新版本清空非空字典，并由长度决定输出路径。
- 原 items() 读到的值后来被 values() 路径覆盖；新版本从 items 结果读取实际权重和 bias，同时用创建视图后新增的键/值构造输出。
- 覆盖探针曾把字符串替换成错误类型或破坏分隔字段数，导致没有得到合法扰动；修正为类型/结构保持探针，增加回归。
- live items 视图必须在消费时扰动，过早快照会错误丢掉后续更新；已增加消费时扰动和回归。被旧门禁拒绝的尝试没有倒推成“已完成 FHE 执行”。
- 最初审计把 JSON 中的 list 与内存中的 tuple witness 比较，出现假失败；改为稳定 JSON 表示。历史审计文件未覆盖。

## “逐项覆盖”的证据强度

117 是 AST 形式、运行计数、方法和控制事件的**观察项**，不是 117 个密态算子。

1. 生成源码包含目标形式，但仅出现不计覆盖。
2. 受限构图解释器记录实际执行的源码位置或计数。
3. 对适用的表达式、else、clear 和 items 路径施加有限扰动，要求仍可合法展开且探测输出改变，以拒绝无用或冗余代码。
4. 原始未扰动候选真实编译、加密执行、解密，与独立 reference 比较。
5. 人工复核存储、参数绑定、别名、视图等结构性证据。

最终矩阵中 **34 项有输出扰动敏感证据，83 项是执行结构证据并经源码复核**。后者不冒称都通过数值消融。尤其 shape/reshape/T 的小数组、逆序后再逆序、排序的特定值等仅证明所测路径，不证明所有排列、形状或语义变异都可被这批模型识别。输出扰动也只是有限探测，不是正式污点分析或全输入等价证明。

## 30 个最终有效候选

每例 4 组密态输入、16 个输出比较。观察项可在多个案例中重复，去重后为 117。

| 案例 ID | 要求观察项数 | 最大绝对误差 | 结果目录名 |
|---|---:|---:|---|
| `closure-nonlocal` | 3 | 1.2075e-8 | `agent-deepseek-_2cw8r38` |
| `call-defaults` | 5 | 1.2500e-8 | `agent-deepseek-v1v_w9up` |
| `call-unpacking` | 6 | 2.2135e-8 | `agent-deepseek-9u5hhtbk` |
| `comprehensions` | 3 | 1.3787e-8 | `agent-deepseek-pxeo5i4v` |
| `iterators` | 8 | 8.7119e-9 | `agent-deepseek-px_s7lhm` |
| `lambda-sorted` | 5 | 1.5473e-8 | `agent-deepseek-r_xphgfm` |
| `sequence-slices` | 4 | 1.3534e-8 | `agent-deepseek-_7t6s8_o` |
| `sequence-alias` | 4 | 1.6292e-8 | `agent-deepseek-98kpjr3f` |
| `numeric-scalars` | 6 | 3.2142e-8 | `agent-deepseek-ik86x_es` |
| `numeric-arrays` | 6 | 1.3792e-8 | `agent-deepseek-w6v8gql_` |
| `while-control` | 7 | 2.6873e-8 | `agent-deepseek-ck063xw5` |
| `boolean-membership` | 7 | 1.4756e-8 | `agent-deepseek-q0qz3p3d` |
| `string-strip` | 5 | 1.1353e-8 | `agent-deepseek-9jpslu7m` |
| `string-split` | 5 | 1.3339e-8 | `agent-deepseek-51snfh5t` |
| `chebyshev-coefficients` | 5 | 1.5117e-8 | `agent-deepseek-pzsqw_i6` |
| `chebyshev-arithmetic` | 9 | 2.7954e-8 | `agent-deepseek-hw9de_u6` |
| `numpy-functions` | 6 | 1.6875e-8 | `agent-deepseek-7vt4mctq` |
| `object-full-empty` | 5 | 1.4486e-8 | `agent-deepseek-a65cweew` |
| `object-empty-initialization` | 3 | 2.2782e-8 | `agent-deepseek-5dkdtc3f` |
| `object-array-alias` | 4 | 1.2082e-8 | `agent-deepseek-d9yj7rii` |
| `object-slice-copy` | 4 | 1.7451e-8 | `agent-deepseek-dgbtn2go` |
| `object-concatenate` | 7 | 2.8566e-8 | `agent-deepseek-lrlwva4i` |
| `object-augassign` | 4 | 2.7062e-8 | `agent-deepseek-j7wssvc8` |
| `mapping-get-default` | 5 | 1.3704e-8 | `agent-deepseek-to8bj33j` |
| `mapping-update-copy` | 4 | 1.7487e-8 | `agent-deepseek-msnxya0m` |
| `string-replace-join` | 4 | 8.9756e-9 | `agent-deepseek-iuzitx7d` |
| `object-reshape-transpose` | 5 | 2.0381e-8 | `agent-deepseek-hs3u8qvi` |
| `loop-else` | 4 | 1.5181e-8 | `agent-deepseek-9tcer_u5` |
| `mapping-pop-clear` | 5 | 1.1363e-8 | `agent-deepseek-7s3yo4bv` |
| `mapping-live-views` | 7 | 1.5493e-8 | `agent-deepseek-4bsrkh3q` |

目录均位于 `/home/lhy/poseidon-work/results`。单例 report、request、模型、候选源码、trace/编译产物、解密结果与比较证据保留；最终审计含源码和解密结果哈希。

## 固定证据与复现

最终验收文件：

`/home/lhy/poseidon-work/results/construction-agent-audit-jb3ay36g/report.json`

SHA-256:

`de24ed4c16c7c58523094c3203b21ad24d31b9a06450c3bb2a0fd9000f6dca7d`

主批、两份补批和所有旧失败文件均保留。最终审计的 sources 数组记录原始批次及五个补测单例 report 的 SHA-256。冻结清单见 [构造规范](../../scripts/baseline/construction-exercises-v1.json) 和 [30 例 manifest](../../scripts/baseline/cases/construction-exercises-30-manifest.json)。

从 WSL 源码目录执行下列命令只做离线再验收，不读取 API key、不调用 API：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 660s python3 scripts/baseline/audit_construction_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-64es49_l \
  --additional-run /home/lhy/poseidon-work/results/agent-deepseek-iuzitx7d/report.json \
  --additional-run /home/lhy/poseidon-work/results/agent-deepseek-hs3u8qvi/report.json \
  --additional-run /home/lhy/poseidon-work/results/agent-deepseek-9tcer_u5/report.json \
  --additional-run /home/lhy/poseidon-work/results/agent-deepseek-7s3yo4bv/report.json \
  --additional-run /home/lhy/poseidon-work/results/agent-deepseek-4bsrkh3q/report.json
```

检查清单、不付费：

```bash
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan --construction-exercises
```

以下会产生**新费用**，仅记录实验配置，不需要为读取本报告重新运行：

```bash
timeout -k 10s 18000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --construction-exercises --provider deepseek \
  --model deepseek-flash --reasoning-effort high --max-tokens 384000 \
  --api-timeout 1200 --provider-retries 3 --loopback-proxy-port 6478 --jobs 10
```

已有隔离环境为 LLVM/MLIR 18.1.2、SEAL 4.0.0、NumPy 1.25.2、Torch 2.0.1+cpu。无安装、sudo、新下载或驱动变更。大型构建策略仍为最多 -j2；本轮并未重建大型依赖。

## 回归验证与清理

最终在现有固定 Nix/Python 环境执行 24 个相关 unittest 模块：**329 项，310 通过，19 条件跳过，0 失败**。其中新增 construction_exercises 测试 15 项。覆盖新旧构图语义、批量 manifest、provider/request、凭据和候选管线；provider 重试日志来自 mock 负例，不是额外付费流量。条件跳过不算通过，也不以这些单元测试冒充真实密态测试。

模块：
`test_construction_exercises test_public_mappings test_object_arrays test_public_polynomial test_public_strings test_public_control test_public_numbers test_public_sequences test_function_literals test_public_iteration test_construction_calls test_closure_construction test_function_construction test_public_construction test_extended_arithmetic test_agent_batch test_custom_batch_manifest test_subscription_providers test_rotation_contract test_provider_retries test_hecate_contract test_deepseek_provider test_candidate_pipeline test_agent_credentials`。

在该隔离环境内，以工作目录 `scripts/baseline` 运行：
`LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 /home/lhy/poseidon-work/venvs/hecate-2.0.1-cpu/bin/python -m unittest <上列模块>`。
现有 `hecate_python_env.enter_nix(command, seconds=480)` 负责离线进入固定环境，不混装系统 Python。

付费各轮自动清理合计 **23,764,300,349 bytes（约 22.13 GiB）** 临时测试密钥。源码/JSON/IR/HEVM/CST/数值结果未清理。旧随机密钥不可恢复，但重新执行可生成新密钥；不保证重新执行产生逐 bit 相同的 CKKS 数值。API Key 不属于清理对象。

## 本轮代码修改与仍未完成的范围

正式契约/执行接线修改：
`candidate_contract.py hecate_contract.py candidate_trace.py candidate_sandbox.py run_candidate.py dsl_grammar_coverage.py deepseek_provider.py function_construction.py run_agent_batch.py`。

新增：
`construction_exercises.py construction-exercises-v1.json cases/construction-exercises-30-manifest.json test_construction_exercises.py golden_cases/construction_exercises/mapping_gate.py run_construction_retries.py audit_construction_batch.py`，以及本报告；同步更新公开字典与总体语义进度文档。

全部修改留在本地 working tree。分支 `feat/agent-dsl-correctness`、HEAD `4995e7cadedf2bfb9104658b5638662ecf6a1d0a` 保持不变；既有 C++/GPU/CMake 和 submodule 修改保留。未 commit、push 或改变 sparse checkout。建议未来经批准分别提交“正式契约与覆盖门禁”“冻结案例/回归测试”“验收与语义文档”，不要混入无关旧修改。

下一步仍需要：

- 所有构造的边界值、异常、不同 shape/dtype、别名/浅拷贝和 iterator 失效变体；本轮不是这些语义的穷举。
- 把构造组合用于不同小网络、跨元素 Linear/归约等，再测模型结构泛化；保留 deterministic translator 对照。
- 按实际支持范围推进其余上游 helpers、一般 packing、IR calls、真实 bootstrap/upscale。
- Poseidon adapter/GPU 的真实执行兼容与数值验收；**本轮 poseidon_gpu_validated=false**。
- 整体 DSL 完整支持目标仍开放；本轮 `all_upstream_semantics_proven=false`。

## 117 项最终逐项矩阵

“执行 + 源码复核”不等于该项已有输出扰动证明。这里列出的都是所选有效候选，原始被排除版本不参与此表。

| 观察项 | 证据等级 | 有效案例 |
|---|---|---|
| `attr.T` | 执行 + 源码复核 | `object-reshape-transpose` |
| `attr.coef` | 执行 + 输出扰动敏感 | `chebyshev-coefficients`, `chebyshev-arithmetic` |
| `attr.domain` | 执行 + 输出扰动敏感 | `chebyshev-coefficients` |
| `attr.double` | 执行 + 源码复核 | `numpy-functions` |
| `attr.float64` | 执行 + 源码复核 | `numpy-functions` |
| `attr.ndim` | 执行 + 源码复核 | `object-concatenate` |
| `attr.shape` | 执行 + 源码复核 | `object-concatenate` |
| `attr.size` | 执行 + 源码复核 | `object-concatenate` |
| `attr.window` | 执行 + 输出扰动敏感 | `chebyshev-coefficients` |
| `aug.Add` | 执行 + 源码复核 | `sequence-alias`, `object-augassign` |
| `aug.Mult` | 执行 + 源码复核 | `sequence-alias`, `object-augassign` |
| `binary.Add` | 执行 + 输出扰动敏感 | `chebyshev-arithmetic` |
| `binary.Div` | 执行 + 输出扰动敏感 | `numeric-scalars` |
| `binary.FloorDiv` | 执行 + 输出扰动敏感 | `chebyshev-arithmetic` |
| `binary.Mod` | 执行 + 输出扰动敏感 | `chebyshev-arithmetic` |
| `binary.Mult` | 执行 + 输出扰动敏感 | `chebyshev-arithmetic` |
| `binary.Pow` | 执行 + 输出扰动敏感 | `numeric-scalars`, `chebyshev-arithmetic` |
| `binary.Sub` | 执行 + 输出扰动敏感 | `chebyshev-arithmetic`, `object-full-empty` |
| `bool.And` | 执行 + 输出扰动敏感 | `boolean-membership` |
| `bool.Or` | 执行 + 输出扰动敏感 | `boolean-membership` |
| `call.Chebyshev` | 执行 + 源码复核 | `chebyshev-coefficients`, `chebyshev-arithmetic` |
| `call.Empty` | 执行 + 源码复核 | `object-full-empty` |
| `call.array` | 执行 + 源码复核 | `numeric-arrays`, `object-array-alias` |
| `call.asarray` | 执行 + 源码复核 | `numeric-arrays`, `object-array-alias` |
| `call.ceil` | 执行 + 输出扰动敏感 | `numpy-functions` |
| `call.clear` | 执行 + 输出扰动敏感 | `mapping-pop-clear` |
| `call.concatenate` | 执行 + 源码复核 | `object-concatenate` |
| `call.copy` | 执行 + 源码复核 | `object-slice-copy`, `mapping-update-copy` |
| `call.dict` | 执行 + 源码复核 | `mapping-get-default` |
| `call.empty` | 执行 + 源码复核 | `object-empty-initialization` |
| `call.enumerate` | 执行 + 源码复核 | `iterators` |
| `call.flatten` | 执行 + 源码复核 | `numeric-arrays`, `object-reshape-transpose` |
| `call.float` | 执行 + 输出扰动敏感 | `numeric-scalars`, `string-strip`, `string-replace-join` |
| `call.floor` | 执行 + 输出扰动敏感 | `numpy-functions` |
| `call.full` | 执行 + 源码复核 | `object-full-empty` |
| `call.get` | 执行 + 源码复核 | `mapping-get-default` |
| `call.int` | 执行 + 输出扰动敏感 | `numeric-scalars` |
| `call.items` | 执行 + 输出扰动敏感 | `mapping-live-views` |
| `call.iter` | 执行 + 源码复核 | `iterators` |
| `call.join` | 执行 + 输出扰动敏感 | `string-replace-join` |
| `call.keys` | 执行 + 源码复核 | `mapping-live-views` |
| `call.len` | 执行 + 源码复核 | `object-concatenate` |
| `call.list` | 执行 + 源码复核 | `iterators` |
| `call.log2` | 执行 + 输出扰动敏感 | `numpy-functions` |
| `call.lstrip` | 执行 + 输出扰动敏感 | `string-strip` |
| `call.next` | 执行 + 源码复核 | `iterators` |
| `call.partition` | 执行 + 输出扰动敏感 | `string-split` |
| `call.pop` | 执行 + 源码复核 | `mapping-pop-clear` |
| `call.popitem` | 执行 + 源码复核 | `mapping-pop-clear` |
| `call.pow` | 执行 + 输出扰动敏感 | `numeric-scalars` |
| `call.replace` | 执行 + 输出扰动敏感 | `string-replace-join` |
| `call.reshape` | 执行 + 源码复核 | `numeric-arrays`, `object-reshape-transpose` |
| `call.reversed` | 执行 + 源码复核 | `iterators`, `mapping-live-views` |
| `call.rpartition` | 执行 + 输出扰动敏感 | `string-split` |
| `call.rsplit` | 执行 + 输出扰动敏感 | `string-split` |
| `call.rstrip` | 执行 + 输出扰动敏感 | `string-strip` |
| `call.setdefault` | 执行 + 源码复核 | `mapping-get-default` |
| `call.sorted` | 执行 + 源码复核 | `lambda-sorted` |
| `call.split` | 执行 + 输出扰动敏感 | `string-split` |
| `call.strip` | 执行 + 输出扰动敏感 | `string-strip` |
| `call.transpose` | 执行 + 源码复核 | `object-reshape-transpose` |
| `call.tuple` | 执行 + 源码复核 | `iterators` |
| `call.update` | 执行 + 源码复核 | `mapping-update-copy`, `mapping-live-views` |
| `call.values` | 执行 + 源码复核 | `mapping-live-views` |
| `call.zip` | 执行 + 源码复核 | `iterators` |
| `compare.In` | 执行 + 输出扰动敏感 | `boolean-membership` |
| `compare.NotIn` | 执行 + 输出扰动敏感 | `boolean-membership` |
| `counter.closure_instances` | 执行 + 源码复核 | `closure-nonlocal` |
| `counter.comprehensions` | 执行 + 源码复核 | `comprehensions` |
| `counter.container_writes` | 执行 + 源码复核 | `sequence-alias` |
| `counter.default_evaluations` | 执行 + 源码复核 | `call-defaults` |
| `counter.empty_operations` | 执行 + 源码复核 | `object-full-empty` |
| `counter.helper_calls` | 执行 + 源码复核 | `closure-nonlocal`, `call-defaults`, `call-unpacking` |
| `counter.iterator_advances` | 执行 + 源码复核 | `iterators` |
| `counter.keyword_arguments` | 执行 + 源码复核 | `call-defaults` |
| `counter.lambda_instances` | 执行 + 源码复核 | `lambda-sorted` |
| `counter.loop_breaks` | 执行 + 源码复核 | `while-control` |
| `counter.loop_continues` | 执行 + 源码复核 | `while-control` |
| `counter.loop_iterations` | 执行 + 源码复核 | `object-concatenate`, `loop-else` |
| `counter.mapping_calls` | 执行 + 源码复核 | `mapping-get-default`, `mapping-update-copy`, `mapping-pop-clear` |
| `counter.mapping_views` | 执行 + 源码复核 | `mapping-live-views` |
| `counter.mapping_writes` | 执行 + 源码复核 | `mapping-get-default`, `mapping-update-copy`, `mapping-pop-clear`, `mapping-live-views` |
| `counter.membership_tests` | 执行 + 源码复核 | `boolean-membership` |
| `counter.nonlocal_writes` | 执行 + 源码复核 | `closure-nonlocal` |
| `counter.object_array_operations` | 执行 + 源码复核 | `object-full-empty`, `object-empty-initialization`, `object-array-alias`, `object-slice-copy`, `object-concatenate`, `object-augassign`, `object-reshape-transpose` |
| `counter.public_array_operations` | 执行 + 源码复核 | `numeric-arrays` |
| `counter.public_numeric_operations` | 执行 + 源码复核 | `numeric-scalars` |
| `counter.public_numpy_functions` | 执行 + 源码复核 | `numpy-functions` |
| `counter.public_polynomial_operations` | 执行 + 源码复核 | `chebyshev-coefficients`, `chebyshev-arithmetic` |
| `counter.public_string_calls` | 执行 + 源码复核 | `string-strip`, `string-split`, `string-replace-join` |
| `counter.sequence_operations` | 执行 + 源码复核 | `sequence-slices`, `sequence-alias` |
| `counter.short_circuits` | 执行 + 源码复核 | `boolean-membership` |
| `counter.slice_writes` | 执行 + 源码复核 | `sequence-slices` |
| `counter.sort_key_calls` | 执行 + 源码复核 | `lambda-sorted` |
| `counter.sorted_calls` | 执行 + 源码复核 | `lambda-sorted` |
| `counter.starred_expansions` | 执行 + 源码复核 | `call-unpacking` |
| `counter.while_iterations` | 执行 + 源码复核 | `while-control`, `loop-else` |
| `event.for_else` | 执行 + 输出扰动敏感 | `loop-else` |
| `event.while_else` | 执行 + 输出扰动敏感 | `loop-else` |
| `expansion.kwstar` | 执行 + 源码复核 | `call-unpacking` |
| `expansion.star` | 执行 + 源码复核 | `call-unpacking` |
| `index.read` | 执行 + 源码复核 | `numeric-arrays` |
| `node.Break` | 执行 + 源码复核 | `while-control` |
| `node.Continue` | 执行 + 源码复核 | `while-control` |
| `node.DictComp` | 执行 + 源码复核 | `comprehensions` |
| `node.Lambda` | 执行 + 源码复核 | `lambda-sorted` |
| `node.ListComp` | 执行 + 源码复核 | `comprehensions` |
| `node.Pass` | 执行 + 源码复核 | `while-control` |
| `node.While` | 执行 + 源码复核 | `while-control` |
| `signature.kwarg` | 执行 + 源码复核 | `call-unpacking` |
| `signature.kwonly` | 执行 + 源码复核 | `call-defaults` |
| `signature.posonly` | 执行 + 源码复核 | `call-defaults` |
| `signature.vararg` | 执行 + 源码复核 | `call-unpacking` |
| `slice.read` | 执行 + 源码复核 | `sequence-slices`, `object-slice-copy` |
| `slice.write` | 执行 + 源码复核 | `sequence-slices` |
| `subscript.write` | 执行 + 源码复核 | `object-empty-initialization`, `object-array-alias`, `object-slice-copy`, `object-augassign` |
| `unary.Not` | 执行 + 输出扰动敏感 | `boolean-membership` |

