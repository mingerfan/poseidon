# v21 公开标量提取：逐项真实 Agent 与密态差分验收

## 结论

本轮新启动 **15 例真实 DeepSeek 生成测试，最终 15/15 通过**。
首个有效候选的 parse、static、compile、execute、numerical 均为15/15，
没有语义反馈修复；15个有效候选共发出16次API请求，其中一次响应体读取中断后自动重试。
因此不能将“首次候选15/15”表述为“首次HTTP请求15/15”。

冻结清单的 **19个去重观察项全部有实际类型执行与有限输出影响证据**。
每个候选实际经过 Hecate tracing → Dacapo Earth/CKKS → HEVM/CST →
upstream SEAL HEVM CPU 加密执行 → 解密 → 独立reference差分。
这不是Poseidon GPU结果，也不是完整上游DSL、任意类型组合或所有输入的证明。

## 本轮实现范围与版本

新契约request `hecate-function-synthesis-v21` 对应检查契约
`hecate-function-v20`，构造展开metadata schema15；通过 `--scalar-conversion` 显式启用。

- 公开float/int零参数默认值、bool/有限数值/数字字符串，以及int(string,base)。
- 数值数组和公开对象数组的size-one转换；int对负小数向零截断。
- 固定NumPy1.25.2：0-D数组转换正常；正rank单元素数组转换属于弃用路径，
  在构造诊断 `legacy_array_scalar_conversions` 中记录。没有换NumPy版本或压低参数。
- `.item()`、平坦索引（含负索引）、tuple索引、多位置参数索引。
- 将原始公开常量构造成dtype=object数组时保留数值存储shape和数值单元。
- 对象数组中的cipher `.item()` 返回符号Expr句柄，绝不解密。
- 密文/派生Plain Expr到float/int、多元素数组转换、非法索引、
  None/Empty转换、非有限数值、超限构造继续拒绝。

保留旧v19/v20请求和清单，不把以前失败的数组转换追认成旧版本成功。
新语法仍先经可信受限解释器检查/展开；不直接exec生成的Python，
不增加候选的文件、网络、进程或任意Python强制转换权限。

## 冻结案例与覆盖定义

14例仿射 `y=0.5*x+x+0.375`，1例二次 `y=0.5*x*x+x+0.375`。
每例shape[4]，公开固定权重、加密输入，四组固定测试输入。
这里15例指指定构造形式，不是15个网络家族或未见结构泛化实验。

- 清单：`scripts/baseline/scalar-conversion-exercises-v1.json`
- manifest：`scripts/baseline/cases/scalar-conversion-exercises-15-manifest.json`
- 检查器：`scripts/baseline/scalar_conversion_exercises.py`
- 实验要求和模型随请求哈希冻结；人工答案、测试输入、reference数值和密钥不进入请求。
- 观察事件包含真实接收者存储类型、shape、索引形式、截断/legacy类别、原始公开常量构造和重叠存储事实。
- 仅AST出现拼写、死代码或未参与输出的操作不能通过。
- 提取结果加一、重叠乘改加等有限扰动必须改变探测输出。扰动不是额外FHE执行，也不是全输入证明。
- 最终15份源码已逐份复核。

| 观察项 | 真实通过案例 |
|---|---|
| `cast.float.bool` | `sc-bools` |
| `cast.float.default` | `sc-defaults` |
| `cast.float.numeric.legacy` | `sc-float-single`, `sc-overlap` |
| `cast.float.numeric.zero` | `sc-float-zero` |
| `cast.float.object.legacy` | `sc-object-constant` |
| `cast.float.object.zero` | `sc-object-cast` |
| `cast.float.str` | `sc-strings` |
| `cast.int.base` | `sc-strings` |
| `cast.int.bool` | `sc-bools` |
| `cast.int.default` | `sc-defaults` |
| `cast.int.numeric.legacy` | `sc-int-single` |
| `cast.int.numeric.negative_fraction` | `sc-int-negative` |
| `item.numeric.flat_negative` | `sc-item-flat` |
| `item.numeric.positional` | `sc-item-pos` |
| `item.numeric.sole` | `sc-item-single` |
| `item.numeric.tuple` | `sc-item-tuple` |
| `item.object.cipher` | `sc-item-cipher` |
| `object.named_numeric_constructor` | `sc-object-constant` |
| `object.overlap_mult` | `sc-overlap` |

这个清单不穷举所有维度、所有索引边界或cast×storage×rank的笛卡尔积。
例如int的0-D对象数组转换虽有本地NumPy对照测试，未由本批独立定向Agent案例验证。
后续扩展应使用独立版本/补充清单，不能修改本批冻结要求。
19项也不能与旧117项、对象算术10项简单相加，宣称全DSL语义数量。

## 真实付费结果

| 指标 | 结果 |
|---|---:|
| 首个有效候选 parse/static/compile/execute/numerical | 15/15 |
| 最终数值通过 | 15/15 |
| 有执行与影响证据的清单观察项 | 19/19 |
| 反馈修复轮数 | 0 |
| API请求次数 | 16 |
| 传输重试 | 1 |
| 有usage的请求 | 15 |
| 已返回prompt tokens | 99338 |
| 已返回completion tokens | 108192 |
| 已返回total tokens | 207530 |
| 缺失usage的请求 | 1 |
| 密态输入组 | 60 |
| 逐标量比较 | 240 |
| 加权MAE | 4.863992410346814e-9 |
| 最大绝对误差 | 3.3573417601573396e-8 |
| 批次耗时 | 212.1535秒 |

207530是收到usage的合计，不是完整账单；中断请求可能也产生费用，
实际金额未查询、未估算。不得据此断言中断请求免费。

配置：DeepSeek `deepseek-flash`、high、输出上限384000 tokens、
单请求超时1200s、最多3次传输重试、初始生成+最多3轮修复，
API并发10、native并发2、命令级代理127.0.0.1:6478。
有限上限保持；未通过无限重试或放宽正确性标准获得成功。

| 案例 | HTTP请求数 | 最大绝对误差 | 证据目录 |
|---|---:|---:|---|
| `sc-defaults` | 1 | 2.196790e-8 | `agent-deepseek-pdkqcu5c` |
| `sc-bools` | 1 | 3.357342e-8 | `agent-deepseek-bjgfugjf` |
| `sc-strings` | 1 | 1.426167e-8 | `agent-deepseek-rhvppmzv` |
| `sc-float-zero` | 1 | 9.885942e-9 | `agent-deepseek-j9003bgn` |
| `sc-int-negative` | 1 | 1.158681e-8 | `agent-deepseek-stlwbgmc` |
| `sc-float-single` | 1 | 1.345857e-8 | `agent-deepseek-dyc7d4cd` |
| `sc-int-single` | 1 | 1.593987e-8 | `agent-deepseek-wslz40wf` |
| `sc-object-cast` | 1 | 1.013903e-8 | `agent-deepseek-09a2oxzd` |
| `sc-item-single` | 1 | 1.952817e-8 | `agent-deepseek-nuoo840y` |
| `sc-item-flat` | 1 | 1.089899e-8 | `agent-deepseek-k_rx698w` |
| `sc-item-tuple` | 1 | 1.810535e-8 | `agent-deepseek-y499cvq0` |
| `sc-item-pos` | 1 | 1.670299e-8 | `agent-deepseek-34lnwcx_` |
| `sc-item-cipher` | 1 | 1.342120e-8 | `agent-deepseek-0rzy77qk` |
| `sc-object-constant` | 1 | 1.282898e-8 | `agent-deepseek-zbnco8o7` |
| `sc-overlap` | 2 | 8.553775e-9 | `agent-deepseek-4bpog0q6` |

## 唯一传输异常

`sc-overlap`第一次请求收到HTTP200，但在body阶段抛出
`transport_incomplete_read`（55.281秒）。5秒退避后，同一generation重试成功
（47.845秒），模型返回身份仍为deepseek-flash，finish_reason=stop。
这是传输重试，不是根据compiler反馈修复DSL；没有可供检查的首份完整候选。
当前证据不支持将其称为证书/TLS握手失败，也不保证以后不会再次发生。

启动前三次无凭据连接探测均为HTTP401、SSL verify result0。
401只说明未认证路由有响应，实际认证成功由后续真实请求证明。
没有关闭证书验证、持久化代理或改系统网络配置。

## 人工基线和负对照

初始批次：
`/home/lhy/poseidon-work/results/scalar-conversion-goldens-py5psmig/report.json`

SHA256：`db8511559bbf5bf2790610b6b83baec3a02e56b37ecb03612f789c8dd3fbfcbc`

原始17项中15个正确程序通过，错误截断被数值差分拒绝；
错误索引最初取到零，触发SEAL `result ciphertext is transparent`，
提前在seal_runtime终止。原批次保留status=failed，不能改记17/17差分验收。

将负对照夹具改成“错误但非零的系数”后，仅重跑该反例：
`/home/lhy/poseidon-work/results/scalar-index-counterexample-ajr1prlg/report.json`

SHA256：`158accdde157790683de5afa04b2ea8621843f371608e34731b66d8db6e94107`

补验成功在numerical_comparison失败，证明该负对照能到达并被差分拦截。
未修改正确模型、输入、reference、门限或安全参数，也未放开透明密文检查。
原失败fixture内容仍在原run冻结payload中保留。

另一个实现期测试失败来自使用两元素公开常量夹具，违反已有“标量或四元素”入口限制；
改为四元素夹具验证shape保留，没有放宽模型输入契约。

## 安全参数、审计与清理

SEAL4.0.0、degree32768、modulus bits=[60]*14、tc128检查；
逐项门限仍为 `abs(actual-reference) <= 1e-5+1e-4*abs(reference)`。
没有bootstrap、decrypt-and-reencrypt、GPU假结果或新CPU backend。

原始付费批次：
`/home/lhy/poseidon-work/results/agent-batch-1codrfmv/report.json`

SHA256：`d10ee56c917cbc2ed3d92b5a22ec65c269caae81f552f6ec99490de7fa90e077`

最终离线审计：
`/home/lhy/poseidon-work/results/scalar-conversion-agent-audit-vd4v_fjc/report.json`

SHA256：`a3cf96b7781d69dbe16aab289134b039bcb8a0980e9680f4c76b9a4fea1e927a`

离线审计重新检查模型/清单、request及trace payload哈希、候选静态结果、
实际Hecate trace标记、Earth/CKKS/HEVM/CST证据、SEAL安全/密钥验证、
独立公式与保存reference的关系、解密值及逐项误差。
不读取API key、不发新请求、不新增FHE执行。

最终回归 **368项：349通过、19条件跳过、0失败**。
显式启用本轮人工密态证据、付费15例审计及旧v20人工/付费证据，它们不属于跳过项。
旧v19付费记录另行重新审计：仍为30/30案例、117/117观察项，0新增API。
历史数据仍按原版本统计，不覆盖失败记录。

自动清理：
- 本轮付费案例临时测试密钥：10,184,700,150 bytes（约9.49GiB）。
- 人工初始批次：11,542,660,170 bytes。
- 人工反例补验：678,980,010 bytes。
- 原始响应、模型、源码、IR、artifact、解密结果和失败日志保留。
- 随机测试密钥删除后不能恢复，可重新生成新密钥；本地.env/API key未改动。
- 最终WSL可用空间约189.18GiB；付费worker已全部结束。

## 复现

只查看清单，不加载凭据、不收费：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan --scalar-conversion-exercises
```

离线审计已完成批次：

```bash
timeout -k 10s 500s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/audit_scalar_conversion_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-1codrfmv
```

本次实际付费命令（再次执行会产生新费用）：

```bash
timeout -k 10s 24000s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_agent_batch.py --deepseek --scalar-conversion-exercises \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 1200 --provider-retries 3 \
  --loopback-proxy-port 6478 --jobs 10
```

人工基线脚本为 `run_scalar_conversion_goldens.py`，无API调用；
再次执行使用当前非零错误索引fixture，属于新的人工实验，不覆写本次原始失败批次。

在既有固定Nix/Python环境中设置以下路径，再运行
`python -m unittest test_scalar_conversion test_scalar_conversion_exercises test_scalar_conversion_evidence`：

```text
POSEIDON_SCALAR_CONVERSION_REPORT=/home/lhy/poseidon-work/results/scalar-conversion-goldens-py5psmig/report.json
POSEIDON_SCALAR_INDEX_COUNTEREXAMPLE=/home/lhy/poseidon-work/results/scalar-index-counterexample-ajr1prlg/report.json
POSEIDON_V21_AGENT_BATCH=/home/lhy/poseidon-work/results/agent-batch-1codrfmv
```

## 本地变更与后续边界

新增标量转换模块、冻结15例规范和manifest、人工fixture、测试、付费审计与本文档；
修改可信normalizer、版本契约、trace、沙箱只读模块列表、单例及批量入口。
无依赖安装、sudo、系统配置、模型/安全参数变更；未改C++/GPU实现。

分支仍为feat/agent-dsl-correctness，HEAD仍为4995e7cadedf2bfb9104658b5638662ecf6a1d0a；
原有dirty文件和Dacapo子模块状态保留。修改仅留本地，不commit/push/PR。
建议未来经批准分为：标量语义与门禁、定向Agent覆盖框架、实验资料三个本地commit。

本轮解决的是新增构造的受限真实Agent证据。完整上游DSL、未覆盖类型组合、
一般packing、真实高层helper、bootstrap/upscale与Poseidon GPU端到端仍须独立推进；
不能以本轮成功把完整目标标记完成。
