# 原生星号参数：逐项 Agent 覆盖前的验证门禁

2026-09-17：8 项定向构造已接入生成请求、静态检查、真实 Hecate tracing 和独立审计。
人工密态门禁 8/8 通过，错误数学表达反例被数值比较拒绝。
本阶段没有新付费 API 调用，不能把人工代码计为 Agent 生成成功。
此前获批的 native-function 11 项已完成，不因收到同一批准回复重复计费。

## 逐项要求与证据

| 案例 | 必须使用的构造 |
|---|---|
| ns-list | list 星号展开 |
| ns-tuple | tuple 星号展开 |
| ns-array | 一维 Expr 数组展开 |
| ns-reverse | 负步长数组切片后展开 |
| ns-matrix-flatten | 矩阵显式展平后展开 |
| ns-nested | helper 返回容器再展开 |
| ns-multiple | 多个非空星号段与普通位置参数混合 |
| ns-empty | 空容器展开、可达无参数 helper 调用 |

共 9 个构造特征分区。前 8 个有固定公开探针上的有限参数影响证据；
空展开仅有结构证据，不能声称不存在的参数具有数值影响。
非空调用要求所有位置参数分别受到扰动时均能影响最终输出，防止用死代码、
未使用参数或抵消调用伪造覆盖。有限探针不是全输入证明，也不证明构造无法被代数化简。

`native_star_exercises.py` 在静态允许列表检查后解释有限 AST，不执行候选 Python。
可信 tracing 在实际 Hecate native 调用中记录 caller/callee、源码位置、各参数段的
容器种类、shape、顺序和数量；`verify_trace_coverage` 要求这些观察与静态见证对齐。
这个观察是 frontend 调用证据，不是动态 HEVM 调用次数。

新任务版本 `hecate-native-function-synthesis-v8` 复用星号语法契约
`hecate-native-functions-v3`，未与数组算术、公开循环或旧 v22 自动合并。
固定 8 项入口为 `run_agent_batch.py --native-star-exercises`；旧请求契约不变。
请求包含构造要求和显式 compiler 配置身份，不包含人工源码、测试输入数组或逐项 reference。

## 真实人工密态结果

链路：人工候选 → 逐项检查 → Hecate native tracing → Dacapo → HEVM/CST →
upstream SEAL HEVM CPU → 解密 → 独立明文差分。

- 配置：显式 `seal-cpu-eva-w45-v1`，默认 waterline40 未改变。
- 安全参数：SEAL4.0.0、N32768、14×60-bit 模数、tc128。
- 数学模型：`1.5*x + 0.375`；这是 8 种构造，不是 8 个模型家族。
- 8 正例共 32 组输入、128 个输出值。
- 最大绝对误差：`8.129851236660102e-10`；加权 MAE：`1.4385749275841464e-10`。
- 原数值门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 错误反例通过构造门禁，但数学方向写反，密态差分最大误差 `3.0000000004004788`，
  在 numerical_comparison 层正确拒绝。构造合规不等于数学正确。
- 没有 bootstrap、Poseidon GPU 执行或付费 Agent 生成。

批次：`/home/lhy/poseidon-work/results/native-star-exercise-goldens-a7vok9vv/report.json`。
SHA256：`cd7cb791622e05717ee4f5658d13ea60c4ec9584f99d7abf70d24ad0905b5d41`。

独立审计：`/home/lhy/poseidon-work/results/native-star-exercise-audit-k3k8atoy/report.json`。
SHA256：`87eced0da4a4c96c7376075a8ecafe7eeba424e3240e35973aa6d2051386e4be`。

审计复核候选与 trace payload 一致性、输入与产物哈希、实际 HEVM metadata、
显式 compiler 配置、独立明文公式、解密数组和密钥清理；审计本身不重新调用 API 或执行 FHE。
默认审计入口要求 live Agent 来源；人工批次必须显式 `--manual`，不能改标签冒充付费结果。
catalog SHA256：`841646f34d9c7052e4dc227e3b286200c484aaafaac1628590ca95932879af90`。

9 个运行的临时密钥已清理，共 6,110,820,089 字节（约 5.69 GiB），不可恢复；
保留源码、冻结输入/reference、IR、HEVM/CST、解密结果、日志和报告。

## 离线复核与后续门禁

```bash
cd '/mnt/d/Code Space/Poseidon'
# 仅审计已有结果；不会调用模型 API 或再次执行 FHE
timeout -k 5s 280s python3 scripts/baseline/audit_native_star_batch.py \
  /home/lhy/poseidon-work/results/native-star-exercise-goldens-a7vok9vv --manual
```

新增 `test_native_star_exercises.py` 的 7 个单元测试与 2 个真实产物证据测试，
覆盖死代码、被忽略参数、抵消调用、伪矩阵展平、真实调用观察不匹配、数学错误、
旧请求边界、provider 数据范围和人工/Agent 证据隔离。

包含以上 9 项、既有 native/数组/循环、显式 compiler 配置、provider/batch、
旧 v22、bootstrap 前端边界、清单和保留策略的综合回归：214 项通过，
0 失败、0 跳过（13.355 秒）。其中已完成的 11 项付费证据也重新离线审计通过；
这些测试不新增付费调用，不把历史结果当作新生成。

新增 8 项付费范围询问与旧 11 项不同，尚未收到对应答复；不能复用旧批次授权扩大范围。
数组构造付费验证、数组算术/循环的专门逐项门禁、native 与旧契约组合、
高层 helper 和真实 bootstrap 仍有剩余工作。`all_semantics_verified` 保持 false。

所有修改保留在本地；未安装依赖、修改凭据、切换分支或提交/推送。
建议未来把星号构造门禁、可信观察、测试和文档作为一个独立本地提交，待明确提交授权。
