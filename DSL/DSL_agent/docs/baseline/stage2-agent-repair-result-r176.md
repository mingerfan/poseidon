# 真实 Agent 修复批次结果 r176

本次计划50项，独立审计状态：{"passed": 39, "failed": 11}。
首次通过16，修复后通过23。
已审计118次生成、118次HTTP；调用数不代表账单金额。
真实编译尝试45、密态执行尝试43，含失败与重复。
最终停止原因：None；运行1240.3秒。
未审计或中断不计通过；部分批次未审计时上述调用统计只是已核实部分。

使用DeepSeek deepseek-flash/high/stream与已有explicit-v6说明，旧模型/权重/构造要求/编译配置不变。
本轮是重新生成和最多三轮修复，没有发送人工修复DSL或替换模型响应。
34项编译/产物问题继续暂缓；93个编译器源码及二进制不变。
原历史成绩保持原版本，不以择优成绩冒充同一版本全量成功率。
门限保持1e-5+1e-4*abs(reference)，成功项已核验真实tracing/编译/密态/解密与逐项比较。
本轮有界授权已执行；没有自动追加批次、安装、SDK重建、commit、push或PR。

汇总脚本原先对None与字符串层级键排序失败；本次用独立修复脚本汇总，
冻结执行脚本、全部原记录及分片审计不变，新增API调用0。
终态失败分层：{"static_check": 7, "provider": 4}。
Provider终止与之前的候选失败分开记录；不能将有历史静态失败的Provider终止记为纯静态终态。
通过项共156组输入、360输出值；
最大绝对误差4.98289444706e-07。
待解决任务共45项，含本轮未通过及34项暂缓编译问题。
这是任务跟踪余额，不是跨版本合并成功率。

## 未通过项

### construct_088_1 — failed

- 尝试0：static_check；Unsupported construction call。
- 尝试1：static_check；Encoding needs scalar/1D length one or four; explicitly reshape, do not flatten silently。
- 尝试2：static_check；Unsupported construction call。
- 尝试3：static_check；Rotation requires a ciphertext and a provisioned public step。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_206_2 — failed

- 尝试0：static_check；Truth value requires public construction data; cipher/array truth is unsupported。
- 尝试1：static_check；Missing contributing public expression: unary.Not。
- 尝试2：static_check；Truth value requires public construction data; cipher/array truth is unsupported。
- 尝试3：static_check；Read-only or invalid binding。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_169_1 — failed

- 尝试0：static_check；Missing contributing public expression: object.named_numeric_constructor。
- 尝试1：static_check；Missing contributing public expression: object.named_numeric_constructor。
- 尝试2：static_check；Unsupported construction call。
- 尝试3：static_check；Missing contributing public expression: object.named_numeric_constructor。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_063_2 — failed

- 尝试0：static_check；Read-only or invalid binding。
- 尝试1：static_check；Encoding needs scalar/1D length one or four; explicitly reshape, do not flatten silently。
- 尝试2：numerical_comparison；Frozen numerical tolerance failed; inspect reduction/layout/operator semantics。
- 尝试3：static_check；Read-only or invalid binding。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_208_1 — failed

- 尝试0：static_check；Ciphertext is not public numerical data。
- 尝试1：static_check；Ciphertext is not public numerical data。
- 尝试2：static_check；Ciphertext is not public numerical data。
- 尝试3：static_check；Missing contributing typed array arithmetic: zero_dim。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_169_0 — failed

- 尝试0：static_check；Symbolic/Empty left operand does not use object-array broadcasting。
- 尝试1：static_check；Object array cell must be a symbolic value, Empty, None or bounded real scalar。
- 尝试2：static_check；Missing contributing public expression: object.named_numeric_constructor。
终态失败层：provider。
Provider元数据：{"error": "empty_or_oversized_content", "finish_reason": "stop", "content_characters": 0}。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_169_2 — failed

- 尝试0：static_check；Symbolic/Empty left operand does not use object-array broadcasting。
- 尝试1：static_check；Symbolic/Empty left operand does not use object-array broadcasting。
- 尝试2：static_check；Rotation requires a ciphertext and a provisioned public step。
- 尝试3：static_check；Symbolic/Empty left operand does not use object-array broadcasting。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_122_2 — failed

终态失败层：provider。
Provider元数据：{"error": "transport_tls_failed", "finish_reason": null, "content_characters": null}。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_027_1 — failed

- 尝试0：static_check；Call target is not a declared construction function。
- 尝试1：static_check；Unsupported construction call。
终态失败层：provider。
Provider元数据：{"error": "transport_tls_failed", "finish_reason": null, "content_characters": null}。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_134_0 — failed

- 尝试0：static_check；Read-only or invalid binding。
- 尝试1：static_check；Missing contributing public expression: event.for_else。
- 尝试2：static_check；Read-only or invalid binding。
- 尝试3：static_check；Missing contributing public expression: event.for_else。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

### construct_034_0 — failed

终态失败层：provider。
Provider元数据：{"error": "transport_tls_failed", "finish_reason": null, "content_characters": null}。
保留失败层和全部尝试；本轮不自动追加重试。

