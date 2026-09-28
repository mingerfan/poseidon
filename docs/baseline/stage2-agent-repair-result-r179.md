# 真实 Agent 修复批次结果 r179

本次计划19项，独立审计状态：{"failed": 2, "passed": 17}。
首次通过3，修复后通过14。
已审计49次生成、49次HTTP；调用数不代表账单金额。
真实编译尝试24、密态执行尝试19，含失败与重复。
最终停止原因：None；运行1534.9秒。
未审计或中断不计通过；部分批次未审计时上述调用统计只是已核实部分。

使用DeepSeek deepseek-flash/high/stream与已有explicit-v7说明，旧模型/权重/构造要求/编译配置不变。
本轮是重新生成和最多三轮修复，没有发送人工修复DSL或替换模型响应。
26项编译/产物问题继续暂缓；93个编译器源码及二进制不变。
原历史成绩保持原版本，不以择优成绩冒充同一版本全量成功率。
门限保持1e-5+1e-4*abs(reference)，成功项已核验真实tracing/编译/密态/解密与逐项比较。
本轮有界授权已执行；没有自动追加批次、安装、SDK重建、commit、push或PR。

本轮 r178 运行器和预设 r179 汇总器均正常退出（exit=0），无需恢复。
历史 r175 曾发生 None/字符串层级键排序故障；该故障未在本轮复现。
本段已纠正自动模板沿用的历史措辞；机器可读 JSON、分片审计与原始证据未改写。
终态失败分层：{"numerical_comparison": 1, "static_check": 1}。
Provider终止与之前的候选失败分开记录；不能将有历史静态失败的Provider终止记为纯静态终态。
通过项共68组输入、232输出值；
最大绝对误差1.52425279704e-07。
待解决任务共28项，含本轮未通过及26项暂缓编译问题。
这是任务跟踪余额，不是跨版本合并成功率。

## 未通过项

### construct_088_1 — failed

- 尝试0：static_check；Unsupported construction call。
- 尝试1：static_check；Unsupported construction call。
- 尝试2：static_check；Unsupported construction call。
- 尝试3：numerical_comparison；Frozen numerical tolerance failed; inspect reduction/layout/operator semantics。
终态失败层：numerical_comparison。
保留失败层和全部尝试；本轮不自动追加重试。

### free_bench_helper_0113 — failed

- 尝试0：artifact_gate；Invalid level/scale: output[0] remaining_level=1, log2_scale=75; requires 1<=level<=13 and 1<=scale<min(180,60*level)=60。
- 尝试1：static_check；Native core arithmetic requires a ciphertext operand。
- 尝试2：artifact_gate；Invalid level/scale: output[0] remaining_level=1, log2_scale=75; requires 1<=level<=13 and 1<=scale<min(180,60*level)=60。
- 尝试3：static_check；Native core arithmetic requires a ciphertext operand。
终态失败层：static_check。
保留失败层和全部尝试；本轮不自动追加重试。

