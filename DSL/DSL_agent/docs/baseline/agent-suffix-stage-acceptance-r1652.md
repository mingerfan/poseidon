# 静态行归约的生成指导

新增可选 explicit-v20，区分窗口求和、gather、均值归一化、紧凑输出和后续广播。
生成上下文根据图形状和现有 packing 绑定提供来源、行宽、行数、输出形状与掩码归一化。内容为数学和坐标语义，没有确定性 DSL 答案。旧 v1–v19 请求的重建和哈希保持。

## 用法

~~~sh
python3 scripts/agent.py --backend local candidate -- \
  --case model.json --prepare --unified-guidance explicit-v20
~~~

应用接口对应 options={"generation_guidance":"explicit-v20"}。非二幂行窗口使用已有 capacity-public-tensors-v10 packing；参数范围保持原定义。

## 本批验收

| 项目 | 结果 |
|---|---|
| 新增与兼容检查 | 61 passed，0 failed／error／skip |
| 真实后端规则案例 | 3/3 |
| 实际编译／Poseidon 密态调用 | 3／9 |
| 初次四组输入 | 12 组、36 值，max abs 3.63383e-07 |
| 新输入与重复执行 | 12 组、36 值，max abs 3.69948e-07 |
| 包导出／重载／复用 | 3/3 |
| 付费 API | 0 |

三个案例为原 rsqrt_mean、非二幂窗口求和、多密文均值。门限保持 1e-5 + 1e-4*abs(reference)。候选为规则基线；v20 的真实 Agent 生成质量仍待验收。

旧失败保留：首次夹具格式错误4个error，非二幂夹具未选v10产生1个failure。补齐新模块沙箱只读登记前，3例因模块缺失在tracing前停止，编译／密态0；修复后建立新绑定复验。

## 真实 Agent 状态

[前一批9项](agent-source-preparation-live-results-r1647.md)中8项首轮通过，原 rsqrt_mean 修复预算耗尽。使用v19重测时，前两轮仍数值失败，第三轮HTTP200接收过程中触及stream_limit，终态service_error。3生成／3HTTP，没有传输重试。现有回执不能区分总流与单行限制，网络故障未证实。该项仍未解决。

[重测机器记录](agent-source-preparation-retry-results-r1649.json)。v20补齐了相关说明，完成当前规则基线验收；下一批以新身份冻结生成评测。历史8项保留原SDK929e8131…，不提升为v20通过。

## 环境与后续

当前 SDK `eb3a7846ba59c85397019d56332b447c6400d2435e8d098e8b84cf741d88cd81`。仅修改 Agent 指导、反馈／CLI接线和新模块只读登记；Poseidon、Runtime、Dacapo/Hecate、参数、模型、reference、门限和网络保持。
结果 38.966/48 GiB，磁盘余量 30.315 GiB。唯一开发 checkout 为 ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl；x86及已有修改保留。

下一步冻结新指导的三个生成上下文，按现有授权与有界账本测试；继续模块校准、原函数精度、算子、容量、bootstrap、CUDA与整模型目标。

[机器报告](agent-suffix-stage-acceptance-r1652.json)。
