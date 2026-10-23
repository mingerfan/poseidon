# 应用控制层 Poseidon CPU 回归

2026-10-11，ARM Ubuntu，开发 checkout `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

## 结果

三项代表性回归全部通过。实际完成 **3 次编译、9 次 Poseidon CPU 密态调用、24 组次和 240 个输出值**，其中 3 组为重复输入。最大绝对误差 **5.72765779e-09**，门限保持 `1e-5 + 1e-4*abs(reference)`。

| 模型 | 结果 | 输入组次 | 输出值 | 最大绝对误差 |
|---|---|---:|---:|---:|
| f64_to_f32_c0 | 通过 | 8 | 48 | 4.259278e-09 |
| f64_roundtrip_c2 | 通过 | 8 | 96 | 5.727658e-09 |
| compat_v23_named | 通过 | 8 | 96 | 5.567005e-09 |

每个模型完成真实 Hecate tracing、Earth/CKKS 编译、RuntimePlan／OperatorSpec／常量 bundle 检查、四组输入的密态执行与独立 reference 比较；随后重载程序包，执行三组新输入及一组重复输入，并确认再次提交时直接复用已验收包。命名输出保持字典顺序及列表中的 `None`。

程序来源为原冻结规则候选。原模型、权重、请求、DSL 和 reference 保持；本批真实 Agent 新生成、API 和 HTTP 均为 **0**。本地 `execute` 返回原执行记录，追加输入的数值结论来自独立比较文件，未改写该记录的验证字段。

## 控制层验收

已确认新 `admission_policy="structured-v1"` 和 `operational_limits` 接通来源捕获、验证目录、程序包导出和本地执行。任务产物账本覆盖已登记目录；独立执行使用自己的产物窗口。重复提交产生缓存命中，未重新生成、编译或密态执行。

199 个不同组件和适配器控制通过，0 failed／skipped；该静态结果按 [r1381](agent-application-admission-budget-checkpoint-r1381.md) 的原日志与源码绑定复核。真实后端回归仅覆盖上表三例。

计量为 `monitored-v1`，记录检查点观察到的保留字节峰值；`hard_filesystem_quota=false`。本批成功路径已验收，预算拒绝、取消和完整性故障依照组件与受控适配器测试计分。

## 执行记录

- 计划使用 [r1382](agent-application-control-offline-proposal-r1382.md)，用户批准三例；调用及资源范围保持。
- 首次启动遗漏模块搜索路径，在编译前停止。修正启动命令后，首例已通过数值验收，但收集脚本读取了不存在的顶层 `model_source` 字段。
- 正确字段为 `manifest.origin.model_source`；程序包来源还经正式 `check_package` 校验。原失败报告和日志保留。
- 收集续验从首例已完成的程序包继续，只编译两个尚未执行的模型，总次数仍为 3 编译／9 密态。工具修正单独建立 [续验绑定](agent-application-control-offline-continuation-r1382.json)，原冻结计划、源码及报告未改写。
- 从首次启动到执行结束耗时 584.70 秒，保持原 720 秒边界。续验 worker 为 187.40 秒；后续只读审计不编译或执行。
- 本批产物 40.81 MiB，结果总量 28.69 GiB，可用磁盘 41.02 GiB，均在原批准范围内。
- 安装、下载、网络变更、后端库及参数修改、commit/push 均为 0。Linux x86_64 配置保留，本批实际执行平台为 ARM64。

4469dc8dea04cdee31c354f728a830d23f447d702836d2d8e8a057c9014e9c86：`4469dc8dea04cdee31c354f728a830d23f447d702836d2d8e8a057c9014e9c86`。当前编译/执行/验证身份与本批产物绑定；r1376/r1380 及其旧包保持历史身份。

## 后续

已确认新控制选项下的代表性真实后端与程序包流程通过。真实 Agent 对新来源写法的生成质量仍需另列有界评测。通用算子、布局容量、bootstrap、整模型及 CUDA 工作继续原计划；本批不改变它们的验收状态。

[机器报告](agent-application-control-offline-results-r1383.json) · [组件接口](application-agent-component-v1.md#接入诊断与产物预算) · [接续](mac-migration-handoff.md)

最终只读复核通过：174 个来源/证据文件及 SDK 身份保持，545 个现有文档链接有效。只读脚本首次调用的状态元组长度错误已修正，原日志保留；此复核新增编译、密态和 API 调用均为 0。[复核记录](agent-application-control-readonly-inspection-r1383.json)。
