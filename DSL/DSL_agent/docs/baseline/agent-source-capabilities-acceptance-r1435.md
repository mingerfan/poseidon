# 应用能力清单一致性

宿主接口现按实际来源契约返回结构化输入与空间调用的版本范围，并补齐静态广播：

| 字段 | 当前范围 |
|---|---|
| input_call.versions | v26、v27、v28 |
| spatial_calls.versions | v27、v28 |
| static_broadcast | v28 的 Tensor.expand_as、Tensor.broadcast_to |

组件接口与应用接口返回相同规则。168 个不同控制测试通过，其中新增 9 项，失败、错误和跳过均为 0。6 份原来源请求（3 个不同请求）的文件哈希及重构字节保持一致；其余能力字段未改变。

此次仅修改 Agent 的能力描述、测试和文档，API、源码捕获、真实编译及密态调用均为 0。已登记与真实验收状态分别表达；能力清单不读取历史实验报告。

当前 SDK 为 `65e3d2514d448221fb61a1d410a812d2bd1908f483db892c63595b2a3ad0a7b2`。四阶段保守身份中的唯一源码差异都是 application_capabilities.py；Poseidon 后端、参数和二进制身份保持。旧 v28 密态结果继续属于 `5a88df990a07ead9e19944721146d997b04e23408b3e0266cff497ced7658343`，没有重新绑定成当前 SDK 通过。

审计器首次因假 Provider 的重试日志穿插在 unittest 状态前而漏数一项。修正日志识别后只重做只读审计，168 项测试没有重跑；原日志与工具副本保留。

结果目录仍为 34,354,462,720 字节，32 GiB 上限剩余 5,275,648 字节。r1434 的 48 GiB 产物预算仍待批准。批准后优先重放少量原样真实 Agent DSL，验收新身份下的应用与程序包接口，再继续通用输入与算子工作。

- [机器可读验收](agent-source-capabilities-acceptance-r1435.json)
- [原 v28 真实验收](agent-static-broadcast-phase-summary-r1433.md)
- [产物预算提案](agent-result-storage-proposal-r1434.json)
