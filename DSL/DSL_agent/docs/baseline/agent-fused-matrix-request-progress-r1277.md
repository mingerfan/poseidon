# 融合矩阵生成请求接入（r1277）

V14 融合矩阵草案完成从已保存来源包到生成请求的接入验收。
九例均能准备来源绑定、布局计划、公开常量描述符及生成上下文，
并通过请求哈希重建。生产入口保持现有契约。

## 范围与结果

| 项目 | 结果 |
|---|---|
| 来源模型 | 9 例：addmm、baddbmm、addbmm 各 3 个上下文 |
| 数学输入输出 | 每例 3 个逻辑输入、3 个命名输出 |
| 静态物理绑定 | 5–7 个输入密文、5–8 个输出密文，属于计划 |
| 生成上下文 | 41,301–68,203 字节，上限 131,072 字节；公开常量使用描述符 |
| 请求与隐私 | 重建一致、常量内容改变使身份变化、错误哈希和私有字段被拒绝 |
| 能力预检 | accepted_pending_validation；执行状态 unknown |
| 运行环境 | Python 3.10.14、NumPy 1.25.2、Torch 2.0.1 CPU，独立网络命名空间 |

本轮复用 r1268 原始来源、固定权重和 reference 记录，重新核对其绑定。
来源代码没有重新执行。r1274 完成九份请求准备和六个测试方法：
其中五个方法通过，一个隐私断言在九例中误判响应 schema 的字段名。
该字段只声明 `hecate_source` 的类型为字符串，并无候选源码内容。
r1275 校验完整可信 schema 后，仅排除该 schema，九例隐私补验通过。

同一适配代码下，六个当前测试方法有合并通过记录；
r1274 的整体 failed 状态保留，r1275 为 passed。
原始测试错误和 skip 为 0，未将本轮统计加到模型集或历史 Agent 成功数。

## 尚待验收

本轮编译、密态执行、解密和 API 为 0。
静态布局绑定不代表中间密文规模、乘法深度、峰值内存或运行耗时已确定；
这些项保持 unknown。来源包、生成请求和编译程序包分别记录。
新增生成契约尚未接入生产。

704 个生产文件、648 个后端文件、6 个 reference／门限文件保持。
本轮新增内容只在 Agent 开发验收工具和文档中，保留旧修改与 x86 配置，
没有安装、下载、后端／参数改动、提交或推送。

下一步仍是 r1263 原三例 common-call 的 Poseidon 离线续验，
再联合接入来源和生成请求；r1263 和 r1232 付费提案等待各自批准，
冻结绑定保持。

[验收工具](../../scripts/baseline/benchmarks/tools/fused_matrix_request_acceptance_v1/README.md)
· [机器可读记录](agent-fused-matrix-request-progress-r1277.json)
· [验收绑定](../../scripts/baseline/benchmarks/tools/fused_matrix_request_acceptance_v1/acceptance.json)

原始结果：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-fused-matrix-requests-r1274`、
`r1275`；独立审计 `agent-fused-matrix-request-audit-r1276`。

绑定：`2cdcdb8a7e5e22f8c5bea6a45680344043ca530c68bc441a48c2157f48c9b393`。
