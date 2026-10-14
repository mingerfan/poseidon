# MHA v12 真实 Agent 测试结果 r1335

r1333 批次已停止。两项任务中，1 项生成预算耗尽，1 项随 Ubuntu 客体重启中断。通过数为 0。

| 案例 | 结果 | 生成 / HTTP | 失败层 |
|---|---|---:|---|
| self Attention → Linear → 残差 | 预算耗尽 | 4 / 4 | 静态检查 |
| 投影后的 cross Attention | 中断，最后一次调用结果未知 | 1 / 4 | Provider / worker |

## Self Attention

四份响应均已保留，候选源码未改：

| 轮次 | 原检查结果 | 定位 |
|---|---|---|
| 0 | 展开工作量 2766，限额 1024 | 入口第 42 行 |
| 1 | 保留绑定被覆盖 | 第 55、56 行重新赋值公开常量 `p10`、`p11` |
| 2 | 展开工作量 3205，限额 1024 | 入口第 10 行 |
| 3 | 旋转步数不在允许集合内 | 第 23 行，第 45 列 |

第 1 轮的原记录为 `contract_rejected`。恢复后使用相同源码复核，确认原因是公开常量赋值；原响应、源码与检查记录保持。新检查器提供 `native_reserved_binding` 和位置，详见 [离线修复验收](agent-supervision-binding-feedback-r1336.md)。

本案例没有进入编译、密态执行或数值比较，尚无本轮精度结论。

## Cross Attention 与中断恢复

受控传输记录显示：第一次调用在收到 HTTP 200 和部分流内容后发生 `transport_incomplete_read`；第二、三次为 `transport_dns_failed`；第四次尚未取得结果时，客体重启。

恢复时已核验旧 worker 消失、客体启动时间晚于批次开始时间。通过现有 `Application.recover()` 将孤立任务标为 `service_error / interrupted_outcome_unknown`，保留已用额度。恢复没有重发请求。

原 `report.json` 的 `running` 状态、原任务状态副本和临时文件均保留。本报告及 [机器报告](agent-mha-tensor-axis-live-results-r1335.json) 是恢复后的汇总。

## 调用和执行

- 已登记 5 次生成、8 次 HTTP；其中 1 次 HTTP 的结果及收费情况未知。
- 收到 4 份 usage，合计 487,018 tokens；这是已收到记录的总量，货币费用未知。
- 编译、Poseidon 密态执行、解密、数值验收和程序包交付均为 0。
- 恢复后新增付费调用、安装、下载、后端／参数修改、提交和推送均为 0。

原批次授权已关闭。新调用需要新的明确范围与批准，未知调用保持原状态。

原始目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-tensor-axis-live-v1`。方案绑定 `475391a7ef31efef047900e89fe39db9bbc70d63ea63a900e1e88c0f82ae2d1f`，恢复汇总绑定 `1a615bb82c71b8a22bc2d06f192c0a732a06ac13fb4346c60dee0d20c4e97f97`。
