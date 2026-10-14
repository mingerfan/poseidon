# 规则表达排程：静态验收

当前完成 Agent 层适配与明文差分，八项历史编译失败尚未重新验收。

| 检查 | 结果 |
|---|---|
| 相关单元及兼容测试 | 67 passed，0 failed/errors/skipped |
| 历史请求参考检查 | 48 份记录、47 个请求绑定、768 组 |
| 新表达 | 43 份源码变化，均通过双参考差分 |
| 代表例重新捕获 | 12/12，每例 16 组原 PyTorch 与独立参考 |
| 请求、模型、权重和布局 | 保持原绑定 |
| 后端工具、参数及参考实现 | 保持 |
| 本轮真实编译与密态执行 | 待执行 |
| 新 Agent 生成及付费调用 | 0 |

## 修改

`capacity_algebraic_schedule.py` 增加完整周期的公开乘法恒等项检查，排程版本为 `capacity-algebraic-schedule-v2`。`unified_graph_scheduling.py` 在容量排程中识别同一完整周期二进制掩码的幂等性，消除重用公开值时引入的乘 1。`unified_capacity_lowering.py` 按图和绑定运行通用排程，沿用现有 AST、展开量和源码限制。

短向量、补零尾块、近似 1、布尔值、非二进制权重、不同掩码以及旋转后的掩码分别检查。规则基线源码可以调整；Provider 候选不经过这项重写。原始模型、公开权重、reference 和参数保持。

直接容量 lowering 与旧标量排程保留原默认行为，旧请求重建保持兼容。源码与排程版本改变后使用新的验收绑定。

## 原失败图

| 原案例 | 旧排程的保守深度 | 新排程的保守深度 | 当前状态 |
|---|---:|---:|---|
| attention_core_0–2 | 15 | 15 | 编译重测待执行 |
| channel_broadcast_mix_0–2 | 17 | 13 | 编译重测待执行 |
| norm_depth_branches_1 | 16 | 13 | 编译重测待执行 |
| norm_depth_branches_2 | 19 | 16 | 编译重测待执行 |

深度为规则 DAG 的诊断值，不能替代 Dacapo 的类型与容量检查。原失败和历史绑定均保留。上一批静态 split 的 24/24 Poseidon 结果属于 r878 的独立绑定。

本轮只修改三个 Agent 文件并新增 `test_capacity_neutral_factors.py`。原 1827 条本地修改与未跟踪记录保留。新安装、下载、SDK 重建、后端修改、参数及门限变更、Git 发布均为 0。

提案准备工具曾使用错误字段名 `compiler`，修正为 `compiler_configuration` 后 12 项准备通过；旧工具失败日志保留。这是提案工具故障，未运行编译或密态程序。

[机器报告](agent-capacity-scheduling-static-acceptance-r885.json) · [下一批离线验收](agent-capacity-neutral-offline-proposal-r892.md)。
