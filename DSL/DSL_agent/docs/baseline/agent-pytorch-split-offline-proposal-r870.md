# 静态 split 离线验收 r870

代码与相关单元测试已完成：196 passed、0 failed/errors/skipped；36 旧矩阵捕获逐字一致。新来源集为六组、18 个上下文，41 个文件可复现，包含三个公开 NPY 权重。

本阶段记录的一小时工作窗口已出现约八小时墙钟差距，原因未确认。实际测试作业约18秒并已终态，后端验收尚未启动。原时间记录保持；本提案请求新的30分钟独立离线验收窗口，不追加任何付费额度。

## 范围

- function_sections_0：[[2, 4]]
- function_sections_1：[[4, 2]]
- function_sections_2：[[1, 2, 6]]
- keyword_sections_0：[[4]]
- keyword_sections_1：[[1, 6]]
- keyword_sections_2：[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8]]
- method_size_0：[[5]]
- method_size_1：[[2, 6]]
- method_size_2：[[1, 2, 7]]
- public_loop_0：[[10]]
- public_loop_1：[[2, 12]]
- public_loop_2：[[1, 1, 14]]
- public_weight_0：[[1, 3]]
- public_weight_1：[[2, 3]]
- public_weight_2：[[1, 2, 3]]
- tuple_output_0：[[4]]
- tuple_output_1：[[2, 6]]
- tuple_output_2：[[1, 2, 8]]
- static_split_stack_0：[[2, 4]]
- static_split_stack_1：[[1, 6]]
- static_split_stack_2：[[2, 6]]
- cipher_batch_1：[[1, 2, 1, 2, 2], [2, 1, 3, 2, 1]]
- module1_frozen_1：[[2, 2, 2]]
- cnn_norm_head_1：[[1, 2, 3, 3]]

共24项：原失败 split 三项、新 split 来源18项及三个兼容回归。保持原模型、权重、reference、w45参数和数值门限。使用规则候选，真实 Agent 生成为0。

## 验收与限制

捕获沙箱、双参考、原三项等价差分、真实编译与Poseidon CPU、四组逐项数值；三个代表包重载、三新输入各执行两次和缓存复用。整批30分钟，每例最多180秒；编译≤2／链接≤1／密态串行，新增产物≤4GiB／全局≤32GiB／余量≥4GiB。超限保留现场；无安装、下载、参数或后端改动，无API。

绑定 43a32b101fe36b9d8d617bb44b833058020a9dcfbeafd5bc9f4adea4b4e5bc16；673个源码和68个来源文件。原r848付费提案按历史绑定保留，源码已经变化，未执行；需在离线验收后另行刷新，不能用于当前代码付费运行。

[机器提案](agent-pytorch-split-offline-proposal-r870.json)。
