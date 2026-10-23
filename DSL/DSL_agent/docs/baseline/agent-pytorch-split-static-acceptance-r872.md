# PyTorch 静态 split 接入 r872

## 结果

torch.split 和 Tensor.split 已接入可信 PyTorch v6/v7 的静态索引捕获策略。相关测试 196 passed、0 failed/errors/skipped；36 份旧矩阵捕获逐字保持。本记录只覆盖代码、明文参考和兼容检查，当前新源码的真实编译、Poseidon CPU 及程序包复用尚未执行。

六种写法共 18 个来源上下文：函数 sections、方法整数 size、关键字参数、直接 tuple 输出、公开分块循环、公开权重分块。41 个来源文件可按字节复现，其中三个 NPY 状态权重。每个上下文用 16 组输入比较原 PyTorch 模块、独立数学解释器、PyTorch 图参考及高层图参考。

## 实现范围

- 拆分大小、sections 与轴来自静态公开值；sections 为正整数且覆盖目标轴。
- 2–4 个内部端口复用已有 split，单端口或更多端口复用 slice，整数 size 保留尾块。
- tuple 迭代与索引在捕获时展开；最终命名输出继续遵守原输出数量和元素限制。
- 公开权重的分块仍由可信捕获与完整来源／状态绑定管理。
- rank 32 的小张量已通过明文检查，密态验收单列。旧非静态索引策略保持原行为。

不修改后端实现或参数。原三项 static_split_stack 的失败记录保留；它们将在新绑定下重新验收，不能用此次单元测试改写旧通过数。

## 绑定与下一步

673 个源码、68 个选定来源文件，以及 648 后端、6 reference 和 12 历史记录通过完整性复核。下载、安装、付费调用、新真实 Agent 生成均为 0。

原一小时工作窗口已关闭：墙钟出现约八小时差距，原因未确认；实际测试作业约18秒且已终态。后端批次未启动。[新的 24 项离线方案](agent-pytorch-split-offline-proposal-r870.md)申请独立30分钟窗口。

原 r848 付费提案按历史绑定保留。当前代码已变化，该未执行提案不能直接用于当前源码付费运行。

[机器记录](agent-pytorch-split-static-acceptance-r872.json) · [使用方式](operator-decomposition-and-python.md#pytorch-张量拆分)。
