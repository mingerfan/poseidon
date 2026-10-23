# PyTorch 静态广播阶段结果

当前 `latest` 为 `poseidon-trusted-pytorch-v28`，接入 `Tensor.expand_as` 和 `Tensor.broadcast_to`。旧默认 v18 与显式版本保留。

| 验收 | 结果 |
|---|---|
| 来源与独立参考 | 6 个来源上下文，全部通过；对应 3 个数学请求 |
| 真实 Agent 新生成 | 3/3；2 项首次通过，1 项自动修复后通过 |
| 原样真实 Agent DSL 重放 | 3/3；新增 API 调用为 0 |
| Poseidon CPU 数值比较 | 48 组输入、272 个输出值，最大绝对误差 3.506945543685358e-07 |
| 程序包 | 6 份完成重载、新输入、重复执行及同来源缓存复用 |
| 入口与兼容检查 | 120 项相关回归、18 项 CLI 检查、6 份来源清单一致；失败和跳过均为 0 |

本批实际发生 4 次生成、4 次 HTTP、6 次真实编译和 18 次密态调用。入口更新与报告审计没有新增上述调用。样本数包含重复执行与等价来源，模型数按 3 个数学请求统计。

覆盖的三个上下文为广播加法、转置视图与别名写入后归约、标量均值广播接 Linear。首轮别名案例因 Expr 数组构造类型不合法被静态检查拒绝；同一 Agent 在预算内修复，原失败响应保留。

`expand_as` 使用目标张量的静态形状；`broadcast_to` 使用公开静态维度。输入值不参与形状选择。dtype、广播规则、别名、容量及输出结构继续检查。数值门限保持 `1e-5 + 1e-4*abs(reference)`。

实现、测试与文档仅修改 ARM Ubuntu 的 Agent checkout。Poseidon、CKKS Runtime、Dacapo、参数与网络未修改，也没有安装、下载、提交或推送。CPU bootstrap、CUDA 和通用化其余批次仍待完成。

结果目录现占 34,354,462,720 字节，32 GiB 上限剩余 5,275,648 字节。下一批后端测试需要先调整产物预算；现有原始证据保留。

当前执行身份：`5a88df990a07ead9e19944721146d997b04e23408b3e0266cff497ced7658343`。

- [真实生成审计](agent-static-broadcast-live-acceptance-r1429.json)
- [原样重放审计](agent-static-broadcast-replay-acceptance-r1431.json)
- [最新入口验收](agent-latest-broadcast-selection-acceptance-r1432.json)
- [阶段机器可读报告](agent-static-broadcast-phase-summary-r1433.json)
