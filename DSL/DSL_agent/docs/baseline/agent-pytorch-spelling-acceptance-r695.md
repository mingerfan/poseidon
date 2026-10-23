# PyTorch 调用形式与离线验收 r695

## 结果

24 个小型 PyTorch 模型完成隔离捕获、独立参考、Hecate tracing、Dacapo 编译、Poseidon CPU 密态计算和逐项比较。原批次已通过的 18 项直接保留；本次只完成剩余 6 项。

| 检查 | passed | failed | skipped / 未运行 |
|---|---:|---:|---:|
| 相关单元、组件与旧接口测试 | 301 | 0 | 0 |
| 原模块与两套分解参考（每例 16 组） | 24 | 0 | 0 |
| 规则候选 Poseidon CPU 数值验收 | 24 | 0 | 0 |
| 程序包重载、新输入重复执行、缓存复用 | 3 | 0 | 0 |

本批使用规则候选，真实 Agent 新生成和付费调用均为 0。原 r620 的 14 项真实 Agent 评测已完成，不在本批重复。

96 组初次密态输入、12 组重载输入、18 组新输入重复执行，合计 126 组、628 个输出值。maximum absolute error 为 `1.3746586567458253e-7`，逐项门限仍为 `1e-5 + 1e-4 * abs(reference)`。三份程序包的新输入执行和缓存复用均未生成或重新编译。

## 接入范围

每组有三个不同的 shape 或参数上下文，全部使用静态输入和公开权重。

| 类别 | 本批写法 |
|---|---|
| Conv1d、Conv2d | 函数调用、静态关键字参数、分组与深度卷积 |
| AvgPool1d、AvgPool2d | 函数调用、默认或显式 stride、静态 padding |
| BatchNorm | `F.batch_norm` 推理调用，rank 2–4 |
| LayerNorm | 模块与函数调用，含 `elementwise_affine=False` |
| GroupNorm | 模块与函数调用，含 `affine=False` |
| 转置与公开矩阵 | `Tensor.T`、`x @ W.T`、三维转置与残差组合 |

公开二维矩阵可在矩阵乘法的左侧或右侧，密文操作数支持 rank 2–3；左侧与右侧的三个上下文已做独立参考检查。本批密态矩阵例覆盖右侧公开矩阵；左侧多上下文密态兼容仍须单独补验。两密文矩阵沿用既有 dot/mm/bmm/matmul 接口。

归一化例使用已声明的 Newton 倒平方根近似，有效区间为 [1,5]。原函数近似误差最高 `0.26429569797432895`；上述密态结果相对近似 reference 通过，应用精度尚未认证。

示例清单：[pytorch_spelling_v1](../../scripts/baseline/cases/pytorch_spelling_v1/)。
输入说明：[Python 与高层算子](operator-decomposition-and-python.md)。

## 续验与证据

原 `agent-pytorch-spelling-offline-r689` 达到一小时外层边界后以 124 退出，已通过 18 项。其报告保留原 checkpoint，另存终止观察，未改写为完成。

获准的 r690 续验只选择原任务中的 `group_0–2` 和 `transpose_0–2`，复用原 24 份已验证来源包；失败和已完成项均未重跑。续验执行约 28.47 分钟，在追加一小时与累计两小时范围内完成。

开发工具增加了有绑定的续跑检查：核对原计划、原报告、工具归档、来源与依赖身份、批准范围和真实终止观察；旧报告与记录保持原样。17 项续验边界测试和 6 项已有源码边界测试通过，与另外 278 项测试无重复。

初次静态测试有一项常量数量断言失败：Torch 对公开 buffer 的转置做了常量折叠，测试错误假定只有两份常量。原失败日志保留，测试改为验证 Parameter 转置的原权重绑定后通过。公开 buffer 转置可能产生重复常量的问题仍列为优化缺口。

当前绑定：`162c86502fd951d0961d56ed4153bd8e05198954d1c8724adc68cdca50ba8d35`。
原绑定：`8c7781a8ff0539c367226eca4bf66181d2129b23f8e1598f00b1c6d9c3423803`。

执行结果位于 ARM work root：

- `results/agent-pytorch-spelling-offline-r689`：原 18 项及全部来源包。
- `results/agent-pytorch-spelling-continuation-r692`：六项续验、原结果引用与完整最终报告。
- `results/agent-pytorch-spelling-audit-r695`：联合审计与 568 个程序包文件的哈希复核。

465 个受保护后端文件、5 个 reference／门限模块及 633 个冻结 Agent 文件核验一致。新增产物约 90.5 MiB，总 results 约 22.33 GiB，剩余磁盘约 46.45 GiB；预算未扩大。Poseidon、CKKS Runtime、Dacapo、参数与门限均保持。未安装、下载、重建 SDK、提交或推送。

[机器报告](agent-pytorch-spelling-acceptance-r695.json)

## 剩余工作

- 在新输入进入密态执行前，校验已声明的近似有效区间。
- 接入静态 input shape 循环和直接标量输出，保留输出结构。
- 增加逐节点／模块路径的近似选择及冲突检查。
- 补验公开矩阵左侧写法，减少公开 buffer 转置的重复常量。
- 改善源码身份检查的缓存开销，继续分离应用组件和开发工具。
- 本批新增输入形式的真实 Agent 评测仍待新绑定及单独授权。r675 已因源码变化失效，保留为未执行历史提案。
- Bootstrap、CUDA、x86 原生、空白安装和完整 ResNet20 保持独立的未完成事项。

下一条只读检查命令：

~~~bash
python3 -B scripts/prepare_model.py --help
~~~

开始下一批修改前，继续核实真实进程和冻结绑定。新生成评测使用新提案，不复用已结算或失效授权。
