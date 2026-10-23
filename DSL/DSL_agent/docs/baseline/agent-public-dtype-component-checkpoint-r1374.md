# 公开参数类型转换接入

ARM Ubuntu checkout 新增显式 PyTorch v24：公开参数和 buffer 支持 `.float()`、`.double()`、`.to(dtype=torch.float32/float64)`。原始数组、每步类型和舍入后的常量分别绑定；float32 结果按实际值进入现有 float64 数学图。默认 v18、latest=v21 保持。

| 检查 | 结果 |
|---|---|
| 控制与兼容测试 | 161 passed，failed/errors/skipped 0 |
| 隔离源码捕获 | 33 passed，failed/skipped/not_run 0 |
| 独立明文参考 | 528 组、4,224 值 |
| 标准库数学复核 | 132 组，最大绝对误差 0 |
| DSL 编译／密态／付费调用 | 本阶段均为 0 |

33 份来源配置包含五种原始公开 dtype、两种浮点目标、三个组合上下文及三个浮点往返上下文；这些是接入测试数量，冻结 Benchmark 的结构数量保持。原始整数与张量容量仍按既有输入契约检查。

## 改动与兼容

新增 public-value v2 策略及 v24 来源、provenance、reference 登记。v1 bool 转换记录与历史实现逐字节一致，旧版本保持原行为；重载通过原始状态与声明式记录核验派生常量。19 个新增测试覆盖舍入、溢出、绑定篡改、类型/版本拒绝及命名 None 输出。

11 个既有 Agent/测试文件及一个新测试已修改，文件哈希见 JSON。Poseidon、Runtime、Dacapo、构建配置、参数、reference 和门限保持；14 份历史记录哈希核对通过。

首次测试模块选择、探针绑定及输出目录错误的源码和日志保留。最后一项由现有隔离目录门禁拒绝，修正探针后通过。

## 验收状态

当前 SDK `8ca41ede84d20c654fdc7885658e343d1d263cfc0885fd1a96ad259c54e38f8a`，四阶段身份见 JSON。r1373 的 named 补验属于原 e2b3e63c 身份：1 编译、3 密态调用、8 组/96 值，最大绝对误差 8.1459362763e-9；它没有被改写为 v24 的结果。

v24 编译、密态数值及真实 Agent 生成待验收；原模型应用精度、容量、bootstrap、CUDA 和整网目标分别推进。

## 下一批

[集中离线方案](agent-public-dtype-offline-proposal-r1376.md)：33 个新来源配置和 3 个 v22/v23 兼容项，归一化后为 12 个请求；全计划 156 组/1,272 值。规则候选已冻结，尚未批准或执行。

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_dtype_offline_v1/run.py --plan
~~~

[机器可读检查](agent-public-dtype-component-checkpoint-r1374.json) · [v24 用法](operator-decomposition-and-python.md#pytorch-v24-公开参数类型转换)
