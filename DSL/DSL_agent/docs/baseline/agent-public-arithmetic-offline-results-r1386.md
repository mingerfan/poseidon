# 公开常量算术：Poseidon CPU 验收

2026-10-11，ARM Ubuntu，`lhy-agent-dsl`，开发目录 `/home/lhohy/Code/Poseidon`。

## 结果

**18/18 通过，0 failed／skipped／未运行。** 15 个 v25 公开算术上下文和 3 个旧版本兼容例，完成 **18 次编译、24 次密态调用、84 组次、504 个输出值**。其中 3 组次为重复输入，不另计为独立样本。

最大绝对误差 **4.174537703e-07**，整体 MAE **6.498315757e-09**。逐项门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。每项的 MAE、非零参考相对误差、cosine similarity 和逐项比较保存在原产物及机器报告。

| 上下文 | 结果 | 输入组次 | 输出值 | 最大绝对误差 |
|---|---|---:|---:|---:|
| add_c0 | 通过 | 8 | 48 | 5.2286704e-09 |
| add_c1 | 通过 | 4 | 24 | 1.0482090e-09 |
| add_c2 | 通过 | 4 | 24 | 4.9538153e-09 |
| subtract_c0 | 通过 | 4 | 24 | 9.9473615e-10 |
| subtract_c1 | 通过 | 4 | 24 | 3.8589849e-09 |
| subtract_c2 | 通过 | 4 | 24 | 2.7653606e-09 |
| multiply_c0 | 通过 | 4 | 24 | 2.9451195e-09 |
| multiply_c1 | 通过 | 4 | 24 | 2.8112214e-09 |
| multiply_c2 | 通过 | 8 | 16 | 1.7546502e-07 |
| divide_c0 | 通过 | 4 | 24 | 7.4120919e-09 |
| divide_c1 | 通过 | 4 | 24 | 7.7775893e-09 |
| divide_c2 | 通过 | 4 | 24 | 2.2874062e-09 |
| negate_c0 | 通过 | 4 | 24 | 3.4518104e-09 |
| negate_c1 | 通过 | 8 | 48 | 4.5490194e-09 |
| negate_c2 | 通过 | 4 | 8 | 4.1745377e-07 |
| f64_to_f32_c0 | 通过 | 4 | 24 | 5.3319710e-09 |
| f64_roundtrip_c2 | 通过 | 4 | 48 | 1.1147500e-08 |
| compat_v23_named | 通过 | 4 | 48 | 1.7457424e-09 |

每例完成真实 Hecate tracing、Earth／CKKS 编译、RuntimePlan／OperatorSpec／plaintext bundle 检查、Poseidon CPU 密态执行、最终解密及独立 reference 比较。18 份程序包均通过导出、重载和来源绑定检查；add_c0、multiply_c2、negate_c1 各追加三组新输入、一组重复输入及缓存复用，后续执行无需生成或重新编译。旧命名输出例保留字典、列表及 `None` 结构。

程序来自冻结的**规则候选**。真实 Agent 新生成、Provider 和 HTTP 调用均为 **0**。本结果验证输入接入和后端执行流程；v25 新入口的真实 Agent 生成质量仍待单独评测。

## 支持范围

显式选择 `poseidon-trusted-pytorch-v25` 后，已绑定公开参数和 buffer 可在 forward 中进行加、减、乘、除、取负，支持运算符、对应 Torch 函数和 Tensor 方法，并可与已有类型转换、广播、转置、reshape、静态切片及 Linear 组合。整数／布尔和显式 float32 中间值先转换到 float64；来源捕获、数值范围和容量规则沿用 [使用说明](operator-decomposition-and-python.md#pytorch-v25-公开常量算术)。

公开算术在可信导入层计算，涉及加密输入的计算保留在图中。默认 v18、latest=v21 保持；旧三例的请求内容与 request_id 保持。较大图、其他算子和任意 Python 写法按各自能力与资源规则处理，本批范围为上表。

204 个不同单元／组件检查通过，0 failed／error／skipped；15 个来源上下文通过真实隔离捕获及独立明文参考，6 个正式应用接入检查通过。[接入记录](agent-public-arithmetic-component-checkpoint-r1384.md)。

## 执行与复核

原 [r1385 计划](agent-public-arithmetic-offline-proposal-r1385.md) 和用户授权保留。首例编译和数值通过后，开发收集脚本错误读取了程序包顶层 model_source 字段；实际字段位于 origin.model_source。修正开发驱动后，从首例已验收程序包继续，未重复编译该例，总次数仍为 18／24。原失败报告、日志、候选和源码绑定保留，[续验绑定](agent-public-arithmetic-offline-continuation-r1385.json) 单列。

只读复核重新校验 1,173 个来源、证据及程序包文件，复算独立参考和追加输入误差，并使用现有无损请求存储接口核对原始请求与程序包的压缩表示。首版复核将这两种表示直接比较而停止，原日志保留；最终复核通过，未增加编译、密态或 API 调用。r1383 历史报告保持原绑定。

从首次启动到批次结束 **916.93 秒**，在原 45 分钟边界内。新增产物保守计量 **137.16 MiB**；结果总量 **28.88 GiB**；可用磁盘 **40.80 GiB**，保持原空间范围。

本批开发与执行均在 ARM Ubuntu。网络、依赖、Poseidon／Runtime／Dacapo、参数、模型、reference 和数值门限保持；安装、下载、commit／push 为 0。x86 配置保留，本批未在 x86 执行。

SDK：`8e88db16925fad029bdeaea53ebb2ed927ebdb9aece66dd7a252923ccb711809`。批次结果绑定：`ad9777d2c3f494ca648252766f65beb7ec2ebfdfdc2804a8b972a309753b79cd`。只读复核报告绑定：`08006e3819098f8d377477d6caf860fba1dcc6bec7bd14bf9ed2a476467fbb4d`。历史报告及旧程序包未重新签名。

## 后续

本批离线验收完成。下一项为按 v25 新来源上下文冻结真实 Agent 请求与预算，再申请付费范围。通用算子、布局容量、bootstrap、整模型和 CUDA 原计划继续，各自验收状态保持。

[机器报告](agent-public-arithmetic-offline-results-r1386.json) · [使用说明](operator-decomposition-and-python.md#pytorch-v25-公开常量算术) · [接续](mac-migration-handoff.md)
