# 结构化 PyTorch 输入：Poseidon CPU 离线验收

方案绑定：`96e0c0f058429781a0d88fa0ed8333ca12e5924b0fc52356bd35ada8dfbc451f`。状态：待批准。

范围：defaults0、defaults1、defaults2、dictionary0、dictionary1、dictionary2、keywords0、keywords1、keywords2、nested0、nested1、nested2、f64_to_f32_c0、f64_roundtrip_c2、compat_v23_named。前 12 项为 v26，每类 3 个上下文；其余 3 项为旧版本兼容。所有模型、权重、reference 和规则候选已冻结。候选为规则基线，真实 Agent 和 API 调用为 0。

通过原应用组件检查真实 Hecate tracing、Dacapo 编译、RuntimePlan／OperatorSpec／bundle、Poseidon CPU 密态执行、解密和四组输入逐项误差。15 包重载；dictionary0、nested1、defaults2 另验 3 组结构化新输入、1 组重复输入、3 组命名输入的一致性，以及输出结构和缓存。

| 上限 | 数值 |
|---|---:|
| 编译／密态调用 | 15／24 |
| 工作窗口／启动器累计窗口 | 40／45 分钟 |
| 初次任务 | 180 秒 |
| 编译／链接／密态并发 | 2／1／1 |
| native 内存／CPU | 4 GiB／150 秒 |
| 新产物／全局产物／磁盘余量 | 512 MiB／32 GiB／4 GiB |

现有 P4／w45、门限 `1e-5 + 1e-4*abs(reference)` 保持。API、安装、下载、网络／后端／参数修改均为 0。完整性、环境或预算故障停止；候选失败保留且其他独立案例继续，退出后不自动重跑。

[准确案例、请求和哈希](agent-input-binding-offline-proposal-r1390.json) · [接入结果](agent-input-binding-component-checkpoint-r1389.md)

只读预检（未带 `--run-approved`）：

~~~sh
PYTHONPATH=scripts/baseline python3 -B scripts/baseline/benchmarks/tools/pytorch_input_binding_v1/offline_cpu.py
~~~
