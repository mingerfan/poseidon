# PyTorch InstanceNorm 接入与离线验收 r847

ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。本批完成 InstanceNorm 的模块、函数与非 batch 输入接入，复用原有算子分解和 Poseidon CPU 后端。

## 接入范围

| 入口 | 输入 rank | 数学路径 |
|---|---|---|
| `nn.InstanceNorm1d` | 2／3 | 当前输入统计量，或公开冻结统计量 |
| `nn.InstanceNorm2d` | 3／4 | 当前输入统计量，或公开冻结统计量 |
| `F.instance_norm` | 3／4 | 显式 `use_input_stats` |
| 非 batch 模块输入 | `[C,L]`／`[C,H,W]` | 增加 batch 轴，归一化后还原形状 |

当前统计量使用现有 `instance_norm` 分解，要求显式 rsqrt 近似。冻结均值、方差使用现有精确 BatchNorm 推理路径。gain／bias 与统计量由公开常量绑定；模块通道数匹配输入签名，epsilon 为 `0 < eps <= 1`。状态更新被输入检查拒绝。

30 个来源上下文按十个分区各安排三例，组合直接 Tensor、tuple／dict 输出、均值及残差；输入规模与乘法深度沿用现有资源配置。原 1,200 个冻结 Benchmark 保持。

## 验收结果

| 检查 | 结果 |
|---|---|
| 静态／组件／旧入口 | 146 passed，0 failed／errors／skipped |
| 原矩阵捕获 | 36 份逐字一致 |
| 原矩阵请求与规则源码 | 30 份保持一致 |
| 独立数学、Torch 与高层参考 | 480 组通过 |
| 新规则候选真实 tracing、编译、Poseidon CPU 与数值 | 30／30 通过 |
| 程序包重载、新输入执行两次、缓存复用 | 3／3 通过 |
| 密态输入与输出总计 | 150 组，1,598 个值 |
| 密态最大绝对误差 | 1.0615338747e-9 |
| 来源身份拒绝 | 60 项通过 |
| 实际产物／包文件哈希 | 723／783 项核验 |
| 新真实 Agent 生成、API／HTTP | 0 |

执行链：Hecate → Earth／CKKS → RuntimePlan／OperatorSpec／plaintext bundle → CKKS Runtime → Poseidon CPU → 最终解密。执行器获得评估密钥；沙箱、网络隔离和原资源限制核验通过。

数值门限为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

## 近似与适用范围

密态误差比较对象是声明的近似计算。输入统计量示例采用 epsilon=0.5、一次 Newton、区间 [0.5,1]，相对原函数误差最高 0.0253193516；原模型应用精度仍待认证。冻结统计量路径相对原函数差异最高 5.55e-17。独立原模块参考与分解误差最高 5.55e-17。

三个新包同时通过现有编译资源分析，保留数据下界约 62.88–544.375 MiB；密钥、分配容量、工作区和峰值仍为未知。这一分析沿用原执行限制。

新增生产改动仅在 `pytorch_model_capture.py`。相关测试、生成器、30 份来源包和完整的 82 个 NPY 状态数组随源码保存。648 个后端文件、6 个 reference 文件及 6 份历史记录保持。Dacapo、Runtime、Poseidon、参数、误差门限与 SDK 均保持。

## 保留的失败记录

- r836：测试驱动错误地要求固定 Torch 发出 warning；修正该断言后，严格拒绝检查保留。
- r838：初始来源包未带完整 state_dict；现有加载器正确拒绝。通过可信导出补齐原固定状态，保持数学模型，重新捕获和执行。
- r843：验收驱动在纯 Nix 内重复调用宿主 CLI；正常客体入口 r844 通过。
- r845：提案准备驱动缺少新目录的父目录；r846 完整准备通过。
- r849：Git 状态未展开未跟踪目录，与起始全量路径快照不一致；r850 使用相同列举选项，确认原 1,487 条路径完整保留。

上述记录和原绑定保留。实际编译、执行及数值批次的失败为 0；这些规则结果与真实 Agent 生成结果分开计分。

## 后续

[当前真实 Agent 提案 r848](agent-pytorch-general-input-live-proposal-r848.md) 包含原 30 个矩阵上下文和 30 个 InstanceNorm 上下文。旧 r816 完整保留；新提案重新绑定当前源码与来源，等待单独批准。

[使用方式](operator-decomposition-and-python.md#pytorch-instancenorm) · [机器报告](agent-pytorch-instance-acceptance-r847.json)。原始逐项数组、IR、日志、响应、产物和哈希保存在 ARM work root。bootstrap、密文分区、CUDA、完整 ResNet20 与更多未见组合继续单列。

最终检查：Git diff check 通过，当前 1,637 条修改／未跟踪路径，worker 均终态。本批记录用时约 40 分钟；新增产物实际占用约 152 MiB，全局约 24.42 GiB，磁盘余量约 45.63 GiB，均在原边界内。Git 提交、推送和 PR 为 0。
