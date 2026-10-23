# Agent 广播与 packing 坐标更新

本轮补齐加、减、乘的逻辑广播说明，并统一直接 gather、BSGS 与 Linear tile 的坐标约定。当前使用 explicit-v19；旧请求和旧指导版本保持兼容。

## 已完成

- 129 项组件和兼容测试通过，失败和跳过均为 0。
- 原 wide_tail_linear 规则程序在新执行身份下通过真实 frontend、编译、Poseidon CPU 数值和程序包复用验收：1 次编译、3 次密态调用，8 组输入、112 个输出值，最大绝对误差 3.998e-7。
- 同一原始 PyTorch 源码及权重完成 v36 捕获、完整来源重载和两套明文参考验证：16 组输入、224 个输出值。
- 模型仍为 2,052 个输入元素、1,025→3 Linear 和按行均值，三个命名输出包含一个共享结果。模型、公开权重、布局、参数、reference 与误差门限保持原样。

规则程序通过和明文来源验证分别记录；它们不计为本轮真实 Agent 生成成功。

## 失败定位与处理

上轮真实 Agent r1566 在四次生成后失败：两次静态拒绝、两次完成编译和密态执行但数值不符。原响应、候选及失败证据保留。

公开源码检查发现，一份候选先使用目标坐标下的 gather 掩码，再旋转整个乘积，改变了输出位置。其他候选还出现广播混合不同行数据、覆盖预绑定常量名称的问题。

explicit-v19 根据实际绑定角色区分三种 mask frame，并为 add、subtract、multiply 统一提供逻辑广播规则。生成和受控反馈使用相同说明；检查器与后端约束保持原行为。

## 正在执行

r1570 使用原始可信本地 PyTorch 源码进行一例真实 Agent 重测，SDK 为 eef0f250a457a24aa658c7921e6bc62e0387a9c7c8ea29f9f7cf47353354ad41。

最多 4 次生成、16 次 HTTP、4 次编译、6 次密态调用；API 和密态串行，4 小时。新增产物上限 4 GiB，全局 48 GiB，磁盘保留 4 GiB。批次结束后自动审计来源、调用账本、实际产物、数值与程序包复用。

主进程 PID948106/start26776728；后置审计 PID948632/start26786806。进入下一次共享源码修改前，核对两者终态和 HTTP 结算。

## 记录

- [组件检查](../../scripts/baseline/benchmarks/tools/binding_frames_component_v1/controls.json)
- [真实数值审计](../../scripts/baseline/benchmarks/tools/binding_frames_acceptance_audit_v1/report.json)
- [当前付费范围和绑定](agent-binding-frames-source-live-proposal-r1570.json)
- [机器可读进展](agent-binding-frames-progress-r1572.json)

下一条查询命令：

```sh
python3 -B -m json.tool /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-binding-frames-source-live-r1570/report.json
```

通用算子与未见组合、容量和布局、原模型应用精度、bootstrap、CUDA 实机及 ResNet20 仍按原计划继续，以上小批结果仅覆盖所列范围。
