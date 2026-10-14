# Agent 通用化阶段报告

本轮完成了能力与修复控制层的工程基础，增加参数化 PyTorch 导入、通用张量 rank 配置、权重描述符和程序包本地执行。实际开发和验收均在 ARM Ubuntu checkout，后端为 Poseidon CPU。

## 改动

- 能力清单统一输入、分解、布局和后端状态。资源值区分精确值、保守下界及未知项。
- 应用与 CLI 可使用同一套结构化诊断。修复循环按错误位置和约束判断重复失败，重复源码直接结束。
- 程序包分别绑定生成、编译、执行和验证身份。生成来源保存实际返回源码、HTTP 账本和 token 用量。
- PyTorch v2 支持工厂参数、float64 或显式 float32→float64、静态形状写法，以及 Tensor、tuple/list、字符串键字典输出。
- general-tensors-v1 开放 rank 1–32，沿用现有最高元素和物理资源上限。transpose、permute 各有三个真实上下文。
- 声明式图可从无 pickle 的 NumPy 权重包导入。descriptors-v1 将提示中的数值换为形状与内容哈希，可信编译端仍绑定完整固定值。
- 应用提供 submit_import 和 execute。通过验收的程序包可重载，处理 1–16 组新输入，并还原静态输出结构。

## 验收结果

| 范围 | 结果 | 来源 |
|---|---:|---|
| 组件、请求、Provider、程序包、平台与兼容测试 | 369 passed / 0 failed / 3 skipped | r616 |
| 新评测入口的预算与无收费控制测试 | 5 passed / 0 failed / 0 skipped | r620 |
| 旧 MLP、RMSNorm 与分块多输入输出 | 3/3 | r580：2 份历史真实 Agent 原源码，1 份规则程序 |
| rank 1–32 及跨块形状组合 | 6/6 + 4/4 | r590、r592：规则程序 |
| 参数化 PyTorch 跨类别模型 | 6/6 | r596：规则程序 |
| 非重复随机权重模型 | 8/8 | r614：规则程序 |
| 输出结构与原始源码字节绑定复核 | 2/2 + 1/1 | r601、r603 |
| 收尾受影响范围回归 | 3/3 | r617–r619：权重、PyTorch 结构输出、历史 MLP |

真实后端案例均完成 frontend tracing、Earth/CKKS 编译、RuntimePlan/OperatorSpec/常量 bundle 检查、Poseidon 密态执行、解密、四组输入逐项比较、程序包重载，以及三组新输入的独立双参考比较。程序包复用检查没有再次生成。各表行使用自己的冻结绑定；复核行不增加模型数量。

联合测试的三个 skip 分别需要 explicit45 native-loop、真实沙箱拒绝探针、fresh legacy40/explicit40 批次。它们没有计入通过数。静态和 mock 测试也没有计为密态测试。

所有密态比较保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。8 个随机权重模型的最大绝对误差为 3.1693303e-08，重载 2.5869513e-08，三组新输入 4.8563828e-08。

万参数请求采用 100×100 Linear、10,000 个不同权重值。完整模型 209519 字节，可信请求 538550 字节，外发生成上下文 124313 字节。该项只完成请求准备和完整性检查，编译、密态执行和真实 Agent 均未运行。

PyTorch 明文检查采用转换后的 CPU float64 eval 模块，并与两套图参考比较。原 float32 执行精度和原模型应用指标尚未认证。程序包 execute 本身只保证新输入执行；本轮开发验收额外进行了独立数值比较。

## 保留事项

Dacapo/Hecate、CKKS Runtime、Poseidon 及已存在补丁的 582 个文件与开工快照相同。独立数学/Torch reference、数值门限和平台配置也保持。未重建 SDK、安装、下载、调用付费 API、提交或推送。旧失败、历史报告和生成来源保留。

准备新评测时，旧 PyTorch 导入报告因工具哈希变化被拒绝。重新跑了六项隔离明文参考，模型哈希与原记录完全相同。新报告建立新绑定，旧报告没有改写。

下一批真实 Agent 提案为 6 个 PyTorch 模型和 8 个随机权重模型，已冻结模型、请求、代码、依赖与预算，尚未启动。它检验新入口的生成与修复质量，离线规则验收不能替代这项结果。

后续继续完成 PyTorch 外部权重直连、布局候选和资源预测、请求级上下文提取，以及更多算子、近似和未见组合。大图密文分区、bootstrap、CUDA 实机和完整 ResNet20 保持独立未完成项。

## 文件与命令

[机器报告](agent-generalization-acceptance-r622.json) · [真实 Agent 提案](agent-generalization-live-proposal-r620.md)

无收费重放与新输入执行仍使用：

```bash
python3 -B scripts/dsl_bundle.py replay --bundle /absolute/program --execute
python3 -B scripts/dsl_bundle.py execute --bundle /absolute/program --inputs inputs.json
```

下一批付费执行命令见提案；只有得到该范围的新授权才执行。
