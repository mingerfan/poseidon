# 通用输入真实 Agent 验收

14 个案例全部通过 Poseidon CPU 验收：12 个首次通过，2 个自动修复一次后通过。6 个参数化 PyTorch 模型和 8 个随机权重模型均完成生成、真实编译、密态计算、数值比较、程序包导出与重载。

## 已确认事实

| 案例 | 生成次数 | 首次结果 | 最终最大绝对误差 |
|---|---:|---|---:|
| PyTorch MLP | 1 | 通过 | 7.022512e-11 |
| PyTorch Conv→Pool | 2 | 修复后通过 | 3.596478e-09 |
| PyTorch 残差 | 1 | 通过 | 1.541462e-08 |
| PyTorch BatchNorm／双输出 | 1 | 通过 | 3.16724e-09 |
| PyTorch 矩阵融合／三输出 | 1 | 通过 | 6.287111e-07 |
| PyTorch 形状→Linear／双输出 | 2 | 修复后通过 | 2.451921e-08 |
| 随机权重 Linear | 1 | 通过 | 3.716803e-08 |
| 随机权重 MLP | 1 | 通过 | 3.368214e-10 |
| 随机权重残差 | 1 | 通过 | 2.139138e-08 |
| 随机权重通道 Linear | 1 | 通过 | 1.996378e-08 |
| 随机权重批量 Linear | 1 | 通过 | 7.960313e-09 |
| 随机权重双输入融合 | 1 | 通过 | 1.49013e-08 |
| 随机权重共享多输出 | 1 | 通过 | 2.204845e-08 |
| 随机权重 Conv→Pool | 1 | 通过 | 5.041888e-09 |

本批比较 56 组输入、268 个输出值，最大绝对误差 6.287111e-07；重载最大绝对误差 4.844279e-07。门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。重载使用原编译产物，生成和编译调用均为 0。

14 份程序包分别保留精确返回源码、原始响应、生成配置、HTTP 账本、用量和生成身份。验证器、编译器、执行依赖及验证范围另有绑定。内部重放验收不记为新生成；最终程序包的 Agent 来源由原始返回内容与实际编译源码逐字节核对。

真实 API 共 16 次生成、16 次 HTTP，传输重试 0；总 token 394355，其中输入 130664、输出 263691。所有响应都有用量，未决 HTTP 为 0。账本不代表服务商账单，货币费用未核对。整批耗时 42.10 分钟，未扩大批准范围。

[调用参数核验依据](https://api-docs.deepseek.com/api/create-chat-completion/)；实际调用范围以[冻结提案](agent-generalization-live-proposal-r620.md)及原始账本为准。

## 自动修复

Conv→Pool 首次候选把 `p0/p1` 当作临时变量，覆盖了同名公开常量。独立静态复核得到 `Native loops cannot overwrite reserved bindings`；下一轮更名后通过。属于候选命名冲突，未进入编译。

形状→Linear 首次完成密态执行，但逐项比较失败，最大绝对误差 0.0778477815。下一轮删除额外旋转分支及掩码组合后通过。两轮模型、权重、reference、参数和门限相同。基于源码差异与结果，原候选的 packed 表达式／布局组合错误是主要判断；这不是对编译器缺陷的证明。

首次失败、响应和原绑定全部保留。未手工替换候选。静态反馈仍可进一步增加受控错误位置与约束；本批的自动修复通过不代表所有失败都能自动处理。

## 程序包新输入执行

矩阵融合、共享多输出、Conv→Pool 各使用 3 组新的确定性输入，分别执行两次。通过应用的 `execute` 接口加密、执行、解密；开发验收随后用独立数学解释器和 CPU float64 PyTorch reference 比较。输出名称与 PyTorch 静态输出结构保持。

9 组不同输入、18 组执行、120 个输出值均通过，最大绝对误差 6.248978e-08。原程序包及源码保持，新增 API、生成和编译均为 0。execute 的返回本身只声明执行观察；本报告增加的数值结论限于这组已比较输入。

r624 验收脚本曾向二维比较器传入一维数组，停在测试调用。原密态执行及失败报告保留；r625 修正调用，复用并核验该执行记录，再完成其余执行。该脚本失败未记为 Agent 或后端失败，也未改写成旧记录通过。

## 范围与剩余工作

本批确认参数化 PyTorch、静态多输出和 descriptors-v1 新入口的自由生成能力。验收对象是 14 个小模型案例，输入最多 9 个元素；不是 14 个互不重叠的模型家族，也不代表任意 PyTorch 模型或任意规模。rank 1–32、万参数请求、历史 DSL/helper 等覆盖仍分别以其原记录为准。

原 float32 执行精度、外部 PyTorch 权重直接绑定、布局候选与资源预测、请求级上下文、更多算子和未见组合仍需继续。密文分区、真实 bootstrap 接线、CUDA 实机、x86 原生、空白安装和完整 ResNet20 保持独立未完成项。

582 个后端文件、3 个 reference／误差模块及 628 个付费冻结源码均与开工／授权快照一致。分支与 HEAD 保持；未安装、下载、重建 SDK、提交或推送。当前旧修改和历史证据均保留。

本批静态测试沿用同一源码下的 r616（369 passed、0 failed、3 skipped）及 r620 控制测试（5 passed）。三个 skip 需要独立 native-loop、沙箱拒绝和旧配置真实批次，未记为通过。

## 文件

[机器报告](agent-generalization-live-acceptance-r626.json) · [工程阶段报告](agent-generalization-acceptance-r622.md)

原始生成与运行：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generalization-live-r623`。新输入原始报告：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generalization-live-programs-r625/report.json`。源码绑定：`207f317b36f217482f095c1b97c26520bdf901f12d1b617f4975a98087dfa850`。

无收费重载示例：

```bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generalization-live-r623/tasks/f928d595ad0448d7b19a0f861a8a9a60/program --execute
```
