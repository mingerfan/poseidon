# PyTorch 公开条件选择与输出绑定验收

本批完成固定公开 mask 的 PyTorch 接入，以及 Runtime 最终输出绑定检查。真实 Agent 原有两个失败案例已通过新一轮测试。

| 验收范围 | 通过／计划 | 编译 | 密态调用 | 输入组／输出值 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| 原规则候选（r1471） | 21/21 | 21 | 27 | 96/624 | 8.56e-9 |
| 原真实 Agent（r1471） | 16/18 | 23 | 22 | 76/464 | 1.38e-8 |
| 输出绑定修复后的规则回归（r1475） | 6/6 | 6 | 12 | 36/352 | 1.01e-8 |
| 输出绑定修复后的真实 Agent（r1475） | 3/3 | 3 | 9 | 24/288 | 1.19e-8 |

当前 3 项均首次生成成功，修复轮数为 0；实际 3 次生成、3 次 HTTP，返回 usage 共 54,899 tokens。调用均已结束。所有程序包完成重载和缓存校验，三份代表包完成新输入及重复执行。

原 18 个真实 Agent 上下文已有通过证据：16 项沿用原版本，两个原失败项在新版本通过；重测中另有一项与原通过集重叠。它们不是同一版本下的 18/18，也不增加冻结 Benchmark 的模型数量。

## 接入范围

显式来源版本 `poseidon-trusted-pytorch-v33` 接收 `torch.where`、`Tensor.where`、`masked_fill` 和 `masked_fill_`。条件为已绑定的公开 bool 张量，支持静态广播。全公开计算在接入时求值，加密分支使用已有算术分解。原地计算使用输入工作副本，保留快照和别名语义。

默认来源 v18、latest=v28；v33 显式选择。既有 JSON、旧来源版本与 x86 配置保留。

## 修复内容

固定 Runtime 要求最终输出使用不同 Value ID。之前 Agent 的产物检查缺少这一项，Python 别名返回通过检查后在 Runtime verifier 被拒绝。

现 `poseidon-artifact-gate-v2` 在执行前检查重复输出，返回 `compiled_duplicate_final_output`、输出位置和受控修复提示。新增 `explicit-v14` 生成指导说明输出绑定规则；旧 v13 请求逐字节保持。检查器不将相同输出位置误判为相同表达式。

原 `public_where_2` 在 Runtime 拒绝后触及 Provider 流事件上限；原 `public_fill_2` 在后续修复中混淆 Expr 和对象数组类型。原响应、失败及绑定全部保留，未人工替换候选。新反馈与指导解决了本批观察到的输出别名问题，其效果仍限于本批测试。

首次回归另有 6 项因新模块未列入沙箱只读清单而 tracing 失败，编译、密态、API 均为 0。补齐单个模块的挂载后，在新绑定下重跑并通过，原失败记录保留。

## 检查与绑定

v33 来源控制：289 项通过。输出绑定相关控制与旧接口回归：162 项通过；执行与预算控制：12 项通过。这些测试存在重叠，分别记录。失败／跳过均为 0。当前独立明文参考与数值审计均通过。

SDK：`94d5490af7f423abee49a04994734d2d8776d357a3f830ed2958c3df098a9fbb`  
后端：Poseidon CPU，aarch64-linux，P4/w45。  
门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。  
验收绑定：`c0b7d75bbdc41fb60f0cd04c9cbdd43bb0fb74570502f5a90195bb1cd711b5e0`。

[机器可读报告](agent-public-mask-output-binding-acceptance-r1476.json) · [PyTorch 用法](operator-decomposition-and-python.md#pytorch-v33-公开条件选择) · [原批次记录](agent-public-mask-acceptance-r1472.json)

## 后续工作

继续混合公开／加密张量 concat/stack、常用来源配置整合及较大布局。任意 Python、动态 shape、加密条件选择、最大容量、应用精度、CPU bootstrap 接线、CUDA 实机和完整 ResNet20 仍分别待完成。

本阶段修改限于 Agent。后端库、参数、模型、reference、门限和网络保持。无安装、下载、提交或推送。全局产物上限 48 GiB、每例／批 4 GiB、磁盘余量 4 GiB。
