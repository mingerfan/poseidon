# MHA v16 Agent 测试与程序包交付

日期：2026-10-09。开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

三项真实 Agent 测试已结束：布尔 mask／padding／残差模型完成密态数值验证；self 和投影后的 cross 模型尚未通过。程序包交付时发现的 Agent 元数据问题已经修复，同一份真实 Agent 源码已在最终源码绑定下重新编译、执行和交付。

## 三项测试结果

| 模型 | 生成／HTTP | 原批次结果 | 当前处理 |
|---|---:|---|---|
| self Attention → Linear → 残差 | 4／4 | 前两轮展开工作量 2455、1913 超过 1024；后两轮 hc.func 签名错误 | 增加计数和签名反馈；新生成尚未重测 |
| Linear → cross Attention | 2／2 | 首轮展开工作量 1721 超过 1024；第二轮 transport_invalid_stream | 保留失败调用；新生成尚未重测 |
| 布尔 mask＋padding＋残差 | 2／2 | 首轮编译 scale 750 超过 720；第二轮密态数值通过，交付时来源校验失败 | 修复 task.json 的声明顺序保存；原 Agent 源码离线重放、交付和复用通过 |

原批次记录保持原样：数值通过 1/3，完整交付 0/3，未运行 0，8 次生成、8 次 HTTP、0 次传输重试。该批次已经关闭。已收到 7 次调用的 usage，共 831,989 个 total tokens；流中断调用没有 usage，实际货币费用未知。

## 本轮修改

Application 原本按字典键排序写入 task.json，改变公开权重声明与来源记录的顺序，导致程序包严格校验失败。现在保留这段任务传输数据的声明顺序；请求哈希、状态、预算和账本继续使用原有规范化方式。

静态检查增加 `expanded_work_budget` 和 `native_signature_mismatch` 两个错误码。反馈提供源位置、实际计数、上限或参数数量，并说明 helper 调用和公开循环如何计费、hc.func 如何填写逗号分隔的 c/p 签名。反馈已进入正常自动修复循环，1024 上限保持。

新增反馈通过可信检查点给出诊断，不凭错误文本推断。原 7 份响应只读复核仍为：3 个工作量拒绝、2 个签名拒绝、2 个仅静态接受。静态接受不计为编译或数值通过。

交付修复测试 92/92；最终反馈、应用接口、CLI、请求兼容、沙箱和交付回归 164/164；最终 failed/errors/skipped 均为 0。两组测试有重叠，数量不相加。首次测试夹具及兼容测试的失败记录保留在开发工具目录。

## 最终源码绑定下的密态验收

复用本批布尔 mask 模型第二轮的 Agent 原源码，模型、公开权重、request、reference 和源码均未修改。正常 Application 入口完成：

- 真实 Hecate tracing、Dacapo-modified 编译、RuntimePlan／OperatorSpec／bundle 检查。
- Poseidon CPU 密态执行、最终解密、四组基础输入逐项比较。
- 程序包导出和重载，3 组新输入执行，再重复执行其中 1 组。
- 缓存复用；复用时生成与 HTTP 均为 0。

共 8 组执行、80 个输出值，含 7 组不同输入和 1 组重复执行。最大绝对误差 **6.11497519287e-09**，满足固定门限 `1e-5 + 1e-4*abs(reference)`。离线复核约 132.9 秒，新增付费调用 0。

当前程序包仍标记 `scripted_replay`、`agent_generated=false`，因为它通过离线重放交付。原始真实 Agent 响应、生成记录和旧生成身份单独保留并校验；这是“真实 Agent 原源码在当前后端通过”，不计为“修改后 Agent 新生成通过”。

曾尝试将旧生成身份与当前执行身份一起交给现有导出器，导出器拒绝 `Application identity changed before export`。失败及部分目录保留，身份门禁未放宽。

## 已确认、推断与未验证

**已确认：** 一个 MHA Agent 原程序在最终绑定下完成编译、密态数值验收、交付和复用。164 项相关测试通过，654 个受保护后端／reference／配置文件哈希一致。当前无运行批次。没有安装、下载、SDK 重建、后端库或参数修改，没有 commit、push 或 PR。

**基于证据的推断：** 原静态反馈缺少工作量和签名细节，影响了修复信息质量。新增反馈消除了这一信息缺口；是否改善真实生成成功率，需要新的有界测试。

**尚未验证：** self/cross 两项的新生成效果；旧生成身份与当前执行身份的直接交付策略；近似 MHA 对原模型的应用精度。此报告不扩展到其他 MHA 参数或全部 Attention 模型。

## 文件与下一步

机器报告：`docs/baseline/agent-mha-live-results-r1315.json`。

原始批次：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-source-live-v2`。

最终重放：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2`。

当前程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks/5384076bf9c04dc097254d1b1db30886/program`

下一条离线复用命令：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks/5384076bf9c04dc097254d1b1db30886/program --execute
```

下一项真实 Agent 测试只包含尚未通过的 self/cross 两个原模型，用于验证新增反馈和生成修复。另行批准后启动；布尔 mask 成功源码不重复付费生成。

阶段绑定：`cedbc99eccdd86cff2b4a9f68f7f904fc67986c0b37423880261445392fb2820`。最终重放绑定：`eb79960aac43650a3e12cb4e234b6a9b65c9bde4e9499c7c99bd2e3370327d63`。
