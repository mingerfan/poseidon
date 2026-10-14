# 应用接入诊断与产物预算

2026-10-11，ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

## 本轮结果

199 个不同组件测试通过，0 failed、0 skipped。其中 36 项是新增控制和适配器测试，其余覆盖旧接口、来源接入、阶段身份、缓存复用和反馈规则。测试使用注入适配器或受控停止；固定 ARM Nix 环境中的真实适配器调用边界另行检查。

本轮没有编译模型、进行密态运算或新的数值验证。API、HTTP、安装、下载、网络变更、后端库及参数修改、commit/push 均为 0。

初次检查保留两处启动问题：系统 Python 使用了旧默认工作目录，导致一个固定环境测试类无法初始化；Nix 测试命令遗漏模块搜索路径，四个模块未加载。修正仅作用于测试命令，随后相关 47 项全部通过。原日志保留在 `scripts/baseline/benchmarks/tools/application_admission_budget_v1/`。

## 已接入接口

- `submit`、`submit_import`、`submit_source` 增加可选 `admission_policy="structured-v1"`。已证明的后端、验收等级、预算、输入数量/rank/元素、节点容量和来源信任错误返回固定原因、位置及公开计数；未细分的 validator 拒绝保留通用代码。
- `operational_limits` 采用 `poseidon-operational-limits-v1`，包含产物字节上限和可用磁盘余量。
- 控制层记录任务目录、来源快照、每轮验证的新证据目录及程序包的保留字节。接入和修复阶段使用同一份持久化账本，继续执行时不重置。
- 新证据目录在准备 reference、复制编译产物或启动执行前登记。检查器拒绝符号链接、未知目录、非空历史目录、身份变更及丢失/被修改的账本。
- `execute(task, samples, operational_limits=...)` 对该次程序包调用建立独立资源窗口。旧调用方式保留，已有程序包绑定不因执行预算而修改。
- 取消、超时、产物超限和磁盘故障返回终态，保留现场，不进入候选源码修复。

原五项生成预算、模型/请求哈希、Provider 上下文、数值门限和验收等级保持。未选新接口的任务记录不添加新字段。预算及接入策略不进入模型语义复用键。

## 计量边界

`enforcement="monitored-v1"`，`hard_filesystem_quota=false`。计量为检查点观察到的保留文件逻辑字节峰值，并覆盖已登记的全部修复目录；不是累计写入流量或文件系统硬配额。检查间隔内的瞬时写入、账本/终态的最后写入可能超过上限，超限后停止并保留证据。现有 native 内存、CPU、编译并发及安全检查继续执行。

## 验收与后续

r1376 的 36 项 Poseidon CPU 数值和程序包验收已结算，原模型、候选、结果和绑定保留。本轮生产 Python 控制层及适配器变化产生新身份；原通过记录仍属于旧版本，未重新标记。

下一批仅选三个已冻结规则候选，检查新应用控制接口下的真实 tracing、编译、Poseidon 数值、程序包重载、新输入及重复执行。真实 Agent 新生成为单独计分和批准范围。原通用算子、布局容量、bootstrap、整模型及 CUDA 目标继续保留。

[机器记录](agent-application-admission-budget-checkpoint-r1381.json) · [接口](application-agent-component-v1.md#接入诊断与产物预算)

[下一批冻结方案](agent-application-control-offline-proposal-r1382.md)（尚未批准/启动）。
