# 应用数据集验收：总期限与取消 r1338

本轮增加跨命令总期限和 CLI 运行中取消。旧入口的默认行为保持；已验收程序包可继续使用。

## 接口

`ApplicationAssessment.submit(..., total_wall_seconds=240)` 选择总期限。CLI 在 `submit` 增加 `--total-wall-seconds 240`，范围 1–14,400 秒。后续 `advance` 沿用提交时的期限和调用额度。

v2 验收计划固定创建时间、期限和 Linux 启动身份，使用 `CLOCK_BOOTTIME` 与墙上时钟共同检查。机器重启、时钟回退或不可用返回服务异常。旧 v1 默认计划、模型 schema、生成请求哈希、程序包及 SDK 身份保持。

CLI 监控所属进程树，在期限到达或取消时停止该命令的 worker。未确认的执行保留已用额度和不确定状态。已经终态的结果可重载，重复 `result/cancel/advance` 保持原结果。直接同步 Python API 在执行边界检查期限与取消；本轮的进程内执行中断由 CLI 提供。

## 静态与控制验收

| 范围 | Passed | Failed / Errors / Skipped | 绑定 |
|---|---:|---:|---|
| 首轮源码：兼容与新增控制 | 102 | 0 / 0 / 0 | `88cad8e9…` |
| 最终监控修正后：受影响范围 | 29 | 0 / 0 / 0 | `e55c1280…` |

首轮测试触及 120 秒窗口，保留日志中已完成的 50 项，仅分片续验剩余 52 项。随后修复 worker 已提交终态、进程尚未退出时停止监控的问题。最终 29 项包括原 27 项和两项监控回归；两次范围合计 104 个不同测试，不能相加成 131 个不同测试，也不宣称 104 项都在最终绑定下重跑。

其中两项测试使用真实 Linux supervisor、worker 和脱离会话的子进程，验证总期限、运行中取消及所属进程清理。控制测试的数值后端为注入测试接口，单独计分。

## Poseidon CPU 验收

复用已验收的 `group_mean_center` 规则程序包，以有符号和固定随机两组新输入运行。真实 CLI 完成 `submit → advance → status → advance → result → cancel → advance`，七条命令均确认所属进程清理。

| 项目 | 结果 |
|---|---|
| 新编译 / API / HTTP | 0 / 0 / 0 |
| 真实密态执行 | 2 次 |
| 输入组 / 比较输出值 | 2 / 8 |
| 最大绝对误差 | 3.9523462835866496e-8 |
| 数值门限 | `1e-5 + 1e-4*abs(reference)` |
| 总期限 | 两次 advance 共用原 240 秒期限 |
| 终态重载、重复取消和 advance | 结果绑定保持，新增执行 0 |
| 缓存收据复用 | 新验收终态 succeeded，额度 used=0 |
| 候选来源 | 确定性规则基线，本轮新 Agent 生成 0 |

两次执行收据、七条 CLI 响应和缓存终态均在原 300 秒窗口内落盘。汇总脚本在最终报告前到时退出，原 `termination.json` 保留；之后只读核验收据、源码和程序包哈希，形成独立审计报告，没有重新运行编译或密态操作。脚本退出与功能验收分别记录。

SDK 身份保持 `6312151da9f93d55509afa80e21ff903f41dbb7191bc3557413367d0b8784f55`。验收控制器和 CLI 使用各自的新源码身份；旧控制器记录不重绑为新记录。654 项后端与参考文件、历史付费请求和 x86 配置保持。安装、下载、参数调整、提交及推送均为 0。

## 剩余工作

explicit-v13 的新生成效果仍待独立付费验收。旧 cross HTTP 的结果和费用保持未知。通用输入、算子组合、容量、应用精度、密文分区、bootstrap、CUDA 和整网目标继续推进。本轮只完成数据集验收控制及上述代表程序包回归。

[机器报告](agent-assessment-total-budget-r1338.json)

源码与冻结证据：`scripts/baseline/benchmarks/tools/assessment_deadline_v1/`。
密态与命令证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-assessment-total-deadline-acceptance-v1/`。

下一条命令：`python3 -B scripts/agent_assessment.py --help`。
