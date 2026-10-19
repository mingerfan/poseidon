# 测试记录与本地产物

这里保存与具体源码版本绑定的验证证据，不参与默认构建。报告、复现脚本、输入指纹、最终测试日志以及失败对照应提交；不能只保留通过结果。历史结果不代表当前 HEAD 已重新测试。

| 目录 | 用途与结论 |
| --- | --- |
| [framework-server-validation-20261015](framework-server-validation-20261015/summary.json) | `6dc8cbfb` 在 188Server 四张 V100 上的 GPU API、Waterline、设备重排和 Prepare OOM 验证；五项检查通过，FP64 Tensor 路径跳过，不代表完整 Qwen 验收 |
| [luhang-merge-cpu-188-20261014](luhang-merge-cpu-188-20261014/report.zh-CN.txt) | CPU 合并验证与父提交对照：17 项检查中 14 项通过、3 项失败；含 BGV 最小复现与原始日志 |
| [luhang-merge-rebase-20261014](luhang-merge-rebase-20261014/report.zh-CN.txt) | 分支合并、rebase 与构建兼容验证；最终五项 CTest 中三项通过、两项失败，另有 CPU 复测记录 |
| [waterline-20261013](waterline-20261013/README.md) | Waterline 资源、流顺序、Rescale 和模型尝试的报告与反例；完整模型尚未完成数值验收 |
| [waterline-benchmark-20261018](waterline-benchmark-20261018/diagnosis.zh-CN.txt) | RTX4060 与 V100 单/双卡的初始全加载、水位、编码后端和轮询对照 |
| [waterline-fix-20261018](waterline-fix-20261018/diagnosis.zh-CN.txt) | 第一轮提交流水线修复的前后测量、数值检查和失败对照 |
| [waterline-legacy-20261018](waterline-legacy-20261018/diagnosis.zh-CN.txt) | 最早 Runtime 头文件与后续实现共用 GPU 后端的历史对照，不是整个历史仓库重建 |
| [waterline-cause-20261018](waterline-cause-20261018/diagnosis.zh-CN.txt) | 标量任务与调度路径隔离；CPU 调度时间不能直接用于分解真实 GPU 延迟 |
| [waterline-batch-20261018](waterline-batch-20261018/diagnosis.zh-CN.txt) | 最终 57 配置、206 次主测量及独立诊断；常用配置改善，V100 8 MiB 水位回退尚未解决，完整四卡 Qwen 未重跑 |

`build-*` 是仓库根目录下可再生的 CMake 构建树，已由根 `.gitignore` 忽略。删除后需要重新配置和编译；部分历史报告引用其中的二进制，清理前应确认是否仍需作对照。`.ruff_cache/` 可直接删除，会自动重建；`.codex/` 用于本地工具状态，保持忽略。`runtime-artifacts/` 含受跟踪的运行脚本、profile 和样例，保留。

Waterline 目录的 `.gitignore` 只忽略编译、配置、下载、传输及持续监控输出；其他测试日志和诊断结果继续提交。各次验证的源码校验文件和最终报告照常保留。全局不忽略 `*.log`、`*.json` 或整个 `test-results/`，避免隐藏后续验证证据。已有受跟踪文件不受新增忽略规则影响。

CPU 合并验证目录的 `logs.sha256` 校验入库的 41 份日志，不包括只在原机器有意义的三个本地 PID 文件。

Nsight 原始报告及 SQLite 数据库由根 `.gitignore` 忽略，已有归档与恢复说明见 [raw-profile-artifacts.json](raw-profile-artifacts.json)。生成的 Qwen MLIR、计划和权重包由各实验目录的忽略规则管理；保留对应报告和复现入口。

2026-10-18 的五组 Waterline 实验按目录忽略逐槽 `decoded.json`；最终批次还忽略生成的 `server-plans/`、传输源码包和本地工作状态。文件仍保留在本机，各目录的 `local-artifacts.json` 记录相对路径、字节数和 SHA-256；通过与失败报告、命令、输入指纹和日志都继续入库。恢复生成输入后再运行依赖这些输入的历史汇总脚本；最终批次的服务器计划副本保存在 `188Server:/home/xuming/poseidon-waterline-batch-20261018/server-plans/`。复现构建和测量还需要报告中指定的本地构建树、合成权重包及历史 profile。
