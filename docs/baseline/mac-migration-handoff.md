# 开发接续

## 工作位置

唯一开发 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。
Mac 副本只作历史保留，不与 Ubuntu 自动同步。

```sh
ssh -F '/Users/lhohy/Virtual Machines/Poseidon-arm64-access/ssh_config' poseidon-local-arm64
```

Ubuntu 使用 `POSEIDON_PLATFORM=aarch64-linux`，基础 work root 为 `/home/lhohy/poseidon-work`。
依赖版本和安装入口见[依赖说明](agent-setup.md)。

## 当前状态

- 应用组件已提供任务、预算、取消、生成修复、分级验收与程序包复用。
- r216：249 项测试通过、0 失败/跳过，2208 个历史请求精确重建；Linear/MLP 真实后端回归通过。
- 上述组件验收使用脚本化回放，没有新增真实 Agent 成功。
- 真实 Agent 剩余 26 项未通过，见[问题记录](stage2-remaining-closure-r211.md)。
- 第三阶段已实现算子分解和受限 Python；MLP 通过，RMSNorm/Attention 仍编译阻塞。
- x86 配置保留，近期没有新 x86 全量执行。
- 文档整理前检查未发现运行中的 Agent、Benchmark 或构建批次；开始新工作仍须检查实际进程。

## 下一步

1. 继续提取验证会话和 RunContext，减少旧 CLI 与全局环境耦合。
2. 推进 RMSNorm/Attention 与更常用模型输入；保持近似误差单独报告。
3. 原 26 项与真实 bootstrap 单列处理，不自动重试付费批次。

只读验收核查：

```sh
python3 -B scripts/baseline/benchmarks/tools/audit_application_r216.py --verify
```

r211 属于更早源码；其 source_hashes 与 r216 已不同，不能用当前源码通过旧 --verify。原报告保持原绑定。

已有修改保留。commit、push、PR、新依赖、下载和付费调用仍需分别授权。未改变编译器、SEAL、HEVM、安全参数、模型、reference 或误差门限。

## 文档

日常使用从 [Agent README](../../scripts/README.md) 开始；[文档目录](../README.md)区分当前指南与历史验收记录。旧迁移日志、过时 Provider 说明和重复阶段文字已清理，原始 JSON、模型及执行产物保留。
