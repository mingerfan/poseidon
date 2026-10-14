# PyTorch 算子与形状组合接入检查

2026-10-06，ARM Ubuntu，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

## 结果

| 范围 | 数量 | 结果 |
|---|---:|---|
| 当前公开入口 | 39 份来源，12 个数学模型组 | 接入 21，拒绝 18 |
| 隔离修补原型 | 原拒绝来源 18 份 | 18 通过 |
| 原型兼容回归 | 已接受来源 6 份 | 6 通过 |
| 拒绝行为与旧 v8 边界 | 7 项 | 7 项符合预期，包含一次诊断期望修正后的独立复验 |

当前公开入口仍是 PyTorch v9。原型保存在开发工具目录，尚未接入应用或 CLI。

39 份来源覆盖自适应平均池化、冻结统计量的 InstanceNorm、padding 和普通平均池化。每组使用三种形状或参数上下文；Module 与 functional 调用、常量形状与结果 shape/size/numel 查询分别保留。它们是 12 个数学模型的写法变体，单独计数；原冻结 Benchmark 保持不变。

## 定位与修补范围

| 根因 | 原拒绝数 | 原型修改 |
|---|---:|---|
| Module/functional 结果缺少首次 tracing 形状元数据 | 15 | 增加已知 InstanceNorm、AdaptiveAvgPool 和 padding 的元数据规则 |
| functional AdaptiveAvgPool 缺少图映射 | 3 | 映射至已有 adaptive_avg_pool 分解，使用独立策略开关 |

修改仅涉及三个 Agent 文件的隔离副本：`pytorch_static_trace.py`、`pytorch_model_capture.py`、`pytorch_model_intake.py`。各自的原文件、补丁和副本哈希保存在 `prototype.json` 中。测试未导入或执行宿主侧模型源码；每份来源在独立 bubblewrap 沙箱内捕获，采用现有资源限制、网络隔离和锁定的 Python/Torch 环境。

24 份原型正例完成 384 组隔离模块参考比较，以及 384 组与原声明式图的独立数学/Torch 对照，最大绝对差异 `1.1102230246251565e-16`。明文对照使用 `atol=rtol=1e-12`。全部 24 份原型参考报告被当前生产来源检查器拒绝，防止原型身份冒充生产证据。

七项拒绝检查：密文数据相关分支、未知算子结果的形状猜测、自适应池化上采样、当前功能入口之外的无 batch 形状、缺少显式 InstanceNorm 近似、旧 v8 的结果元数据边界，以及旧 v8 的 functional AdaptiveAvgPool 边界。

首次探针的上采样测试已正确拒绝，测试期望漏写了诊断中的节点位置，导致探针总退出码为 1。原报告保持 6/7，不改写；只复验该项后得到 1/1，新增记录同时绑定原报告。24 份正例未重复执行。

## 状态与保留范围

- 当前公开入口：21/39 接入；18 项缺口已由原型验证，生产尚未集成。
- 编译、Hecate tracing、Poseidon 密态执行、解密及新 Agent 生成：本批均为 0。
- 付费 API、安装、下载、SDK 重建、参数/后端/reference 修改、commit/push：均为 0。
- r976 的 683 份运行源码、2,193 份案例/证据和四份冻结驱动全部匹配。103 项后端验收仍待授权，收据和后端输出目录均未创建。
- 已批准的 24 项静态 split 后端验收在 r878 关闭：24/24；它属于旧绑定的真实 Poseidon 证据，不由本次输入检查重新计数。
- 本批新增工作证据约 1.23 MiB；`git diff --check` 通过。原模型、原失败日志、原型初次失败报告与已有工作区修改保留。

## 下一步

先执行已冻结且待授权的 r976 后端批次。安全边界后，将原型接入新的显式可信输入版本，保留 v1–v9 规则，更新能力目录和新绑定，再检查新来源的 CLI、应用、程序包和代表性 Poseidon 执行。新增生成能力的真实 Agent 测试另行冻结并申请批准。

原八项编译失败、物理容量、bootstrap 精度、密文交接/分区、应用精度、ResNet20、CUDA 实机、x86 原生与空白安装继续单列。

## 记录

- `results/agent-operator-shape-intake-r980/report.json`：当前公开入口。
- `results/agent-operator-shape-prototype-r982/report.json`：24 正例与首次七项拒绝检查。
- `results/agent-operator-shape-refusal-r983/report.json`：一项节点诊断复验。
- `scripts/baseline/benchmarks/tools/fixtures/operator-shape-capture-prototype-v1/prototype.json`：隔离原型及补丁身份。

results 路径相对于 `/home/lhohy/poseidon-work/platforms/aarch64-linux`。本报告的机器记录为 `agent-pytorch-operator-shape-acceptance-r984.json`。
