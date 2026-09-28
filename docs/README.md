# 文档

## 使用 Agent

- [快速开始](../scripts/README.md)：环境配置、构建、生成与程序包。
- [模型输入](baseline/supported-inputs-models-configurations.md)：JSON 格式、算子和尺寸。
- [Python / 高层算子](baseline/operator-decomposition-and-python.md)：静态导入和近似计算。
- [安装与依赖](baseline/agent-setup.md) · [Linux 平台](baseline/linux-platforms.md)。

## 接入和开发

- [应用接口](baseline/application-agent-component-v1.md)：任务、状态、预算和验收等级。
- [模块与契约](baseline/agent-current-contract.md)。
- [DSL 构造与 helper 选项](baseline/dsl-options.md)。
- [编译配置](baseline/compiler-configuration.md)。
- [SEAL 产物检查](baseline/seal-artifact-gate-v2.md)。
- [近似误差与执行误差](baseline/approximation-error-decomposition.md)。
- [Benchmark](baseline/semantic-benchmark-v1.md)。
- [本机开发接续](baseline/mac-migration-handoff.md)。

## 验收记录

这些是指定版本的记录，不是安装或运行指南。JSON 报告、模型与执行产物按原绑定保留。

- [应用组件 r216](baseline/application-component-acceptance-r216.md)。
- [第二阶段 r152](baseline/stage2-acceptance-r152.md)。
- [剩余问题 r211](baseline/stage2-remaining-closure-r211.md)。
- [验证适配器 r171](baseline/validation-adapter-acceptance-r171.md)。
- [旧 worker 接口快照](baseline/agent-component-v1.md)：作为验收输入按原文保留；新集成使用上面的应用接口。

旧 Provider、迁移过程、阶段启动说明和已被替代的重复文档已删除。仍被原始记录引用的少量报告保留原文件名，不代表推荐旧流程。

## Poseidon GPU

以下文档属于 GPU 后端，与 Agent 当前的 SEAL CPU 路线分开：

- [Modswitch 与 level 约定](baseline/poseidon-modswitch-level-contract.md)。
- [GPU modswitch 验证记录](baseline/poseidon-gpu-modswitch-primitive.md)。
- [GPU drop schedule 验证记录](baseline/poseidon-gpu-drop-schedule.md)。
- [Multi-GPU 模块](../src/poseidon/mgpu/README.md)。
