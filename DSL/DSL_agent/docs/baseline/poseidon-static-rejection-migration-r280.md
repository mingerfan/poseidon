# Poseidon 静态拒绝入口迁移

## 结果

`scripts/benchmark.py rejections` 已接入 Poseidon 静态适配器。原 CLI 的 offset、limit、计划模式及输出文件接口保留，增加冻结语料哈希、显式 CPU 配置和控制产物参数。

| 验收 | 结果 |
|---|---:|
| 单元、兼容和审计测试 | 50 passed，0 failed，0 skipped |
| 原拒绝任务 | 48 passed，0 failed，0 blocked，0 skipped |
| 模型校验 | 24 项 |
| 候选 AST | 9 项 |
| 请求绑定 | 9 项 |
| RuntimePlan 能力拒绝 | 6 项 |

原 16 个拒绝分区各有 3 个上下文；任务哈希、模型和候选源码保留。模型和候选反例使用原始内容，配置与布局反例绑定新 Poseidon 请求，bootstrap/upscale 反例使用 RuntimePlan。历史 HEVM/SEAL fixture 仅用于核对原任务身份，旧检查器未执行。

六项产物反例绑定三个不同拓扑、已独立数值验收的 Poseidon CPU 控制产物。检查器接受原产物，并拒绝只改一个计算指令所得的 bootstrap/upscale 产物。它们是已有控制产物的静态重验，不增加模型、密态执行或 Agent 成功数量。

## 终态和记录

缺少三个控制产物时，对应六项记为 blocked，其他 42 项继续检查，整批返回非零状态。候选反例被意外接受或失败原因不符时记为 failed。

每完成一项写入原子检查点。源码、控制产物身份或预算检查失败时停止本批，保留完成记录和 fixture，剩余项记为 not_run。测试覆盖了源码漂移后保留一项已完成记录，以及缺少产物时 42 passed / 6 blocked 的终态。

r279 的原始工具源码与结果已单独保留。最终 r280 静态绑定为 `6d3c844aa57fbaffe0d43f7db8d60d5540da399a5db9752bed488ace8b7c0af1`；50 项回归用时约 49.45 秒，48 项静态任务约 0.46 秒。

## 使用

~~~bash
python3 -B scripts/benchmark.py rejections \
  --suite scripts/baseline/benchmarks/semantic-v2-ledger-r38 \
  --frozen-suite-index-sha256 ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2
~~~

默认显示计划。执行前 24 项模型反例：

~~~bash
python3 -B scripts/benchmark.py rejections \
  --suite scripts/baseline/benchmarks/semantic-v2-ledger-r38 \
  --frozen-suite-index-sha256 ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2 \
  --offset 0 --limit 24 --execute \
  --output /absolute/platform/work/results/new-rejections.json
~~~

全量执行增加三个 `--artifact-control <Poseidon候选证据目录>`，要求三份不同拓扑的人工/规则数值验收记录。适配器重新核验请求、模型、RuntimePlan、OperatorSpec、bundle、参数和原数值记录。实际 CLI 使用已验证的 Poseidon CPU Nix 环境；每片最多 48 项，静态循环 240 秒、fixture 16 MiB，入口 300 秒。

原产物从其数值验收目录读取。新正反例写入输出文件旁的 `<文件名>.fixtures` 目录。输出采用新的文件和目录，原报告保留。

## 当前状态

应用核心 513 文件仍匹配 r274 请求绑定。此次修改位于 Benchmark 分派、开发工具、测试和文档；Dacapo、Runtime、Poseidon 库、参数、模型、reference 和数值门限保持原值。

本轮静态入口验收没有生成、编译、sandbox 执行、密态执行或 API 调用。三个 helper 的真实编译和密态结果见 [r277](poseidon-helper-compiler-migration-r277.md)，两者分开计分。

仍有 12 个原数学模型的编译容量失败，见 [r271](poseidon-dsl-migration-r271.md)。真实 bootstrap 的 r234 数值失败保留；r235 独立试验待批准。r274 的 MLP、RMSNorm、Attention 三例真实 Agent 评测仍待批准。x86 原生、空白安装和 CUDA 暂缓。

[机器可读报告](poseidon-static-rejection-migration-r280.json)。原始结果：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-static-cli-r280`。下载、安装、付费调用、commit 和 push 均为 0。
