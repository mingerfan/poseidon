# Poseidon 应用生成上下文 r241

已确认：新增 `explicit-v10`，用于 Poseidon CPU 的统一图请求。应用组件按公开图中的归约、标量广播、公共乘法、张量系数和多项式选择相关规则；CLI 可显式选择。生成和修复仍使用同一 Provider，没有新增规划调用。

旧 `explicit-v1`–`explicit-v9` 保持原内容与哈希；本轮修改前冻结的 54 个代表性请求均能精确重建。新上下文使用 Poseidon RuntimePlan 的 level/scale 约束说明，旧 SEAL 说明只保留在历史契约中。人工表达、私有输入和 reference 不进入新上下文。

| 验收 | 结果 |
|---|---|
| 静态、组件、兼容回归 | 90 passed / 0 failed / 0 skipped |
| RMSNorm 应用任务，compiled | 真实 tracing、编译和产物检查通过；没有数值通过声明 |
| RMSNorm 应用任务，numerical | 真实 Poseidon CPU、四组输入、16 值比较通过；最大误差 3.89174e-9 |
| numerical 程序包重载 | 真实密态执行通过；最大误差 4.18147e-9；生成与编译调用均为 0 |
| 两个等级的缓存复用 | 通过，Provider 调用为 0 |
| 原 Attention 编译失败反馈 | 原失败层保留，新规则进入真实编译失败返回路径；未密态执行 |

RMSNorm 输入和候选来自原冻结模型及已验证的人工等价程序。本轮共提交两项脚本化候选，新增真实 Agent 成功为 0。compiled 与 numerical 的包和缓存分别计分。没有修改 Dacapo、Runtime、Poseidon 库、安全参数、模型、reference 或门限。安装、下载和付费 API 调用均为 0。

初次单元测试发现一处新上下文内的旧 SEAL 提示残留，以及一处将响应格式字段误判为私有答案的测试断言，已修正；首次报告保留。应用核验脚本误读取不存在的 request.json，随后从 task.json.request 继续核验原已成功任务，没有重复生成。

基于证据的推断：组件现在能向 Poseidon 请求提供相关的公共表达指导，并在既有预算内用于修复反馈。尚未验证它对真实 Agent 成功率或成本的改善。

仍待完成：r235 独立 bootstrap 试验批准、真实刷新数值验收和 Runtime 接线、Attention 密态验收、Poseidon 上的有界真实 Agent 批次及完整 DSL 迁移回归。x86 原生、空白机器安装和 CUDA 继续暂缓。

机器报告：[r241 JSON](poseidon-application-context-r241.json)。结果目录：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-guidance-r241`。

CLI 准备命令实测成功，生成 explicit-v10／Poseidon CPU 请求；没有生成、编译或密态执行。报告中的 511 个源码哈希最终检查全部一致，git diff --check 通过。
