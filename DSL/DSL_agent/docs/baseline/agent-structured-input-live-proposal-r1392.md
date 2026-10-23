# PyTorch v26：真实 Agent 测试方案

对 12 个已通过规则候选 Poseidon CPU 验收的结构化输入上下文进行自由生成。目的是验证来源捕获、Agent 生成、真实编译、数值验收及程序包复用的完整流程。

| 输入写法 | 案例 |
|---|---|
| 公开默认参数 | defaults0、defaults1、defaults2 |
| 字典输入 | dictionary0、dictionary1、dictionary2 |
| 位置／关键字参数 | keywords0、keywords1、keywords2 |
| 嵌套 tuple/list | nested0、nested1、nested2 |

每例保持模型、公开权重、输入输出绑定、P4/w45、explicit-v13 和数值门限。当前验收结果为规则候选；新测试必须使用真实 DeepSeek 响应。

## 调用与执行上限

- Provider：DeepSeek deepseek-flash，high，stream。
- 每例最多 3 轮语义修复，每次生成最多 3 次传输重试；全批最多 48 次生成／192 次 HTTP。
- 单次 API：1200 秒，384000 tokens；每例 4 小时，全批 12 小时。
- API 与密态均串行；编译最多 2 jobs，链接最多 1。
- 最多 48 次候选编译／72 次密态调用，含每例新输入与重复执行的程序包检查；缓存复用另核验。
- native 保持 4 GiB 地址空间／150 秒 CPU。
- 新批次产物最多 4 GiB，同时受总量 32 GiB 和磁盘余量 4 GiB 限制；触及任何边界即停止并保留证据。

外发仅包含公开计算图、权重或描述符、布局、DSL 规则和受控诊断。规则候选源码、私有测试数组、reference、凭据和私钥均不外发。

无明确货币上限；超时与传输重试可能重复计费。未知调用结果保留为未知，不自动重试历史未知调用。网络、后端、依赖和参数保持。

## 冻结与准备结果

方案绑定：`300f7f4d74f414858a5c837b83a16f052f16a53c40609eade8c739cfa0098f93`。
SDK：`b9c4890b3569a998bd7d30f8252469a29d448ed7b200e834a2d9c89e60013b10`。
离线结果绑定：`0acc87d3e40c3b527c450e5a2f5adca148df358763c23ebfa4fec7c733a51b01`。

12 份完整请求与公开请求已冻结，来源和依赖哈希已核验；38 项无收费权限、预算和转发检查通过；只读计划通过。API、HTTP、新增编译及密态调用均为 0，当前等待本方案的付费批准。

协议选项沿用当天 r1387 的官方接口核验和成功调用记录；实际执行前再次核对，发生不兼容时保留阻塞并报告。

只读命令：`POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work PYTHONPATH=scripts/baseline python3 -B scripts/baseline/benchmarks/tools/structured_input_live_v1/run.py --plan`。

[准确案例、请求与预算](agent-structured-input-live-proposal-r1392.json) · [离线验收](agent-input-binding-poseidon-results-r1391.md)
