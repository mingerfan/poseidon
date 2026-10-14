# PyTorch v26 结构化输入：Poseidon CPU 验收

15 项全部通过，包括 12 个结构化输入上下文和 3 个旧接口兼容。候选来自已冻结的规则转换器，本批 API 调用为 0。

## 结果

| 项目 | 结果 |
|---|---:|
| 字典输入 | 3/3 |
| 嵌套 tuple/list 输入 | 3/3 |
| 位置／关键字参数 | 3/3 |
| 公开默认参数 | 3/3 |
| v23/v24 兼容 | 3/3 |
| 真实 tracing、编译、密态及四组输入比较 | 15/15 |
| 程序包重载 | 15/15 |
| 新输入、重复执行、命名输入一致性及缓存 | 3/3 |
| 本次转发修复的组件回归 | 86 passed，0 failed，0 skipped |

累计 15 次编译、24 次 Poseidon CPU 密态调用，比较 81 组输入、610 个输出值。最大绝对误差 `4.477258896962155e-8`，MAE `2.19008006307619e-9`，非零 reference 最大相对误差 `3.0268313970249616e-7`，有定义的最小 cosine similarity `0.9999999999999892`。

冻结门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。真实 Hecate tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、plaintext bundle、Poseidon CPU 执行、解密及文件哈希均已核验。

## 修复与绑定

首例完成数值验证后，验收脚本误查程序包的 model_source 顶层字段；改为核对 model_provenance 与来源绑定。随后发现 ContextBackend.execute 未转发 sample_format，导致结构化新输入在执行前被拒绝；已增加格式检查与转发，并保留命名输入的旧调用方式。

两次停止记录原样保留。前三个已完成的密态验收结果保持原绑定，其余 12 个模型继续验收；程序包执行使用新的 worker 来源身份。原有 SDK 的生成、编译、执行及数值身份保持，模型、权重、候选源码、reference、后端和参数均保持。

验收绑定：`0acc87d3e40c3b527c450e5a2f5adca148df358763c23ebfa4fec7c733a51b01`。
SDK：`b9c4890b3569a998bd7d30f8252469a29d448ed7b200e834a2d9c89e60013b10`。
worker 来源身份：`7a4ed1a3f022f7aa7eb520f8ac05701bff369c3e8b4ddc148c5813d7613c53c4`。

## 范围与后续

已确认：本表中的静态输入树、公开默认值和旧接口兼容可完成 Poseidon CPU 数值验收及程序包复用。

基于证据的推断：转发改动只影响应用适配器；现有编译产物与密态执行身份可复用。

尚未验证：v26 的新真实 Agent 生成、资源上限内的所有形状组合、任意动态 Python、后续空间算子、完整 bootstrap、CUDA 和 ResNet20。

原 40 分钟窗口内完成，累计约 25.57 分钟，新增产物约 162.0 MiB。未扩预算；安装、下载、付费调用、后端／参数／网络修改和 commit/push 均为 0。

[机器可读结果](agent-input-binding-poseidon-results-r1391.json) · [原方案](agent-input-binding-offline-proposal-r1390.md) · [实现接续](mac-migration-handoff.md)

原始证据：

- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-input-binding-offline-r1390/continuation-r1391/report.json`
- `/home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/pytorch_input_binding_v1/context-adapter-fix-r1391`
