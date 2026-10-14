# 程序包来源导出修复与验收（r1310）

正式程序包出口已修复声明顺序丢失的问题。106 项相关控制与兼容测试通过，一个保留的布尔 mask／padding／残差模型在当前源码绑定下重新编译，并完成真实 Poseidon CPU 数值验收、正式导出、重载、新输入和重复执行。

## 修改

| 项目 | 结果 |
|---|---|
| 正式 Agent 模块 | 仅 `scripts/baseline/candidate_bundle.py` 的来源序列化与出口连线 |
| 常规测试 | `test_candidate_bundle.py` 保留原 4 项，新增 8 项 |
| 来源 JSON | 保留公开常量及嵌套字典的声明顺序 |
| 完整性 | 原完整来源检查、canonical 身份、实际字节限制与 `check_bundle` 均保留 |
| 普通 JSON 与旧请求 | 既有排序方式和请求哈希规则保持 |
| 依赖与历史 | 新四阶段身份；旧证据与旧程序包保持原文件 |

原出处重建按权重声明顺序分配绑定；`sort_keys=True` 会把 `weight_10` 放到 `weight_2` 之前。内容的 canonical 哈希相同，但按排序后数据重建会触发 `Ordered source normalization/provenance`。新出口保留原声明顺序，并在创建目录前检查编码字节上限。所有新来源包采用此表示，新文件哈希可能变化；旧包仍按原绑定校验。

## 验收

| 验收层 | 本批结果 |
|---|---|
| 静态／组件／来源与旧接口控制 | 106 passed，0 failed，0 error，0 skipped |
| Hecate tracing、Earth/CKKS 编译与 RuntimePlan gate | 1 个固定规则候选通过 |
| 原四组密态资格 | 4 组／40 值；最大绝对误差 5.164676020186221e-9 |
| 正式程序包新输入 | 3 组／30 值；最大绝对误差 4.411598308884379e-9 |
| 重载与重复输入 | 1 组／10 值；最大绝对误差 2.874093185667448e-9 |
| 合计 | 8 组执行／80 值，包含 1 组重复输入 |

误差门限继续为 `1e-5 + 1e-4*abs(reference)`。reference 是该样例已声明的低阶近似数学计算，不代表原 Softmax／网络应用精度认证。

原资格证据绑定旧版 exporter SHA，故本批使用完全相同的 request 和 DSL 重新 tracing、编译、执行；未修改旧通过报告的绑定。新程序包由正式 `PoseidonBackend.export` 产生，运输修复工具未参与。随后直接加载编译产物执行新输入，无须再次生成 DSL。候选为规则基线，`agent_generated=False`；新增 Agent／HTTP 调用均为 0。

本批约 116.33 秒。参数、reference、后端库与源码、安全和数值门限保持。654 项保护源码逐项核验；来源捕获的 113 项实现保持，四阶段生产闭包各 685 项，仅 exporter 模块 SHA 变化。ARM Ubuntu 已实际执行；x86 配置保留，x86 本批未执行。

## 证据与复核

- [机器报告](agent-formal-provenance-export-results-r1310.json)
- 原始本批报告：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-formal-provenance-export-v1/report.json`
- 新程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-formal-provenance-export-v1/program`
- [r1309 原失败与历史绑定](agent-replicated-scalar-results-r1309.md)

~~~bash
python3 -B scripts/baseline/benchmarks/tools/model_provenance_serialization_v1/inspect_current.py
~~~

旧 inspector 的生产 SHA 是历史快照，正式修复后会报告预期漂移；本批 inspector 独立校验新闭包、保护表、来源、Runtime 产物、程序包及旧证据原文件。它只核验已有证据，不执行编译或同态计算。

## 后续

两个原 Attention 样例仍因 `std::bad_alloc` 未完成数值验收，已保留原失败。下一步在 Agent 自有规则和布局层审计 mixed packed／replicated 表达，测量实际 RuntimePlan 的资源占用。MHA v16 生产接入、真实 Agent 新生成、bootstrap、扩容整模型与 CUDA 仍分别待验收。

本批无安装、下载、后端或参数修改、付费调用、commit 或 push。
