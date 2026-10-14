# GELU 来源真实 Agent 测试（r1306）

本批已结束：3 项失败，0 项通过。12 个候选均完成真实编译、Poseidon CPU 密态执行和数值比较；失败原因是公开多项式系数使用错误。

| 输入写法 | 生成 / HTTP | 最终状态 | 最后一轮最大绝对误差 |
|---|---:|---|---:|
| tanh_module | 4 / 4 | 数值失败，修复预算耗尽 | 1.00859555 |
| tanh_functional | 4 / 4 | 数值失败，修复预算耗尽 | 0.126075999 |
| tanh_input_keyword | 4 / 4 | 数值失败，修复预算耗尽 | 0.312630416 |

每项有四组固定输入；12 轮合计比较 288 个输出值。门限保持 `1e-5 + 1e-4*abs(reference)`。程序包导出、重载、新输入执行和缓存复用均未运行。

## 原因

原生成策略 `descriptors-v1` 把系数向量换成形状和哈希，却没有给出系数位置与预绑定常量名称的对应关系。候选反复猜测 `pN` 的用途；`pN` 编号实际没有数学顺序。

只读算术解释复核了全部 12 个候选：候选计算与已保存的解密值最大差异为 6.93e-12～2.57e-9；独立正确计算与冻结 reference 的差异不超过 2.78e-17。共同根因是生成上下文缺少系数绑定信息，不能据此认定 GELU 无法表达或 Poseidon 精度不足。

真实 Hecate tracing、Earth/CKKS IR、RuntimePlan 和 plaintext bundle 均已生成。290 项产物哈希、62 个常量 blob 的大小和哈希已复核；执行参数与产物门禁一致，使用既有 Poseidon CPU w45，无 bootstrap。

这里比较的是已声明的 GELU 多项式。原始 GELU 的应用精度未认证。

## 调用和预算

实际 12 次生成、12 次 HTTP，传输重试为 0，所有调用已结清。使用 DeepSeek deepseek-flash/high/stream。总 usage 为 621,384 tokens：prompt 117,681，completion 503,703；reasoning 501,617 已包含在 completion 中。报告没有实际金额。

计划建立到终态报告写入用时 39.37 分钟，未扩大预算。三个案例均只执行原批准范围。

## 记录与后续

原响应、源码、请求、失败报告和旧绑定保留。批次闭合前四阶段各 684 个源码文件及 5 个执行工具匹配冻结清单；闭合后才应用 Agent 层修正。654 个后端、reference 和门限保护文件保持。

新策略 `descriptors-math-roles-v2` 增加公开标量与多项式系数的绑定映射，保留旧默认及旧请求哈希。修正后的离线验收与本批结果分别记录。它形成新请求身份，原付费批准不覆盖新绑定重测。

[机器报告](agent-gelu-live-results-r1306.json) · [授权](agent-gelu-live-approval-r1306.json) · [原提案](agent-gelu-live-proposal-r1306.json) · [闭合前源码检查](agent-gelu-live-pre-cutover-audit-r1306.json)

原 76 项仍为 56 通过、20 失败；本批不与历史计数相加。安装、下载、后端修改、参数修改、提交和推送均为 0。
