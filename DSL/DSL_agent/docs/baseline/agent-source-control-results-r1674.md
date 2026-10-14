# 源码校准与响应流控制

## 实现

源码校准可识别 Conv1d/2d、推理 BatchNorm1d/2d、AvgPool1d/2d、AdaptiveAvgPool1d/2d。
这些模块沿用既有模型捕获与算子分解；准备阶段绑定类型、参数及模块路径。
激活近似仍按节点、模块、默认配置选择，未知计算模块继续返回明确拒绝。

响应流诊断增加 response_bytes/line_bytes 分类、固定上限和实际字节计数。
总响应32MiB、单行1MiB保持；超过上限的响应继续拒绝，stream_limit继续不重试。
应用返回 provider_response_stream_limit 终态；凭据、原始响应文本及reference不进入诊断。

## 验收

| 验收 | 范围 | 结果 |
|---|---|---|
| 响应流/应用控制测试 | 111项 | passed=111，failed/error/skipped=0 |
| 源码元数据/准备测试 | 100项 | passed=100，failed/error/skipped=0 |
| 原始源码校准与捕获 | 8类模块×3上下文 | 24/24，两个明文参考通过 |
| 规则候选真实Poseidon | Conv2d→GELU、BatchNorm1d→Sigmoid、AvgPool2d→Sigmoid | 3/3 |
| 程序包 | 3包重载、新输入、重复执行 | 通过 |

源码验收覆盖原生float32带权重模块及float64无权重模块。
初批12项在预检被拒绝：测试包将NPY留在子目录，且导出dtype与原模型不匹配。
旧包及失败记录保留；仅补验这12项，原模型源码不变、公开权重数值逐项一致，
采用与原生模块匹配的float32声明及权重格式，模型捕获仍显式转换为float64。

真实链路完成3次编译、9次密态调用，使用规则候选，API=0。
首次12组64值，max absolute error 6.6706104182e-10。
重载/新输入/repeat12组64值，max absolute error 5.3546264948e-10。
门限保持 abs(actual-reference) <= 1e-5+1e-4*abs(reference)。

本批不计为真实Agent生成通过，也不认证原函数应用精度。
历史LayerNorm真实Agent批次单列：[r1669](agent-layernorm-live-results-r1669.md)，1/3通过、2项stream_limit服务错误。

## 身份与后续

当前SDK 11e364f58973761e602518d9adef1a804fd683b5a9c9476a4bbb6b4df854d5e4
审计绑定 d0e52e30307b68c08f19ca4ca93fb33fc1dc41d3bf95dfa7970093b6d504f77f
旧SDK与旧报告保持原绑定。
执行目录：ARM work root/results/agent-source-transparent-modules-r1671、
agent-source-transparent-modules-repair-r1672、agent-source-transparent-poseidon-r1673。
审计：scripts/baseline/benchmarks/tools/source_transparent_modules_v1/audit-r1674.json。

下一步将已核验的高层模型数学语义和分解来源纳入生成上下文，减少只呈现展开基础图的歧义；
再冻结新增入口和LayerNorm失败项的真实Agent请求。复合近似点、原函数精度、容量、bootstrap、
CUDA和整模型仍在总计划中。

本批仅修改Agent自有代码、测试、文档。后端库、参数、reference、门限、网络、x86配置保持。
安装、下载、commit、push、PR=0。
