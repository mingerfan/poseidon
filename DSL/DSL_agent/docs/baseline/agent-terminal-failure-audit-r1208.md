# 原批准批次失败层复核（r1208）

本记录按关闭任务的 loop 与已结算 HTTP 账本分类，原报告和状态文件保持原样。

| 终态层 | 数量 | 后续处理 |
|---|---:|---|
| 数值比较 | 11 | 检查公开常量、布局和候选数学计算，使用受控反馈修复 |
| Provider | 3 | 新的有界调用方案获准后重测；此前候选失败仍保留 |
| 静态检查 | 3 | 用已准备的类型诊断、公开绑定指导和重复检测处理 |
| 编译 | 1 | 检查表达与固定编译容量，不直接判定模型无法表达 |

其中 matrix__public_vector_r_1 与 instance__functional_input_2 的旧 failure.layer 为 numerical_comparison，实际 loop 终态为 provider_failed。该分类变化不改变失败数量，也不将此前数值失败记为通过。

待切换组件在精确反馈策略下已按终态覆盖 Provider／环境／完整性分类。当前生产仍使用原冻结代码。只读分类器 8 测试通过，0 失败／错误／跳过；API、编译、密态执行均为 0。

| 案例 | 原状态标签 | 复核终态层 | 生成／HTTP |
|---|---|---|---|
| pytorch_functional_layer_0 | numerical_comparison | numerical_comparison | 4／4 |
| pytorch_functional_layer_1 | numerical_comparison | numerical_comparison | 4／4 |
| pytorch_functional_layer_2 | numerical_comparison | numerical_comparison | 4／4 |
| pytorch_functional_group_0 | numerical_comparison | numerical_comparison | 4／4 |
| pytorch_functional_variance_1 | numerical_comparison | numerical_comparison | 4／4 |
| matrix__public_vector_r_1 | numerical_comparison | provider | 3／3 |
| instance__functional_input_0 | numerical_comparison | numerical_comparison | 4／4 |
| instance__functional_input_1 | static_check | static_check | 4／4 |
| instance__functional_input_2 | numerical_comparison | provider | 3／3 |
| instance__module1_input_0 | numerical_comparison | numerical_comparison | 4／4 |
| instance__module1_input_1 | compiler | compiler | 4／4 |
| instance__module1_input_2 | static_check | static_check | 4／4 |
| instance__module2_input_1 | provider | provider | 1／1 |
| instance__module2_input_2 | numerical_comparison | numerical_comparison | 4／4 |
| instance__unbatched1_1 | numerical_comparison | numerical_comparison | 4／4 |
| instance__unbatched1_2 | numerical_comparison | numerical_comparison | 4／4 |
| instance__unbatched2_0 | numerical_comparison | numerical_comparison | 4／4 |
| instance__unbatched2_1 | static_check | static_check | 4／4 |

三个代表模型的离线执行入口已修正为实际存在的 general-tensors-v1 资源配置。旧计划预览保留；新预览为 staged_component_cutover_v1/post-cutover-preview-r1206.json。该入口仍待原批次关闭和联合切换，尚未编译或执行。

下一命令：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/staged_component_cutover_v1/guarded_apply_joint.py
~~~

未解决范围继续包括新契约的真实 Agent 质量、common-call 与别名组合、算子／布局容量、bootstrap、CUDA 实机及整模型。
