# Replicated scalar 规则表达策略

本目录保留 Agent 层的表达策略草案。原统一图、公开权重、period-4 输入输出绑定、
编译配置和 reference 保持不变。输入的每个逻辑元素只提取并复制一次；
中间 reshape、广播、归约使用独立密文元素，最后依原 chunks 打包。

`replicated_scalar_lowering.py` 生成可信规则基线，受原 AST、源码和展开操作预算检查。
缺少预绑定标量的公开权重使用精确 float64 字面量，每项记录原常量坐标和来源哈希。
这种规则侧权限与真实 Agent 的权重访问接口分别登记。

`acceptance_window.py` 运行相关控制及三个原低阶近似 MHA 样例的真实后端验收。
`reload_acceptance.py` 单独验收来源元数据的顺序保持序列化、包重载与新输入执行。
每次产生独立报告和程序包，原失败目录及绑定保留。

生产输入接口和后端库保持原状。本目录结果按规则基线计分；
真实 Agent 生成、原始 Softmax 的应用精度、生产 MHA 激活和 CUDA 另行验收。
池化、卷积、BatchNorm 和 Chebyshev 分支的覆盖以各自测试记录为准。

完整阶段结果见 `docs/baseline/agent-replicated-scalar-results-r1309.md`。
