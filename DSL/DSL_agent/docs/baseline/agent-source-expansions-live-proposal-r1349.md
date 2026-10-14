# 36 项来源扩展的真实 Agent 测试方案

状态：**待批准，未执行**。本方案替代尚未执行且已失效的 r1341 八项方案；历史方案、原请求和失败证据保留。

开发与执行位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。后端为 Poseidon CPU，原 P4/w45，`explicit-v13`。模型、公开权重、来源文件和独立 reference 沿用已有定义。

## 案例与目的

| 数量 | 来源版本 | 内容 |
|---:|---|---|
| 8 | v18 | 矩阵、分头、广播收缩、多操作数、公开权重及视图组合 |
| 6 | v17 | chunk/unbind、尾块、共享写入、公开权重与多输出 |
| 6 | v19 | 对角线、trace、标量因子与共享视图写入 |
| 6 | v20 | 一维 unbind、标量视图、快照、stack 与公开权重 |
| 8 | v21 | 平方、四次方、公开标量除法与别名组合 |
| 2 | v16 | self Attention＋Linear＋残差；投影后的 cross Attention |

合计 **36 项**，请求及数学/拓扑签名各 36 个。每项完成隔离来源捕获和独立明文参考；此处不计为 Agent 或密态通过。测试当前来源扩展、生成指导及预算内自动修复；不重跑完整 Benchmark。

## 外发内容与调用边界

外发公开模型、权重或描述符、布局、DSL 规则与受控诊断。规则 DSL 答案、私有输入、reference 输出、凭据及来源内部信息均不进入模型请求。

| 配置 | 上限 |
|---|---|
| Provider | DeepSeek `deepseek-flash` / high / stream |
| 每项生成与修复 | 最多 4 次生成（首次＋3 轮修复） |
| 每次传输重试 | 最多 3 次；每项最多 16 次 HTTP |
| 整批调用 | 最多 **144 次生成／576 次 HTTP** |
| 单次 API | 1200 秒／384000 tokens |
| 并发 | API 1，密态 1，编译最多 2，链接 1 |
| 时间 | 整批 24 小时；每项最多 4 小时 |
| 新产物 | 整批新增最多 4 GiB；每项最多 4 GiB |
| 全局空间 | 32 GiB，磁盘余量至少 4 GiB |
| native 限制 | 4 GiB／CPU 150 秒 |

无明确货币上限。超时或传输重试可能重复计费。各上限独立执行，到界保留证据并停止；不保证最坏耗时下全批完成。旧未知 HTTP 不重试，失败与未运行不计通过。

## 验收

每项必须取得当前真实 Agent 源码，经过真实 tracing、编译、RuntimePlan/OperatorSpec/bundle 检查、Poseidon CPU 密态执行、解密及独立 reference 比较。程序包重载后另外执行 3 组新输入、1 组重复输入，并检查缓存复用不新增生成或编译。

门限保持 `1e-5 + 1e-4*abs(reference)`。近似函数相对原函数的应用精度另计。完整性、环境、资源或未知调用停止相关批次；其他独立案例的失败分别保留。

## 绑定与执行入口

- 精确方案 SHA-256：`8d9388ac7797a97c31687a618a52b4d032f592f7e190216a4aebb74c504e9415`。
- 方案绑定：`06411a2e643d76d81e974349268f4bc887b84eacb0771748870e1093a1e792dd`。
- SDK 身份：`67fc349e857bff911cc66c6a889d033853ba69b86d6df3b6114a9aa477e1539f`。
- 执行工具清单 SHA-256：`461b1f16db67dec7fceb341a241476f814528be4da8ca1e9afd7d9cd1124e17a`。

执行入口 `source_live_v2/run.py` 复用 Application、Provider、真实验证和程序包接口。读取计划不会授权执行；执行要求与上述案例、预算、源码和工具一致的本次人工批准记录。36 个来源准备任务已关闭。

[官方参数文档](https://api-docs.deepseek.com/api/create-chat-completion/)核验 flash/high 和 token 范围：384000 位于文档的 393216 上限内。当前未验证凭据或 1200 秒流连接。

## 精确任务清单

| 案例 | 来源版本 | 当前 request_id |
|---|---|---|
| einsum_matrix | v18 | e43941b7c90837d474414f0ada4872dbe6dc03769a22bde2ac8dc1897a003573 |
| einsum_heads | v18 | 170472c70114fff92270d2bf53c899203f25722c519bf0b16e58420d505acc4d |
| einsum_contraction_broadcast | v18 | 4d16461f3757d18f84cd4fe0bc9ee60f1a61a39ebf7a1fe08982d8af115e5993 |
| einsum_three | v18 | b54f474fe96b4f785baf5b5d0c7903cbc42e8e8a018bedd6beb00b69ed4beca1 |
| einsum_four | v18 | 2bca77cafcf9e07e1633c1277dd33d81e725e4bdc66f07bd20895160fefda27b |
| einsum_public_weight | v18 | cf26597c4c1f4b2ad6a691e9fcbe721f85c06712b542281c815ba4bd9ca26a28 |
| einsum_view_mutation | v18 | 975294540b7da10089d6d3958f49574c0e34ed25e1878ad0c6ac1d8579954ee1 |
| einsum_partition_composition | v18 | 79cd63fb6765f72e2a47dd6b42f7a4bda114525cfd9dc0450791ec9cd96dbc8a |
| chunk_branches | v17 | c552d4a1b7082a97bed9a360505f81406c8dbd2fcd88235319feaeee6e95527f |
| chunk_projection | v17 | cdee73b6e38b98770bad5811992cbc512f272501684dce7f6f6d15155c4d20fc |
| chunk_public_weight | v17 | 9c051d352d132ed351b27abe372f32e3282fb67f3a93bd24b552ed92e72896ef |
| unbind_fusion | v17 | bddfa1aac84687694155ab15257cedb784cce207643f361b965bb44bc692c217 |
| unbind_mutation | v17 | 4bc59987e9c53c54b72597a99cc4f33c4e028e8cb5676d26621321e6930c0c6d |
| unbind_sequence | v17 | b3da7ca958fe9ea983ffdd5b84bf609b6ea4998357a6ac789d6a722e6034c6d8 |
| diagonal_output_write | v19 | ad21251e9a85f3fb7558458ca8970c27af0fee96457f04e9d6b55e0d97d1ee04 |
| diagonal_source_write | v19 | 8dbf09d56ba1b058fa9d7a5394202b1a67693645cf4c9d1c9a2d6f5818a3d307 |
| diagonal_contraction | v19 | 509888cdaf09ac3692ce2d4f155f1aa8425603e7091c74bfd8ba108c30613ecf |
| trace_scalar_factor | v19 | 2413fba8ea15ef82dc31b131ecefc2e8add6af88a57fa919a6cb82f6c5a4dfe3 |
| scalar_index_write | v19 | 9dd064b660763968297c81455364b2a37bb59986486dd1470b3bdff3375957d3 |
| three_scalar_factors | v19 | d96bd56e1d3a21ab0f2148e346b5a18d7ed2121852ad8bf14478fcb9bf25d817 |
| scalar_function_merge | v20 | a4f0179cc5613bf06955878c2d7dc2dd9bb58f51f6d7a8aeb071678c70bc4ed4 |
| scalar_shared_write | v20 | 35534299dda23d508ddba937d7e0d89e34ac7f7d65b5bac68b5a5dc4c20200f4 |
| scalar_source_snapshot | v20 | f8f5f5745b20581629c47fea052a71431f5db754d3c46f94e2ac4802cfd37eae |
| scalar_repeated_stack | v20 | 5286ddc039101cd167b0cfe768830a83df86b8d080e2720d2e3c137661c0b823 |
| scalar_singleton_view | v20 | e4d66a6922fd3f6851fa3d85b81ad15c864a3d34de0ebb41fbedbabce332a5b7 |
| scalar_public_weights | v20 | 60a1e0945a77d7bedeb0e1e38f0bbead9dd8af231337ff84f15e9a491b4256e5 |
| tensor_square_division | v21 | 98d25bba49c9de88371eec38abcd377d42329612bed9a0a28ec1231248369cab |
| linear_fourth_power_division | v21 | e2d1a23954a8e46f28aef4cd2321ea1a13a3cfd610fb220ffdfdefcfaeec12d8 |
| scalar_shared_power_division | v21 | aa8a063f908cb0c57cd4627ae5bedf7bbe97f2e5f815beeda68bc15b79127dd9 |
| snapshot_fourth_power_division | v21 | 179a4650abb9c1f8e29275b845c5eee6c951312d0c70eeb6256df0fbcf830c08 |
| diagonal_square_division | v21 | c51b276100730bb3593e82bd71d8d158d86b94aa133925447c2a9579b8733a71 |
| public_buffer_division_square | v21 | 5a2411b7ff83cd457b6d1537b495fdb27a64f18b94115e53ca0056753c7ab9e8 |
| method_power_shape_division | v21 | 05618fd6daab197e80c22e7972cbe5f80ebb72ffc85aa59c6db86e7945974526 |
| multi_input_power_division | v21 | 6b381c6d308d70d9d60258dd5f25edcda38a338b9bb6f42438dfa84082152743 |
| self_post_linear_residual__declared_shallow_v1 | v16 | f66488392d7ef0567f7a8e12282b76d7f63d6103cbeba178a8a90e3984a6377b |
| linear_into_separate_cross_none__declared_shallow_v1 | v16 | 4b1918ad274c3401fe937f1035207ef4b48fd7448e8dbbf9edd91c81cade26ad |

[机器方案](agent-source-expansions-live-proposal-r1349.json) · [离线准备结果](agent-source-expansions-readiness-r1349.md)
