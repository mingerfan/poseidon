# 资源参数读取边界

已提取 `application_planning/backend_parameters.py`，复用既有参数校验和 `process_resource_limits`，保持模数、参数身份和 4 GiB 原生限制。11 项控制测试通过，0 failed/errors/skipped；两个原 MHA 请求的真实参数读取与旧接口一致。

当前耦合来自 `compiled_resources → resources → layout → application_component`，以及 `resources → sandbox_policy → 历史探针／运行器`。接线补丁只把低层参数读取迁出完整布局模块，并保留旧函数入口；`git apply --check` 通过。补丁尚未应用到活动入口。

在与当前源码逐文件一致的 Agent 副本中应用补丁，完整依赖分析得到：

| 阶段 | 当前文件数 | 接线后副本 |
|---|---:|---:|
| 生成 | 691 | 625 |
| 编译 | 691 | 621 |
| 执行 | 691 | 621 |
| 验证 | 691 | 621 |

副本中的编译、执行和验证不再包含 Provider、凭据加载、生成控制层及旧实验运行器。修改 Provider 文件只改变生成闭包；修改参数读取文件会使四阶段闭包失效。共享数学与模型语义依赖仍然保守绑定，历史生成来源的导出策略尚未改动。

该表属于源码依赖证明。当前应用仍使用原入口，尚未进行接线后的真实编译、密态执行或程序包验收。待 MHA 重测冻结边界结束后接入补丁，再验收资源门禁、程序包和代表性旧接口。

首次控制测试把压缩传输格式当成展开请求，导致请求哈希拒绝；改用现有解码器后通过。旧测试源码、日志与失败报告保留，模型和原请求文件未改。

本轮新增模块、测试和开发接线补丁；691 个冻结源码和654 个保护文件保持。原两项只读计划仍通过，授权文件未创建。API／编译／密态／安装／下载／后端及参数修改／commit/push 均为 0。

[机器报告](agent-resource-parameter-boundary-r1320.json) · [待批准 MHA 范围](agent-mha-feedback-live-proposal-r1317.md)。
