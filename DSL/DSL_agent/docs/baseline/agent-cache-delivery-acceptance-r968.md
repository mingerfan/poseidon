# 程序包缓存与交付修复

应用组件已补齐缓存复用的请求核验，并将 worker 超时与取消覆盖到预检、缓存检查阶段。模型输入、旧请求哈希、Provider 和程序包格式保持。

| 检查 | 结果 |
|---|---:|
| 组件、诊断、能力、身份、来源及程序包回归 | 113 passed，0 failed/errors/skipped |
| 实际 Poseidon 适配器拒绝检查 | 4/4 符合预期 |
| 本批新编译、密态执行、解密、Agent 生成 | 0 |
| API、安装、下载、SDK 重建、Git 发布 | 0 |

## 修复内容

原控制层以状态文件中的 `reuse_key` 寻找缓存，随后检查包完整性，却没有对照包的原任务和当前请求。使用注入后端、两个合法模型复现：旧缓存索引被复制后，组件交付了另一模型的包。原复现与源码保留；该复现不是 Poseidon 密态执行。

复用与结果交付现在检查原任务 JSON 绑定、完整请求摘要、模型来源摘要、验收等级和当前依赖身份。缓存索引只用于查找。来源指向错误任务、请求不一致或元数据损坏都会在生成前终止。

新增稳定错误码：`cached_task_binding_rejected`、`cached_request_mismatch`、`cached_provenance_mismatch`、`cached_level_mismatch`、`cached_dependency_mismatch`。返回 `service_error`，层为 `delivery`，由宿主处理；公开结果保留代码，不包含来源内容或路径。

原 worker 定时器在缓存搜索之后才注册。现在预检和缓存读取也受同一控制器的取消、超时约束。新增真实进程取消与阻塞缓存读取超时回归；原生成、HTTP 额度及恢复规则保持。

## 验收范围

113 项包含 10 项新增缓存与控制回归，其余覆盖既有接口。测试使用注入后端，不能作为密态或 Agent 成功结果。

另外四项使用实际 `PoseidonBackend` 身份与程序包检查器，故障任务状态明确作为测试夹具：错缓存索引、错交付来源、原元数据绑定损坏、缺少程序包。前三项在加载包前拒绝；第四项由真实包检查器拒绝。生成／HTTP／编译／执行均为 0。

本批不扩展模型、算子或输入容量。原模型请求示例保持，SDK 工具身份与648份后端、6份reference及历史报告核验一致；原2,260条工作区状态路径保留。分支与HEAD保持，Git diff check通过。

## 执行与接续

682 份当前运行源码建立新绑定；生产代码只修改 `application_component.py`，新增 `test_application_cache_binding.py`。开发复现与检查工具在 `benchmarks/tools`。

r950 的94项提案、1367份案例／证据和4驱动保留为未执行历史绑定；由于控制层源码更新，该提案不能用于当前代码。未生成其批准收据，也未启动后端批次。后续先将已验证的 PyTorch shape 元数据原型接入显式输入版本，再对稳定源码冻结后端验收范围和预算。

bootstrap、密文分区、CUDA、原八个编译失败、真实Agent新输入、应用精度、物理容量与ResNet20仍保持独立待办。

[机器报告](agent-cache-delivery-acceptance-r968.json) · [组件接口](application-agent-component-v1.md) · [shape 接入审查](agent-pytorch-shape-composition-acceptance-r960.md)。
