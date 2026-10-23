# Agent 身份分析缓存验收

2026-10-05，ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

本批完成源码身份分析的缓存调整和回归。三个既有规则模型、原 Linear CLI、程序包重载及新输入复用通过。没有新增模型生成，也没有付费调用。

## 修改

- `application_identity.py` 的缓存条目上限由 256 调整为 1,024；可序列化语法事实的大小预算仍为 2 MiB。
- 缓存键继续包含文件名和源码内容哈希，只缓存不可变语法事实。每次身份计算仍重新读取并哈希文件，核验路径、符号链接、依赖和动态加载权限。
- 新增六项安全与失效回归，覆盖大源码闭包、同大小同时间戳修改、资源文件变化、符号链接替换、大小限制及缓存淘汰。

## 测量

对当前应用的生成、编译、执行、验证四种身份联合测量，每种身份覆盖 646 个文件，其中 634 个 Python 文件。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 首次身份分析 | 5.863 s | 1.268 s |
| 重复身份分析 | 5.781 s | 0.466 s |
| 重复分析的 Python 解析次数 | 4,048 | 0 |
| 每轮文件读取次数 | 5,534 | 5,534 |
| 缓存语法事实 | 83,378 bytes / 256 条 | 126,480 bytes / 634 条 |

这是身份分析这一环节的测量，重复分析约快 12.4 倍，不能外推为整个生成或密态执行流程的加速比。新旧算法在相同当前源码字节上得到完全相同的身份记录。修改源码本身产生新身份，历史绑定保持原样。

## 验收结果

| 范围 | 结果 |
|---|---|
| 身份、应用组件、反馈、上下文和验收接口测试 | 79 passed，0 failed/errors/skipped |
| 既有 PyTorch v5 规则模型 | 3 passed，0 failed/not_run |
| 程序包导出、重载、新输入执行两次、任务缓存复用 | 3/3 |
| 原 schema 1 Linear CLI self-test | passed |
| Poseidon CPU 比较 | 46 组输入，120 个输出值，最大绝对误差 4.5789604053e-9 |

三个模型为 `modules`、`nodes` 和 `functions_nodes`，分别检查模块近似、节点近似及功能式节点近似和字典多输出。模型、请求及候选源码哈希与 r712/r720 一致；当前源码、编译产物和执行记录使用新绑定。它们是已有模型的回归，不增加模型数量。

r726 三模型包含真实 Hecate tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、Poseidon CPU 密态执行和最终解密。四组约定输入完成数值验收；重载再执行四组输入。另三组新输入执行两次，由开发验收工具及独立数学计算比较结果，本地执行接口本身保持“执行完成、未认证新输入数值”的返回口径。

Linear self-test 的两个内置错误候选分别被解析和数值检查拒绝，第三个规则候选通过；该流程验证原 CLI 和反馈链，属于脚本化候选测试。

所有比较保持：

`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`

原函数应用精度、当前版本真实 Agent 新生成、bootstrap 和 CUDA 未在本批验收。

## 完整性与资源

核对 r726 的 644 个冻结源码文件、648 个后端文件及 6 个 reference／门限／产物检查器文件。后端和 reference 文件保持原哈希；每个程序包的文件哈希、执行器不持有私钥、tc128 参数和无 bootstrap 状态均复核。

新产物约 29.9 MiB，累计 worker 用时约 185.4 秒；沿用编译最多 2、链接 1、密态串行。下载、安装、参数调整、后端修改、API、commit 和 push 均为 0。

预检的身份包装层比较错误、两个不存在的测试模块名，以及审计工具对物理输入和本地执行状态的错误假设均保留。更正工具后使用原证据复核，没有为这些工具错误重跑模型。

## 证据

- [机器报告](agent-identity-cache-acceptance-r730.json)
- 预检、原实现备份、测量、单元测试和独立审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-cache-preflight-r722`
- 真实后端回归：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-cache-backend-r726`
- 批次绑定：`ad0d8ee0b1448666f81976def3d00b46e6559fe8aa368575d2fbd1d8c11bec9f`
- CLI 证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/candidate-replay-whmj4uc6`

## 下一步

优先把已验证的中心化归约和多项式乘法重排接入通用 Agent lowering，解决 r721 留下的 LayerNorm 规则编译失败，并补充跨上下文回归。继续扩展静态索引和输入写法。

源码闭包的阶段耦合仍存在，本批只减少重复解析。新 PyTorch v5 输入的真实 Agent 验收须重新冻结范围后申请付费批准。Attention 深度、真实 bootstrap、密文交接分区、CUDA、x86 原生执行、空白安装和 ResNet20 分别保留未完成状态。
