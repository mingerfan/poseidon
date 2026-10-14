# PyTorch 源码包验证清单

三个来源的模型、公开权重、源码和 reference 已冻结：helper 平方、嵌套关键字广播、包级重导出 Linear。

先执行三例规则候选的 Poseidon CPU 数值验收、程序包导出重载、新输入、重复输入与缓存检查；随后执行三例真实 Agent 测试。两类来源分开计数。

测试范围已获用户总授权。当前状态为准备完成、等待产物预算。r1434 申请将全局产物额度从 32 提高到 48 GiB；原批次尚未启动。

离线最多 3 次编译/9 次密态调用，30 分钟；真实 Agent 最多 12 次生成/48 次 HTTP/12 次编译/18 次密态调用，12 小时。模型 DeepSeek deepseek-flash/high/stream，每次 1200 秒/384000 tokens，每例最多 3 轮修复和每次 3 次传输重试。API 与密态串行，编译 2/link 1，native 4 GiB/CPU150 秒，新增产物合计 4 GiB、磁盘余量 4 GiB。

外发公开图、公开权重或描述符、布局、DSL 规则与受控反馈。无明确货币上限，超时重试可能重复计费。私有测试数组、reference、规则 DSL 答案和凭据均保留在执行端。

具体案例、请求和 SDK 绑定见 [机器可读清单](agent-source-package-validation-plan-r1438.json)。
