# PyTorch 外部状态权重：真实 Agent 测试提案

新增 v3 入口的六项模型已完成独立参考、Poseidon CPU 密态数值比较及程序包重载。本提案验收真实 Agent 在新入口下的生成能力，尚未执行付费调用。

案例为 MLP、Conv→Pool、残差、BatchNorm 多输出、带公开增益的矩阵融合、形状变换→Linear。使用已有外部 NPY 权重和原数值 reference，不向 Agent 提供规则候选。

| 案例 | request_id | 外发上下文字节 |
|---|---|---:|
| pytorch_state_mlp | f0129557d31183a0c0a5f598d3bb727312433268485352c5797a13cfeff9f04b | 34620 |
| pytorch_state_conv_pool | 37a2b3528859fded8156ef6d896b6ef37f0b9e27837f5a0af95f5feb1e47d1c5 | 33394 |
| pytorch_state_residual | eb33785cf0bccfa6f1e23628f1179db8c9753041688abf1d9e8fcd96240ba352 | 31142 |
| pytorch_state_batchnorm | 51a671c85539f422aee72789ddd88302c5358d7f611c3088f0055cce656fdec3 | 30678 |
| pytorch_state_matrix_fusion | bcf5b77cb7e2427aaf9b1aece291f03c6ee7b046f8c3a5ab55043c3ce447d1d8 | 32114 |
| pytorch_state_shape_linear | c37ca847f2ec24d87bc012ee455dfc15e26c3189cb6981dfd1e992406e85325f | 32839 |

## 绑定与预算

- 源码、模型、来源、依赖和请求冻结见 [JSON](agent-pytorch-state-live-proposal-r635.json)。
- 绑定：`d3377af18686f84d36e58a5b24b36407bf1513273c86cb8543d04d44627d7623`。
- DeepSeek `deepseek-flash`，high，stream；explicit-v10，Poseidon w45。
- 每例最多 3 轮修复、4 次生成；每次生成最多 3 次传输重试、4 次 HTTP 尝试。
- 六例最多 24 次生成、96 次 HTTP；单次 1200 秒、384000 tokens。
- API 与密态串行；编译最多 2 jobs、链接 1 job。
- 全批 12 小时，每例 4 小时、4 GiB；全局产物 32 GiB、磁盘余量 4 GiB。
- 无明确货币上限；超时及传输重试可能重复计费。
- 外发限于公开图、权重或描述符、布局、合法 DSL 规则与受控反馈。私有输入、reference、来源源码、规则答案及凭据留在本地。

成功标准为真实生成、真实编译、Poseidon CPU 密态执行、解密与四组逐项比较，并导出、重载程序包。门限保持 `1e-5 + 1e-4*abs(reference)`。人工程序另计。首次通过、修复通过、各层失败与预算耗尽分别记录。

此前 14 项 r620 的授权批次已完成；本提案需独立批准，额度不从旧批次转移。当前代码不变时，可使用现有 `run_application_matrix.py` 的批准绑定入口执行。

[离线验收](agent-pytorch-state-weights-acceptance-r636.md)
