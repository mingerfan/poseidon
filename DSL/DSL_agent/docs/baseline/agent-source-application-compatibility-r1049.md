# PyTorch 输入与应用终态检查

本批核实了六种常见 PyTorch 写法，并通过生产应用接口检查对应任务状态。接入代码及待批准的 151 项验收绑定保持不变。

## 实际结果

| 范围 | 结果 |
|---|---|
| 隔离来源捕获 | 1 项接入，5 项在输入层拒绝 |
| 接入成功的写法 | `transpose → contiguous → view`；16 组明文参考差异为零 |
| 应用来源准备 | 同一项正常准备后取消，5 项返回输入层终态 |
| 重复处理终态任务 | 5/5 保持原状态，未重新捕获或修改产物 |
| 新 DSL／Hecate／编译／密态／解密／API | 均为 0 |

应用的终态检查符合当前契约，不代表六个模型均支持。公开 CLI 和应用重用了同一组参考输入，只计16组。没有新增冻结 Benchmark 模型。

## 输入缺口与下一步

| 写法 | 原因 | Agent 层修改方向 |
|---|---|---|
| `nn.MultiheadAttention` | 通用模块解析只接受一个 input | 专用 Q/K/V 参数解析、投影权重绑定，组合已有 Linear、矩阵乘和 SDPA |
| `F.max_pool2d` | functional 目标未映射 | 静态参数与形状映射，复用已有 MaxPool 分解 |
| `torch.div(x, 2.0)` | 公开量除法未映射 | 检查有限非零公开分母，再生成可信公开倒数与乘法 |
| `nn.GELU(approximate="tanh")` | 当前只登记原函数 none 变体 | 登记独立原函数语义、参考和显式近似配置 |
| `nn.ReLU(inplace=True)` | 原地修改与别名语义未开放 | 分析共享值和修改顺序，仅对可证明等价的情况做纯图转换 |

这些是输入映射或契约缺口，尚未进入编译。不能归因于 Poseidon、Runtime 或 Dacapo，也不能仅关闭检查或清除参数来开放。

首批可先处理 functional MaxPool 和公开量除法。每个新增可执行分区仍须三个上下文、组合回归、真实后端及获准的真实 Agent 验收。

## 应用反馈

五种拒绝目前都返回 `validation_failed`、`phase=complete`，错误为 `model_source_intake_rejected`、`layer=input`、`retry_policy=never_automatic`。没有继续生成或自动重试。

反馈已脱敏，但位置为 null、上下文为空，使用者不能由这一错误码区分五种原因。后续应由可信接入检查产生版本化错误码和公开约束，绑定来源后传递给应用；保留未知原因的通用错误。原始 traceback、环境、测试数组和 reference 不进入反馈。

## 保留与待验收

本批新增开发检查工具、公开小来源和报告，未修改生产或后端源码。691 个源码、5 个上下文文件、5248 个案例/证据、8 个驱动，以及648个后端文件和历史记录核对保持；原有修改与分支保留。

151 项提案仍未批准或启动。本批未安装、下载、调整参数、调用付费 API、提交或推送。Frozen reference 与数值门限保持。

[机器记录](agent-source-application-compatibility-r1049.json) · [逐项来源诊断](agent-common-source-gaps-r1047.md) · [v11 接入报告](agent-pytorch-common-v11-acceptance-r1045.md) · [待批准离线提案](agent-common-pytorch-offline-proposal-r1042.md)
