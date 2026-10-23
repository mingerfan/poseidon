# 空间图的近似区间检查

Agent 执行前检查器补齐了算子图 v4 的参考解释器分派。Resize、ConvTranspose1d/2d 与 GELU、SiLU、sqrt、exp 等已有近似算子组合时，使用对应版本的独立参考计算中间值并检查声明区间。

修复前，六个合法模型均被泛化的“无法计算区间”错误误拒绝。相同模型在独立参考中可计算，故障层为 Agent 输入检查。修复只涉及 execution_input_domain.py 的版本分派，使用统一 FORMATS 清单；参考实现、Poseidon、Runtime、Dacapo、参数和数值门限保持。

| 验收范围 | 结果 |
|---|---:|
| 相关控制与兼容回归 | 169 passed，failed/errors/skipped 为 0 |
| 新空间区间控制 | 20 个不同测试 |
| 独立参考与分解图差分 | 6 模型，96 组输入、400 个输出值通过 |
| 本批真实编译、密态及 Agent 调用 | 0 |

六个上下文为一维 Resize→GELU、二维 Resize→GELU、ConvTranspose1d→SiLU、ConvTranspose2d→sqrt、GELU→Resize→exp，以及公开左操作数减法→Resize→exp。覆盖闭区间端点、端点外一个浮点步长、首次越界节点、16 组输入中的稳定位置、输入与模型保持、反馈脱敏及 v1–v4 兼容。

应用控制测试注入 packing/runtime 接口：合法输入到达打包边界，越界输入在产物、密钥、执行器和生成调用之前返回类型化失败。这些控制不计为真实密态执行。分解差分使用既有明文参考门限 1e-12/1e-12；真实密态门限继续为 1e-5 + 1e-4*abs(reference)。声明的近似计算与原函数精度分别验收。

首次 150 项回归中 148 通过、2 项旧模拟测试报错；其测试夹具漏接已有产物资源预检。补齐该夹具，并加入独立产物完整性控制后，169 项通过。生产资源检查保持，原错误日志及测试源码副本保留。

新 SDK 为 438cf30898f5b3a055c3bc60c37e0923a7425677a6d7011eec02edc0125ce525，之前为 05fcafb594be96bfcbed5fafbf21b818b74d261b97c507352e390bd83b79029b。四阶段来源闭包唯一变化为 execution_input_domain.py；编译器、运行库、依赖、参数和 x86 配置保持。三个已有源码包的来源、reference、公开请求及权重重新核验通过，捕获记录保留原身份。

[r1438](agent-source-package-validation-plan-r1438.json) 未执行且保留原样；其三个案例和调用额度迁到新 SDK 的 [r1440 清单](agent-source-package-validation-plan-r1440.json)。仍等待 r1434 的产物额度 32→48 GiB 回复；当前仅余约 1.96 MiB。预算批准前继续独立的 Agent 层工作。

已确认：区间分派缺口修复、正反例与应用控制通过、新身份及旧来源可追溯。尚未验证：当前 SDK 的真实编译、密态、程序包新输入执行和真实 Agent 生成；bootstrap、CUDA 及其他通用化目标维持各自状态。

开发 checkout：/home/lhohy/Code/Poseidon，分支 lhy-agent-dsl。未安装、下载、调整网络、修改后端或安全参数，也未提交推送。

[机器可读报告](agent-spatial-domain-acceptance-r1439.json) · [后续执行清单](agent-source-package-validation-plan-r1440.json)
