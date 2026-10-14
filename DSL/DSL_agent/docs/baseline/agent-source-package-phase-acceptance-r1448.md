# PyTorch 多文件源码包阶段结果

三种来源全部通过规则基线和真实 Agent 的 Poseidon CPU 验收。模型数量为 3；两条轨道分别计分。

| 模型来源 | 规则候选 | Poseidon 数值与包复用 | 真实 Agent | HTTP 次数 |
|---|---|---|---|---|
| square_helper | 通过 | 通过 | 首次通过 | 1 |
| nested_keyword_broadcast | 通过 | 通过 | 首次通过 | 1 |
| reexport_linear | 通过 | 通过 | 首次通过 | 1 |

- square_helper：本地 helper、模块组合、平方与公开加法。
- nested_keyword_broadcast：嵌套 Python 包、关键字双输入、广播、归约与命名多输出。
- reexport_linear：工厂函数重导出、独立 NPY 权重和 Linear。

## 测试结果

| 范围 | 结果 |
|---|---|
| 相关控制测试 | 186 passed，failed/errors/skipped 0 |
| 执行器账本与完整性控制 | 12 passed，failed/errors/skipped 0 |
| 规则基线 | 3/3，3 次编译、9 次密态调用 |
| 真实 Agent | 3/3 首次生成通过，3 次编译、9 次密态调用 |
| 程序包 | 两轨各 3 包：重载、新输入、重复执行和缓存复用通过 |

规则轨和 Agent 轨各验证 24 组输入、112 个输出值；其中初次验收使用每例四组输入，重载后追加三组新输入和一组重复输入。最大绝对误差分别为 1.63199904e-07 和 1.78016025e-07。每个输出按 1e-5 + 1e-4*abs(reference) 检查。

已核对真实 Hecate tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、Poseidon 后端身份、真实密态执行及最终解密结果。程序包的模型来源和哈希保持，重载执行无需重新生成或编译，执行器仅获得评估密钥。

真实生成使用 DeepSeek deepseek-flash/high/stream、explicit-v13、P4/w45。实际 3 次生成／3 次 HTTP、修复 0，使用 54,520 tokens。服务返回的 usage 已保留；费用字段未提供，未据此估算账单金额。本批最多 12 次生成／48 次 HTTP 的预算已结算，未使用部分不自动用于其他模型。

## Agent 修改

应用任务的快照完整性检查支持显式 v29 的嵌套文件目录和空初始化文件。沿用现有无链接源文件读取器，校验完整目录清单、文件长度、哈希及清单解析字节一致性。旧平面目录检查保留。

初次原任务在旧快照检查处停止，API、编译和密态调用均为 0，记录保留于 agent-source-package-offline-r1444。修复后使用新源码绑定；原模型、权重、独立参考、候选预算和公开请求保持。

当前源码身份：4001c16ea743d6e8f96e82b07c5fd482c16c872f38ae4317e72d1df6a406c95d。分支 lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。本轮后端库、依赖、安全参数和网络保持；安装、下载、commit、push、PR 均为 0。

## 使用范围

多文件输入显式选择 poseidon-trusted-pytorch-v29，并通过 --source-root 与 --helper 登记包和依赖文件。上限仍为 8 个源码文件、64 KiB、路径深度 8；公开权重通过独立 NPY 文件绑定。默认输入版本 v18，latest 为 v28。

本轮证明三种小模型来源能够经过正常应用入口完成生成与交付。输出仍为单份 Agent DSL 加可信依赖、权重、配置和编译执行产物的程序包。更复杂跨模块组合、布局容量、近似区间和应用精度按后续计划分别验收；bootstrap、CUDA 与完整整网仍保留独立状态。

## 文件和下一步

完整证据位于 Ubuntu：
- /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-package-offline-r1446
- /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-package-live-r1446

当前已验证的真实 Agent 程序包：
- square_helper: /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-package-live-r1446/square_helper/tasks/cee32cb4e2b0454ba1dda5ac8aff10da/program
- nested_keyword_broadcast: /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-package-live-r1446/nested_keyword_broadcast/tasks/de8e077932564484b79c0828defff76d/program
- reexport_linear: /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-package-live-r1446/reexport_linear/tasks/131c9d4c3acf4ccc80438b0220fb3ddd/program

机器可读报告：docs/baseline/agent-source-package-phase-acceptance-r1448.json。源码冻结已释放，全部任务终态 succeeded，当前批次结束。

用户已批准全局产物上限 48 GiB；当前约 32.079 GiB，客体可用磁盘约 37.46 GiB。每批 4 GiB、余量 4 GiB及其余预算保持。下一步按通用化清单补充跨模块与算子组合，冻结新增任务后沿用已批准的计划内测试权限。

下一条只读命令：

python3 -B -m json.tool docs/baseline/agent-source-package-phase-acceptance-r1448.json
