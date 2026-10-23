# 工作进程管理与构造反馈验收 r1336

完成两项 Agent 层修复：命令结束或超时后清理所属工作进程树；为覆盖公开常量等保留绑定的错误提供明确反馈。

开发位置仍为 ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。Poseidon、Runtime、Dacapo、模型、reference、参数和数值门限保持。

## 修改

| 模块 | 行为 |
|---|---|
| `application_assessment_entry/supervision.py`、`cli.py` | 独立 Linux 子进程负责追踪和清理所属进程，包括脱离原进程组的子进程；CLI 返回 `process_cleanup` 收据 |
| `native_public_loops.py`、`candidate_check_diagnostic.py` | 保留原拒绝规则，增加 `native_reserved_binding` 和赋值位置 |
| `application_diagnostics.py` | 公开修复提示要求保留常量、函数和 DSL 名称，使用新的局部变量保存中间表达式 |

历史非法候选仍被拒绝。错误消息只包含允许公开的规则和源码位置；原两份请求内容与哈希保持。

## 控制与命令验收

- 进程清理、入口和控制层兼容：29 passed，failed/errors/skipped 为 0。其中 11 项使用真实 Linux 进程。
- 保留绑定、受控反馈及兼容：39 passed，failed/errors/skipped 为 0。
- 真实系统 Python 启动 4 条命令：`submit`、`status`、零额度 `advance`、`result`。提交及查询正常，后两项按额度返回 `budget_exhausted`；四次均确认进程清理。新增编译、密态执行和 API 调用为 0。

两组控制分别保存自身源码绑定。进程控制和命令验收先于构造反馈改动完成，其相关实现随后保持。首个控制窗口超时、旧 CLI harness 的重复 Nix 入口、v2 harness 的字段错误以及重放工具的父进程误判均已保留；修复只重测受影响范围。

## Poseidon CPU 真实回归

以未修改的历史真实 Agent MHA 源码建立新绑定：

`Hecate → Dacapo-modified → Earth / CKKS IR → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU`。

| 项目 | 结果 |
|---|---|
| 真实重新编译 | 1 次，通过 |
| 密态执行、解密 | 2 次 |
| 输入 / 输出 | 8 组 / 80 个值 |
| 最大绝对误差 | 1.4454680646869278e-8 |
| 数值门限 | `1e-5 + 1e-4*abs(reference)`，通过 |
| 程序包重载与新输入 | 4 组，通过；重新生成及编译均为 0 |
| 新生成的真实 Agent 程序 | 0 |
| 后端及参考保护文件 | 654 项保持 |

原源码 SHA-256：`292bff0b21f31c77036cd063c8e641df458058b9599cba4d5096b3fce98077da`。原包、请求和指定历史报告核验保持。

当前源码身份为 `6496fe54419966cc73d0eab2c4c4e715a190fce26e653cc0899930eded32f9ee`；生成 628 项，编译、执行和验证各 622 项。新的后端回归记录绑定 `13ccf5f7d42d9c15b1e601137682febdb9980f0f9e05c1a055d84bf8112392e8`。

实测后端窗口 49.43 秒，窗口内新增产物约 20.83 MiB；当前总产物约 25.71 GiB、磁盘余量约 42.68 GiB。沿用原生 4 GiB / 150 秒、编译 2 jobs / 链接 1 job，密态串行。

## 剩余工作

[r1333 付费结果](agent-mha-tensor-axis-live-results-r1335.md)仍为 0 通过；self 候选的工作量和旋转约束尚需生成侧修复，cross 保留中断及未知调用。新反馈的真实生成效果需要新绑定下的有界测试。

应用数据集的跨命令总体截止时间、运行中取消和大数据集验收继续推进。通用输入、算子组合、容量、应用精度、bootstrap、密文分区、CUDA 实机与 ResNet20 仍独立验收。x86 配置保持，本轮实测平台为 ARM CPU。

本轮修复后新增 API、安装、下载、后端或参数改动、提交、推送均为 0。

[机器报告](agent-supervision-binding-feedback-r1336.json)。原生回归目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-native-binding-feedback-acceptance-v1`。

下一条只读命令：

~~~sh
python3 -B -c 'import json; p="/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-native-binding-feedback-acceptance-v1/report.json"; d=json.load(open(p)); print({k:d[k] for k in ("status","compilations","encrypted_calls","maximum_absolute_error")})'
~~~
