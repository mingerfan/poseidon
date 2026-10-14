# 分阶段工作上下文与程序包复用

日期：2026-10-09，r1329。开发：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

新任务目录采用 `poseidon-worker-context-v2`（`staged-sdk-v1`），以编译、执行和验证身份绑定 SDK；Provider 和生成控制模块更新后，三项身份一致时可以继续复用已验证程序。待生成任务仍核对完整生成身份，配置漂移返回终态。

## 实现

- `application_execution/context.py`：新目录选择 v2；已有 v1 目录继续严格比较原身份。显式 `identity_policy="legacy-v1"` 保留旧策略。
- `application_execution/context_identity.py`：核验阶段身份、worker 代码、SDK 和环境；生成模块从 worker 的运行依赖中分离。
- `application_reuse.py`：从不可变任务元数据计算程序包复用身份，绑定请求内容、权重、来源、验收等级及三项后端身份。
- `application_component.py`：缓存判断使用上述身份；交付继续检查原任务和程序包。旧 `reuse_key` 字段及记录保留。

原请求哈希、Provider 适配器、预算、取消及修复接口保持。Poseidon、CKKS Runtime、Dacapo、模型、reference、参数和误差门限保持。

## 验收

86 项组件与兼容测试通过，failed／errors／skipped 均为 0。覆盖仅改变生成身份、编译／执行／验证漂移、非法缓存键、请求与验收等级变化、旧记录保持、SDK 环境检查、预算与取消。其中验证输出使用注入结果，源码捕获和 SDK 检查在实际环境中执行。

真实回归重载 r1328 的 Linear4→2 和 MLP4→4→square→2 程序包。当前生成身份已变化，三项后端身份与原包一致；在独立私有任务目录中复用原任务及程序包副本，新请求没有生成或重新编译。

两份包各执行四组新输入两次，共 4 次 Poseidon 密态执行、16 组执行输入、32 个输出值，最大绝对误差 `9.972899639443256e-7`。逐项通过 `1e-5 + 1e-4*abs(reference)`；组数包含重复执行。原任务、worker 记录、候选源码、包绑定及历史证据逐文件核验保持。

实际产物沿用原包的 Hecate tracing、IR、RuntimePlan、OperatorSpec 和 bundle，执行前核验。此次编译 0、付费 API 0。原生限制仍为 4 GiB／CPU 150 秒，后端及参考保护清单 654 项无变化。

## 使用

在已有 Poseidon CPU Nix worker 中：

~~~python
from application_execution.context import WorkerContext
from application_execution.backend import application

worker = WorkerContext.capture(host_owned_task_store)
app = application(worker)
task_id = app.submit(model)
outcome = app.run(task_id, trusted_provider)
outputs = app.execute(task_id, samples)
~~~

已有 v1 上下文记录保留原格式；选择新策略使用独立任务目录。程序包的编译验收、数值验收和应用精度等级保持各自含义。

## 下一批

两项 MHA 重测已按当前源码重新冻结为 r1330：self Attention→Linear→残差、投影后的 cross Attention。模型、公开权重、原请求、P4/w45 和数值门限与 r1317 一致；旧方案保持。新版 runner 的 54 项额度、授权、终态及清理控制测试通过，`--plan` 通过，真实执行尚未获批准。

[机器报告](agent-staged-worker-reuse-r1329.json) · [两项 MHA 方案](agent-mha-feedback-live-proposal-r1330.json)

真实验收目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-staged-context-package-acceptance-r1329`。本批未安装、下载、提交或推送。其他模型族、输入与容量、近似精度、bootstrap、CUDA 和整网继续按原计划推进。
