# 本地数据集验收命令

Ubuntu checkout：`/home/lhohy/Code/Poseidon` · 分支 `lhy-agent-dsl` · 2026-10-09

新增 `scripts/agent_assessment.py`，为现有程序包任务提供提交、查询、推进、取消和取结果命令。它复用 `ApplicationAssessment`、请求上下文和 Poseidon 后端，自动进入已配置的 Nix SDK。

## 启动

~~~sh
python3 scripts/agent_assessment.py --help
~~~

公共参数：

| 参数 | 用途 |
|---|---|
| `--tasks` | 现有 Application 任务目录 |
| `--store` | 数据集验收任务的持久目录 |
| `--platform` | `aarch64-linux` 或 `x86_64-linux` |
| `--work-root` | 已配置的工作目录 |
| `--timeout` | 本条命令的完整时间上限，默认 600 秒 |

两个目录使用源码 checkout 之外的绝对路径。平台检查沿用已有配置和真实系统架构。SDK 的 Python、Torch、动态库和 worker 身份由现有上下文检查确认。

提交绑定数据集：

~~~sh
python3 scripts/agent_assessment.py \
  --platform aarch64-linux --work-root "$HOME/poseidon-work" \
  --tasks /absolute/task-store --store /absolute/assessment-store \
  submit --task TASK_ID \
  --dataset /absolute/dataset/dataset.json \
  --dataset-binding DATASET_SHA256 \
  --max-new-assessments 2 --part-samples 16
~~~

`TASK_ID` 为已有程序任务的 32 位十六进制 ID；`DATASET_SHA256` 为数据集清单的绑定值。返回 JSON 包含 `handle.evaluation_id` 和 `handle.binding`。

后续命令：

~~~sh
python3 scripts/agent_assessment.py \
  --platform aarch64-linux --work-root "$HOME/poseidon-work" \
  --tasks /absolute/task-store --store /absolute/assessment-store \
  advance --evaluation-id EVALUATION_ID --submission-binding SUBMISSION_SHA256 \
  --max-new-parts 1
~~~

`status`、`result`、`cancel` 使用相同的任务选择参数。默认 `max-new-assessments=0`，调用额度由提交者显式确定。每片保持 1–16 组输入，模型元素和物理密文资源上限沿用原配置。

原生 PyTorch 精度模式在 submit 中增加：

~~~sh
--reference-policy bound-source-native-dtype-finite-v1 \
--trust-local-code \
--input-dtypes-file /absolute/input-dtypes.json \
--criterion-file /absolute/criterion.json
~~~

dtype 文件按模型输入名称声明原始 dtype；指标文件使用已有 MaxAbs 或带标签的 Top-1 契约。缓存收据可通过 `--cache-receipts-file` 提交，仍需通过程序包、数据、参考及源码身份核验。

## 返回值

任务命令的 stdout 是一条 JSON 响应，版本 `poseidon-application-assessment-command-v1`。SDK 输出保存到工作目录内的私有调用日志。每次调用记录入口身份、参数哈希、随机响应标识、截止时间和结果绑定。

| 退出码 | 含义 |
|---:|---|
| 0 | 命令已接收，或状态为排队、就绪、运行、成功 |
| 2 | 输入拒绝或平台不支持 |
| 3 | 验证失败或验收调用额度耗尽 |
| 4 | 取消或命令超时 |
| 5 | 服务、环境或完整性异常 |

`submit` 的 0 表示已建立验收任务。实际验收结果位于 `result` 中，分别记录 CKKS 数值检查和应用指标。

截止时间覆盖 SDK 启动与当前命令。进程组终止检查使用真实 sleep 子进程验收；本批未在运行中的 FHE worker 上测试强制终止。中断后验收账本继续按既有“不确定执行”规则处理，已用额度保留。

## 本批结果

- 32 个入口控制测试与 30 个原控制层兼容测试，共 62 个不同测试通过；失败、错误、skip 均为 0。
- 初版联合测试 60 项通过；增加两个目录安全检查后，只重复了受影响的 32 项入口测试。旧代码及记录保留。
- Ubuntu 系统 Python 3.10.12 通过公开命令启动已锁定 SDK，三个独立进程完成 submit、advance、result，共 159.93 秒。
- 复用两份旧收据，覆盖 3 组输入、30 值。新增编译、密态执行、原生参考 worker 和 API 调用均为 0。
- 已有 CKKS 最大误差 `1.859025669515013e-8`；原模型误差 `0.7657156485007809`，MaxAbs0.01继续返回 `validation_failed`。
- 原任务、程序包、worker 上下文及历史记录逐字节保持。退出码 3 对应应用精度失败，与接口工程验收通过分别记录。

真实调用证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-assessment-cli-acceptance-v1`。
[机器报告](agent-assessment-cli-r1327.json) 保存源码、测试、命令和结果绑定。

本批只增加 Agent 入口、测试和工具。654 保护文件、四阶段各 691 文件、既有生成 CLI 和 README 保持；两项待批准 MHA 重测尚未启动。ARM 真实启动已验证，x86 配置保持，本批没有 x86 实机验收。

## 接续

现有应用 SDK 可直接复用控制对象；独立 CLI 适用于外部程序调用。正式生成入口继续使用原接口，避免改变当前冻结请求。

下一步在冻结边界后接线已准备的参数身份调整，补受影响回归。继续推进模型多样性、原模型精度、算子、布局容量、Bootstrap 和整模型目标。
