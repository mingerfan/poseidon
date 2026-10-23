# 应用数据集验收接口

ARM Ubuntu：`/home/lhohy/Code/Poseidon` · 分支 `lhy-agent-dsl` · 2026-10-09

新增 `ApplicationAssessment`，为已经生成并验证的程序包提供数据集验收入口。宿主提交任务后，可查询状态、推进执行、请求取消并取得结果。首片验收、后续分片与续跑共用一份持久账本。

## 接口

`ApplicationAssessment(app, store)` 复用现有 `Application`。`store` 是源码目录外的绝对路径，保存任务计划、输入快照、账本和绑定收据。

~~~python
from pathlib import Path
from application_assessment import ApplicationAssessment

control = ApplicationAssessment(
    app, Path("/home/lhohy/poseidon-work/assessment-store")
)
handle = control.submit(
    program_task,
    dataset_path,
    expected_dataset_binding=dataset_binding,
    max_new_assessments=2,
    part_samples=16,
)
state = control.status(handle)
state = control.advance(handle, max_new_parts=1)
result = control.result(handle)
~~~

- `submit` 固定程序包、任务、数据集、参考实现、指标和额度，保存经校验的 NPY 输入原始字节。
- `status`／`result` 返回状态、已用额度、分片进度、验收范围和诊断。
- `advance` 串行推进指定数量的未完成分片；首次参考收据在内部生成，并消耗同一额度。
- `cancel` 在分片边界结束任务。已完成任务的取消保持原结果。
- 数据、程序包或实现身份改变时，返回类型化 `AssessmentControllerError`；`feedback()` 提供受控错误码。

默认 `max_new_assessments=0`。每片 1–16 组样本，最大 4,096 片／65,536 组；单次模型、张量及 native worker 限制保持原值。收据空间默认 64 MiB，可显式设置到 256 MiB。

原生 PyTorch 精度验收显式指定：

~~~python
from task_numerical_assessment import NATIVE_POLICY

handle = control.submit(
    program_task,
    dataset_path,
    expected_dataset_binding=dataset_binding,
    reference_policy=NATIVE_POLICY,
    input_dtypes={"q": "float64", "k": "float64", "v": "float64"},
    trust_local_code=True,
    application_criterion={"kind": "max_absolute_error", "limit": 0.01},
    max_new_assessments=2,
    part_samples=16,
)
~~~

输入名称、dtype 和形状必须对应具体模型及数据集。原生参考复用任务绑定的可信本地源码，由隔离进程运行。

## 状态与恢复

任务状态包含排队、就绪、运行、成功、验证失败、能力不支持、预算耗尽、服务异常、取消和超时。结果分别给出 CKKS 数值验收与显式应用指标。

执行前先持久化消耗额度；完成后保存收据及其哈希。重新创建控制对象可从已提交状态继续，完成分片使用收据复核。中断后无法确认结果的调用保留已用额度，以 `assessment_controller_execution_uncertain` 结束。未绑定的孤立收据保留在原目录。

首版控制调用次数和收据空间。整体截止时间、Runtime 总产物空间及运行中强制取消仍属于宿主执行控制的后续接线范围。本接口是显式本地入口；当前冻结的生成 CLI 保持原行为。

## 本批验收

| 检查 | 结果 |
|---|---|
| 控制测试 | 73 passed，0 failed／errors／skipped；30 新增、43 兼容 |
| 测试范围 | 额度、持久状态、取消、重载、缓存绑定、数据及收据篡改、中断、错误脱敏和终态 |
| 新增真实执行 | 1 次 Poseidon CPU、1 个原生参考 worker；1 组输入／10 个输出值 |
| 历史收据复用 | 2 组输入／20 值；未重新执行 |
| 合计数值范围 | 3 组／30 值，CKKS 最大绝对误差 `1.859025669515013e-8` |
| 原模型精度 | 最大误差 `0.7657156485007809`，MaxAbs ≤ 0.01 未通过 |
| 重复使用 | 控制对象重建、终态读取／推进／取消保持结果，没有额外执行 |
| 时间与空间 | 406.33 秒；新验收目录约 8.71 MiB，处于 600 秒／128 MiB 范围 |

控制测试使用注入输出；真实执行单独计数。当前真实测试复用了首片收据，未直接测试“无缓存首片”的原生执行；该路径已有控制测试。进程中断恢复采用注入测试，未进行真实进程强杀实验。

工程接口验收通过，模型原函数精度返回 `validation_failed / assessment_application_criterion_failed`。两者分别记录。冻结 CKKS 门限仍为 `1e-5 + 1e-4*abs(reference)`；生成任务及程序包原等级保持。

原始执行证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-application-assessment-acceptance-v1`。
源文件、测试、收据和资源绑定见 [机器报告](agent-application-assessment-r1326.json)。

## 接续

654 个保护文件和 691 文件阶段身份保持。旧三项 MHA 付费批次已关闭；待批准两项仍未启动。README 和默认生成入口保持冻结。

本批 API、重新编译、安装、下载、后端／参数／模型／reference／门限修改、commit 和 push 均为 0。开发变更为新 Agent 控制模块、测试、验收工具及本报告；其余现有修改保留。

下一步在生成源码边界结束后接入稳定 CLI 和参数身份边界，补受影响兼容回归。模型多样性、原生源码精度、布局容量、Bootstrap、CUDA 和整模型工作继续按总计划推进。

接续检查的首次命令在 SDK 环境之外选择了历史默认构建路径，报 CMake 配置缺失。该命令未触发编译或执行；按 ARM/work root 配置的只读计划复核后，四阶段各 691 文件及 654 保护文件匹配。记录保留于 `scripts/baseline/benchmarks/tools/application_assessment_v1/postcheck.environment.failure.json`；此前固定 Nix 环境中的真实验收保持有效。
