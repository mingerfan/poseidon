# 原函数与近似图的独立精度检查

ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，2026-10-14。

新增 TaskApproximationAssessment，复用现有 Application、任务来源和隔离 PyTorch reference，比较原始源码与声明的近似计算图。每次接收1–16组样本，保留输入dtype、权重、来源、参数及reference身份。

这个入口用于原模型／近似模型的明文精度检查。密态误差继续通过现有 ApplicationAssessment 单独验收，生成、编译、执行和程序包等级保持原行为。

## 调用

~~~python
from pathlib import Path
from task_approximation_assessment import TaskApproximationAssessment
from task_numerical_assessment import SOURCE_SELECTION

result = TaskApproximationAssessment(app).assess(
    program_task,
    samples,
    expected_package_binding=program_binding,
    native_source={
        "format": SOURCE_SELECTION,
        "input_dtypes": {"left": "float32", "right": "float32"},
        "trust_local_code": True,
        "evidence_directory": str(Path(work_results) / "native-reference"),
    },
    application_criterion={"kind": "max_absolute_error", "limit": 1e-5},
)
~~~

输入名称对应具体模型。来源参考复用任务绑定的源码及公开权重；该调用只启动一个隔离原生worker。结果中的 application_metric_subject 为 declared-approximate-computation-v1，encrypted_execution 和 ckks_validation_performed 均为 false。现有程序包的原验收等级单独保留。

指标沿用最大绝对误差或带标签的 Top-1 下降。Top-1 的比较对象明确写为 original_source_top1 和 approximate_top1；max_drop=0.01 表示一个百分点，使用精确正确数比率判断边界。未指定指标时，仅返回差异，不给出应用精度通过结论。

## 验收

| 检查 | 结果 |
|---|---|
| 不同控制测试 | 83 passed，0 failed／errors／skipped；14新增、69兼容 |
| 三份原始来源 | 3/3，24组输入、64个输出值 |
| 原生精度 | float32矩阵、float64负轴contraction、float64标量中间值 |
| 声明MaxAbs1e-5 | 三例通过；最大误差4.77861211712e-9 |
| 新增实际调用 | 3个隔离原生worker；API、编译、密态执行为0 |
| 保持范围 | 原任务、真实Agent源码、程序包及分阶段SDK身份保持 |
| 时间与产物 | 111.30秒，233,472字节；处于1200秒／128MiB窗口 |

控制测试覆盖错误绑定、float32舍入、指标失败、错误标签、dtype／shape混淆、缺少信任、越域、来源篡改、终态失败和身份漂移。实际源码运行及回执核对单独计数。

本批声明样本不代表全数据集精度。完整数据集汇总、校准输入域及精度认证仍需继续接入；bootstrap、CUDA、容量与完整ResNet20保持独立进度。[机器报告](agent-approximation-reference-acceptance-r1610.json)。前一批真实密态结果见[三份程序包的Poseidon验收](agent-current-source-assessment-r1606.md)。

后端、参数、模型、reference、CKKS门限和网络保持；安装、下载、commit和push均为0。
