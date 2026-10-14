# 原始 PyTorch 输出与密态数值验收

开发环境：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。本批更新 Agent 的可选数值适配器和测试，Poseidon、Runtime、Dacapo、参数及原数值门限保持。

`TaskNumericalAssessment.assess()` 新增显式 `native_source` 选项，复用同一个 Application。原入口继续使用独立 float64 数学参考；选择新选项后，原始 PyTorch forward 在现有隔离环境中执行，保留指定 dtype、权重和输出结构。调用者保留程序包 binding，并为每次参考执行指定新的证据目录。

```python
from task_numerical_assessment import TaskNumericalAssessment

assessment = TaskNumericalAssessment(app).assess(
    task_id,
    samples,                         # 每次 1–16 组，名称和 shape 与签名一致
    expected_package_binding=package_binding,
    native_source={
        "format": "poseidon-source-assessment-selection-v1",
        "input_dtypes": {"x": "float32"},
        "trust_local_code": True,
        "evidence_directory": absolute_new_directory,
    },
    application_criterion={"kind": "max_absolute_error", "limit": 0.01},
)
```

上例的精度指标由应用明确给定，和 CKKS 门限独立。证据目录位于既有 work root 的 `results` 内。可信源码来自任务已绑定的 PyTorch 包；JSON 图使用数学参考。静态 tuple/list/dict 等输出仍按已捕获的命名张量绑定处理。

结果分开记录：

| 字段 | 比较对象 |
|---|---|
| `decomposition_comparison` | 独立近似计算与分解图 |
| `ckks_comparison` | 解密输出与分解图参考，保持 `1e-5 + 1e-4*abs(reference)` |
| `source_vs_registered_mathematics` | 原始源码与登记的原函数数学计算，识别 dtype 或函数变体差异 |
| `source_conversion_and_approximation_comparison` | 近似计算与原始源码，包含类型规范化、分解表示和近似带来的差异 |
| `source_total_comparison` | 解密输出与原始源码 |
| `application_criterion` | 应用明确给定的最大绝对误差或 Top-1 下降指标 |

`reference_policy=bound-source-native-dtype-finite-v1` 表示应用指标以实际源码输出为参考。原有数学模式及其 Top-1 字段保持；原生模式使用 `original_source_top1` 和 `approximate_computation_top1`。CKKS 失败始终保留，Top-1 通过不能替代逐项比较。

`source_application_criterion_validated` 仅表示给定样本上 CKKS 和显式应用指标均通过。`source_application_validation_scope` 保存样本数量、输入和指标哈希。完整数据集精度与形式化等价仍单独验收。已有任务结果、包等级、生成来源和历史记录保持。

## 本批验收

- 控制测试：47 passed，0 failed/errors/skipped。含 22 项旧数值接口兼容、18 项新原生模式控制、7 项来源任务边界。控制执行和参考输出来自注入夹具。
- 真实执行：历史 Agent 布尔 mask MHA 的同一 DSL 和程序包，原任务逐字节快照后建立当前 worker。原始 PyTorch forward 和 Poseidon CPU 新输入执行均完成：3 组、30 值。
- CKKS 最大绝对误差 `2.48294335092e-09`，MAE `4.09343491062e-10`；原始源码与独立原函数数学计算差异 `1.11022302463e-16`。
- 解密输出相对原始源码误差 `0.765715648047`。显式诊断指标 MaxAbs `0.01` 返回 `assessment_application_criterion_failed`；CKKS 数值检查通过。该案例的原模型精度未达标，接口正确返回失败。
- API 0，重新编译 0，实际密态执行 1 次。生成源码没有修改，也没有新 Agent 生成。
- 最终冻结 MHA 两项只读计划通过，691 阶段文件身份、654 保护文件保持；其付费重测没有启动。

初始失败和日志保留：隔离 worker 的受限源码导入缺少参考身份登记、测试夹具输入格式错误，以及源码快照按引用持有的漂移检查问题。新身份登记仅作用于本适配器；导入依赖继续由可信源清单、只读挂载和请求哈希约束。最后一项修复为执行前保存独立快照。

## 文件与后续

接口：`scripts/baseline/task_numerical_assessment.py`。控制与离线验收：`scripts/baseline/benchmarks/tools/native_assessment_v1/`。

真实证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-native-criterion-acceptance-v1`。机器报告：[agent-native-source-assessment-r1323.json](agent-native-source-assessment-r1323.json)。

下一步是有界数据集精度汇总及来源参考复用；再接入非冻结的应用路径。待批准 MHA 重测、资源身份接线、bootstrap、CUDA 和整网验收分别继续。当前 README 和既有 CLI 行为保持。

原生源码模式的数学比较、CKKS 门限和应用精度分别计分。本批接口负例通过，不计为 MHA 原模型精度通过或新增 Agent 模型通过。
