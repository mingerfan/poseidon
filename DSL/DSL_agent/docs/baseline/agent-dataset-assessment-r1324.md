# 数据集精度汇总接口

`DatasetNumericalAssessment` 收集已有 `TaskNumericalAssessment` 结果，按声明的数据集计算整体误差和 Top-1。它用于应用精度验收与缓存证据复用；生成、编译和执行继续由现有 Application 管理。

## 使用

```python
from task_dataset_assessment import (
    reference_anchor, declare_dataset, DatasetNumericalAssessment,
)

# first_receipt 与各片结果均来自可信组件，binding 由宿主保留。
anchor = reference_anchor(first_receipt, expected_binding=first_receipt_binding)

batches = [
    {
        "id": "part_0",
        "sample_ids": ["sample_0", "sample_1"],
        "samples": [{"x": [0.1, 0.2]}, {"x": [0.3, 0.4]}],
    },
    # 其余分片按相同结构声明。
]
declaration = declare_dataset(
    model_request,
    batches,
    anchor=anchor,
    application_criterion={"kind": "max_absolute_error", "limit": 0.01},
)
assessment = DatasetNumericalAssessment(
    model_request,
    declaration,
    expected_declaration_binding=declaration["binding"],
)
for part_id, receipt, trusted_binding in checked_receipts:
    assessment.record(part_id, receipt, expected_binding=trusted_binding)

result = assessment.summary()
```

声明保留样本 ID、输入哈希、参考身份和指标。输入数组留在宿主侧。Top-1 配置沿用单组接口：`kind=top1_accuracy_drop`、输出名称、class axis、全体样本对应 labels，以及 `max_drop`。`part_criterion()` 按声明顺序切出每片的指标，用于既有数值适配器。

每片 1–16 组样本；汇总最多 4,096 片、65,536 组样本，声明 4 MiB、单收据 16 MiB。以上是汇总容量，张量大小、布局、图和执行资源仍按既有配置检查。

## 结果与校验

- 模型、程序包、数学参考、源码及 dtype 身份相同，且输入、指标和收据 binding 匹配后才收集。
- 重新计算逐项 CKKS 门限及误差，保持 `1e-5 + 1e-4*abs(reference)`；Top-1 单独计分。
- 最大绝对误差取全部已评估输出的最大值；MAE 按输出值数量汇总；Top-1 按标签数量加权。
- 原生源码模式使用真实 PyTorch 输出，并要求完整的原生比较与 forward 收据绑定。
- 同片重复录入同一收据返回 `already_recorded`；不同证据冲突或同一收据用于两片时拒绝。
- 缺失、取消、超时及不支持状态保留。`planned_samples`、`evaluated_samples`、`numerically_validated_samples` 和 `part_status_counts` 分别报告。
- `dataset_criterion_validated` 限于完整声明的数据集及明确指标。`original_source_accuracy_certified` 和 `formal_equivalence` 保持 false。

某片的 Top-1 下降可能超过单片门限，而整个数据集加权后的下降满足要求。汇总保存该片原始失败状态，并单独报告整体指标；任意逐项 CKKS 失败都会使整体数值验收失败。

## 本批结果

65 项控制测试全部通过：25 项数据集接口、40 项已有数学／原生模式兼容测试，0 failed/errors/skipped。包括不等长分片、25 样本汇总、多输出、重复／缺失／冲突、身份变化、dtype、指标和原生证据拒绝。测试使用注入夹具。

历史真实 MHA 收据完成完整性及汇总复核：3 组、30 个值，重复收据计数一次。CKKS 最大误差 `2.48294335092325e-9`，相对原始 PyTorch 输出的最大误差 `0.7657156480471775`；MaxAbs `0.01` 指标仍为失败。历史记录及 binding 保持。

本批新增源码 forward、密态执行、编译、API、安装、下载和 Git 提交／推送均为 0。真实多片大型数据集和完整应用精度尚待验收。691 阶段源码身份、654 保护文件及待批准 MHA 两项计划校验通过。

## 文件与接续

接口：`scripts/baseline/task_dataset_assessment.py`。控制与审计：`scripts/baseline/benchmarks/tools/dataset_assessment_v1/`。

缓存复核产物：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-dataset-receipt-audit-v2`。[机器报告](agent-dataset-assessment-r1324.json)。

初始多输出夹具失败、首次通过记录及原生证据收紧前的审计均保留。后续接入正常应用的数据集加载和有界验收调度，复用当前接口；完整模型精度、容量、bootstrap 与 CUDA 分别推进。
