# 程序包新输入数值评估

`TaskNumericalAssessment` 为现有 `Application` 增加显式数值评估。它执行已交付的程序包，计算独立 reference 并返回逐项误差；任务状态、程序包等级和生成记录保持原值。

```python
from task_numerical_assessment import TaskNumericalAssessment

# app 为现有 Application；binding 由宿主保存。
report = TaskNumericalAssessment(app).assess(
    task_id,
    samples,
    expected_package_binding=binding,
)
```

每次接收 1–16 组符合签名的样本。执行前检查近似区间、程序包身份和请求绑定；执行后检查命名输出、数量、shape 与执行端密钥约束。

| 返回内容 | 比较范围 |
|---|---|
| `decomposition_comparison` | 声明的近似计算与分解图 |
| `ckks_comparison` | 解密输出与分解图 reference，门限 `1e-5 + 1e-4*abs(reference)` |
| `approximation_comparison` | 近似计算与登记的原数学函数 |
| `total_comparison` | 解密输出与登记的原数学函数 |
| `application_criterion` | 宿主显式指定的指标 |
| `package_validation` | 程序包交付时的验收等级 |

可选指标：

```python
application_criterion={"kind": "max_absolute_error", "limit": 0.01}

# labels 按样本顺序，与去掉类别轴后的输出 shape 对应。
application_criterion={
    "kind": "top1_accuracy_drop",
    "output": "logits",
    "class_axis": -1,
    "labels": labels,
    "max_drop": 0.01,
}
```

CKKS 比较始终逐项进行。指标失败返回 `validation_failed`，既有任务和程序包仍可复用。指标未指定时保留近似误差，指标状态记为 `not_requested`。

reference 范围是登记数学图的 float64 计算。原始 Python/PyTorch forward、原始 dtype 及完整数据集精度另行验收；`original_source_accuracy_certified` 当前为 `false`。输入和逐项比较属于宿主数据，生成模型仍只接收原有受控反馈。

## 本轮验收

- 控制测试：22 passed，0 failed/error/skipped；使用注入执行 fixture。
- Poseidon CPU：未改动的历史 Agent MHA 源码程序包，3 组新输入、30 值通过。
- CKKS 最大绝对误差 5.218970255960187e-09；分解差异 5.551115123125783e-17。
- 原数学函数近似误差 0.7657156493384426；总误差 0.7657156472095206，原函数应用精度未认证。
- 新 API／编译 0、真实密态执行 1 次；参数、模型、reference、门限与原包等级保持。
- 来源为历史真实 Agent DSL 的脚本化交付重放；原包绑定、旧生成证据和任务保持原样。

首次启动在 worker 记录检查处拒绝，尚未密态执行。现有 backend 总源码摘要包含新加的顶层模块，旧 worker 记录因此与当前记录不同。失败和原记录保留；复制 96 个文件、8,974,528 字节的任务快照到独立目录，校验逐字节一致后建立当前 worker，完成验收。旧失败在 `agent-task-assessment-offline-v1`。

源码新增可选适配器、测试和开发工具；691 个冻结文件和 654 个保护文件保持。原两项 MHA 待批准方案的只读计划仍通过。

机器报告：`docs/baseline/agent-task-numerical-assessment-r1321.json`。

Ubuntu 密态证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-task-assessment-offline-v2/report.json`。

## 后续

补原始源码与 dtype 的精度评估及数据集指标；冻结 MHA 边界结束后接线已检查的资源参数补丁。本接口完成程序包新输入验收，完整近似精度流程和其他模型另有验收范围。

待批准 MHA 方案还冻结了 README 哈希。新增接口说明曾触发该只读检查，已精准撤回本轮新增的段落，README 既有内容和原哈希恢复；接口用法保存在本独立文档，冻结范围不变。
