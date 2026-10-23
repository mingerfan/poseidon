# 程序包数据集验收

数据集适配器复用现有 `Application` 和已交付的 DSL 程序包，提供输入加载、串行分片验收、状态查询与取消。生成过程和程序包等级保持原接口。

## 输入

输入使用 `dataset.json` 与 NPY 数组。每个输入数组的 shape 为 `[样本数] + 模型输入 shape`，名称与输入签名一致。数组支持 float32、float64；分类标签支持 int32、int64。manifest 保存 shape、dtype、大小和 SHA-256。

数据集最多 65,536 组样本、4,096 片，数组合计上限 256 MiB；每片 1–16 组。模型自身的 rank、元素、节点、权重和密文资源配置沿用现有规则。数学参考采用 float64；选择原生源码参考时，文件 dtype 必须与已绑定的源码输入 dtype 一致。

## 导出与加载

~~~python
from assessment_array_dataset import export_dataset, load_dataset

manifest = export_dataset(
    request,                         # 宿主保留的已检查请求
    inputs,                          # {输入名: ndarray[N, ...]}
    dataset_directory,               # 新的绝对路径目录
    sample_ids=sample_ids,
    labels={
        "output": "logits",
        "class_axis": -1,             # 单个样本输出的类别轴
        "values": labels_array,
    },
)
dataset = load_dataset(
    request,
    dataset_directory / "dataset.json",
    expected_binding=manifest["binding"],
)
~~~

`labels` 可省略。NPY 加载校验头部、内容哈希、shape 和 dtype；输入值从经过校验的文件字节快照读取。

## 验收调用

参考身份由宿主保存的数值收据提供。首片已有收据可直接复用。后续调用使用同一程序包、参考身份及输入绑定。

~~~python
from task_dataset_assessment import reference_anchor
from task_dataset_evaluation import TaskDatasetEvaluation

anchor = reference_anchor(receipt, expected_binding=trusted_receipt_binding)
evaluation = TaskDatasetEvaluation(
    app,
    task_id,
    dataset,
    anchor=anchor,
    application_criterion={"kind": "top1_accuracy_drop", "max_drop": 0.01},
    max_new_assessments=3,
    part_samples=16,
    native_source={
        "input_dtypes": {"x": "float32"},
        "trust_local_code": True,
        "evidence_root": str(fresh_evidence_root),
    },
    cached_receipts={
        "part_0": {"receipt": receipt, "binding": trusted_receipt_binding},
    },
)

state = evaluation.advance(max_new_parts=1)
state = evaluation.state()
# 需要停止时：evaluation.cancel()
~~~

`native_source` 选择原始 PyTorch 源码参考；数学参考模式省略此项。原生证据根目录使用当前 work root 的 results 下的新路径，父目录先由宿主提供，适配器按需创建根目录，worker 创建各片目录。

`max_new_assessments` 默认 0，限制本适配器发起的验收调用；已存在的缓存收据按绑定复用。一次验收可能在参考检查阶段结束，调用额度仍记账。每次 `advance` 同时受总额度和本次额度限制，服务异常及超时结束本次评估，数值失败保留后继续处理独立分片。

返回状态包含额度、调用账本、各片状态、CKKS 误差和数据集指标。取消在分片边界生效。此版本为同步适配器，沿用 worker 的单次资源和超时限制；跨进程续跑与全局硬超时由后续宿主控制层接入。

## 验收结果

- 108 项测试通过：43 项新增文件与调度控制，65 项数值、原生参考和汇总兼容测试。失败、错误、跳过均为 0。控制测试使用注入的执行输出。
- 历史真实 Agent 的 MHA 程序包完成两片 Poseidon CPU 执行，共 3 组输入、30 个输出。CKKS 最大绝对误差 `4.913293216723957e-9`，MAE `8.952909744766952e-10`；冻结门限逐项通过。
- 相对原始 PyTorch 输出的最大误差 `0.7657156504221879`。显式 MaxAbs `0.01` 诊断指标返回 `validation_failed`，原模型精度未通过。接口验收通过与模型精度通过分开报告。
- 原生参考 worker 执行 2 次；缓存复用和重复读取状态新增执行 0。新增生成、编译、付费调用均为 0。
- 首次父目录准备失败，以及工具拒绝覆盖已有证据目录的记录均保留。修复只涉及本轮 Agent 适配器。
- 654 个保护文件与既有阶段身份保持；旧任务、程序包和历史结果逐字节保留。README 及 x86 配置保持原绑定。

测试报告：`scripts/baseline/benchmarks/tools/dataset_io_v1/controls.final2.json`。
真实验收：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-dataset-io-acceptance-v2`。
机器报告：[agent-dataset-io-r1325.json](agent-dataset-io-r1325.json)。

## 后续

将可选评估接入稳定的应用入口和控制层，补充更大数据集与跨进程续跑。两项 MHA 新生成测试仍对应原待批准方案；本批只补数据加载和程序包验收。原模型精度、容量、Bootstrap、CUDA 和整网目标继续分别推进。
