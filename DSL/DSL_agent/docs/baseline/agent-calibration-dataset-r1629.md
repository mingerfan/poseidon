# 校准数据集接口

TaskCalibrationDataset 将原始 PyTorch 模块的观察结果绑定到校准数据集，汇总输入范围、均值和总体标准差。数据使用已有 NPY 格式，保留 float32／float64；每片 1–16 组。

## 使用

~~~python
from task_calibration_dataset import TaskCalibrationDataset, DECLARATION
from benchmark_graph import digest

declared = {
    "format": DECLARATION,
    "role": "calibration",
    "dataset_binding": dataset_manifest["binding"],
    "source": {"name": "my-calibration-data", "revision": "v1"},
}
collector = TaskCalibrationDataset(
    app, task_id, dataset_path,
    expected_dataset_binding=dataset_manifest["binding"],
    calibration_declaration=declared,
    expected_declaration_sha256=digest(declared),
    expected_package_binding=program_binding,
    selection=selection,
    expected_selection_sha256=digest(selection),
    input_dtypes={"x": "float32"},
    trust_local_code=True,
)
collector.record(part_id, observation_frame,
                 expected_binding=observation_frame["binding"])
summary = collector.summary()
~~~

输入为已有成功任务、NPY 数据集和[模块观察选择器](agent-source-calibration-r1625.md)。宿主从 batches() 取得每片输入，在既定预算内调用 TaskSourceCalibration.observe()，再记录返回回执。

续用时将 receipts 作为 cached_receipts 传入。数据内容、分片方式、选择器、数据声明、任务、依赖及原证据文件均须匹配。汇总器只接受完成的回执；尝试账本、取消和执行预算由宿主管理。

均值和标准差按实际观察元素数加权，最大值、最小值和域外调用数按所有完成片合并。未完成的数据集返回 partial；全部完成返回 observed。已有程序包的验收等级单独保留。

role 及来源由提交者声明并绑定，接口检查其一致性，不能独立证明数据来自训练集。统计不自动拟合近似或修改区间；功能算子 FX 节点观察仍待接入。

## 验收

| 项目 | 结果 |
|---|---|
| 当前控制检查 | 47 passed，0 failed／error／skip |
| 数据集 | dot、矩阵乘、标量中间值，3/3 |
| 输入 | 51 组，观察 187 个值 |
| 原生 worker | 6 次，全部成功 |
| 分片复用 | 每例 16＋1，首片复用、尾片观察通过 |
| 当前收集器复核 | 复用上述 6 份观察；新 worker 为 0 |
| API／编译／密态调用 | 0／0／0 |

三份来源均为已有真实 Agent 任务；本批只验收校准观察及汇总，未重新生成 DSL。原任务、程序包和历史证据保持。原模型应用精度与 CKKS 数值验收分别计分。

早期两轮控制夹具缺少任务适配器和 source 路径，分别产生 22、9 个 error。日志及源码快照保留；修正测试夹具后通过，生产检查未放宽。

新增 task_calibration_dataset.py 及其测试；README、应用接口说明和接续文档已更新。后端、模型、reference、参数、门限、网络、x86 配置保持。安装、下载、提交和推送为 0。

[机器报告](agent-calibration-dataset-r1629.json)。当前数据集是固定随机接口验收数据，完整模型精度、真实校准数据、容量、Bootstrap、CUDA 和 ResNet20 仍需各自验收。
