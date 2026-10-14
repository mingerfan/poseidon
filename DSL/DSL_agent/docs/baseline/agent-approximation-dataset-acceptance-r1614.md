# 原模型与近似图的数据集验收

ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，2026-10-14。

新增 ApplicationApproximationAssessment，复用现有任务、NPY 数据集、锁、账本、取消、期限和隔离原生 PyTorch reference。宿主可以提交数据集、分片推进、查询结果和重建控制对象继续执行。原 ApplicationAssessment 及数值验收入口保持原行为。

## 使用

~~~python
from application_approximation_assessment import ApplicationApproximationAssessment

control = ApplicationApproximationAssessment(app, reference_store)
handle = control.submit(
    program_task,
    dataset_path,
    expected_dataset_binding=dataset_binding,
    input_dtypes={"left": "float32", "right": "float32"},
    trust_local_code=True,
    application_criterion={"kind": "top1_accuracy_drop", "max_drop": 0.01},
    max_new_assessments=2,
    part_samples=16,
    total_wall_seconds=600,
)
partial = control.advance(handle, max_new_parts=1)
result = control.result(handle)

control = ApplicationApproximationAssessment(app, reference_store)
result = control.advance(handle, max_new_parts=1)
~~~

NPY 数据集和标签沿用 assessment_array_dataset.export_dataset；标签绑定输出名和类别轴。reference_store 放在平台工作目录 results 下的独立私有目录，source worker 按原限制执行。

每片1–16组，总量最多4096片／65536组，输入数组总大小最多256MiB。默认总期限3600秒，可显式选择1–14400秒。调用额度先入账，完成后保存绑定回执；未确认的中断不会自动重发。当前接口针对可信本地 PyTorch 来源。

最大绝对误差按全数据集取最大值；MAE按实际输出数量加权。Top-1使用正确预测数和实际标签数汇总，max_drop=0.01为一个百分点。尾片与完整片按样本数计入，部分数据的指标不能作为完整数据集通过结论。

数据、模型、权重、程序包或参考身份改变时拒绝原记录复用。返回结果明确标为原函数／近似计算比较；密态和CKKS验收继续单独进行。

## 本批结果

| 检查 | 结果 |
|---|---|
| 不同控制测试 | 138项有效通过：当前新增范围25项，未受影响兼容范围113项 |
| 失败／错误／skip | 当前范围均为0；首轮失败源码、日志和绑定保留 |
| 实际数据集 | 3/3，共51组输入、136个输出值 |
| 分片 | 每例16＋1，两片均完成；重载后复用首片回执 |
| 原生精度 | float32矩阵、float64负轴contraction、float64标量中间值 |
| 原任务与程序包 | 历史真实Agent候选SHA和包绑定保持 |
| 新调用 | 6个原生参考worker；API、编译、密态执行均为0 |
| 时间／产物 | 455.40秒，675,840字节 |

float32矩阵的34个标签上，原函数和近似图均预测正确18个，准确率下降为0；固定标签用于接口检查。该例最大输出差异为9.83583903214e-9。其余两例的MaxAbs1e-5指标通过，最大差异分别为2.77555756156e-17和0。

10000条合成分类记录验证了一个百分点边界和汇总逻辑，仅属于控制测试。本批实际运行为51组，不代表CIFAR-10或完整原模型精度验收。

首轮接入测试失败来自夹具改动了原任务身份，修正为在输出适配器边界注入；加权统计夹具改用准确的二进制分数以隔离统计行为。生产阈值、模型和reference保持。独立审计复核实际native数组、数学公式、NPY及标签哈希、回执账本、候选源码和程序包绑定。

## 后续

只读剖析中，已完成结果的查询耗时22.10秒，source_closure被调用66次。六个参考worker的文件时间区间约0.8秒／片。重复身份与完整性复核是该小数据集的主要开销；后续先减少确认冗余的Agent层读取，保留哈希、依赖、来源、预算和期限检查，再测量较大数据集。

校准域、实际整模型精度、声明式图的明文参考模式、容量和算子组合继续推进。Bootstrap、CUDA与完整ResNet20仍分别验收。[机器报告](agent-approximation-dataset-acceptance-r1614.json)。

本批开发仅在ARM Ubuntu；x86支持及既有修改保留。后端、参数、模型、reference、数值门限和网络保持；安装、下载、commit和push均为0。
