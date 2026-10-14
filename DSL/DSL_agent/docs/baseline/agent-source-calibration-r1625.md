# 原始 PyTorch 校准域观察

ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，2026-10-14。

新增被动模块观察入口。原始模型保持dtype和权重，在既有bubblewrap隔离内运行；读取指定模块输入的范围、均值和标准差。观察前后的forward输出在本批输入上逐项一致，hook在结束后移除。

## 入口

~~~python
from task_source_calibration import TaskSourceCalibration
from source_calibration_contract import FORMAT
from benchmark_graph import digest

selection = {
    "format": FORMAT,
    "purpose": "calibration",
    "selectors": [{
        "id": "activation_input",
        "module": "activation",
        "argument": 0,
        "projection": {"kind": "tensor"},
        "domain": [-1.0, 1.0],
    }],
}
result = TaskSourceCalibration(app).observe(
    task_id, samples, observation_directory,
    expected_package_binding=program_binding,
    selection=selection,
    expected_selection_sha256=digest(selection),
    input_dtypes={"x": "float32"},
    trust_local_code=True,
)
~~~

模块路径采用named_modules名称，空字符串表示入口模块；argument可为位置索引或显式关键字名。每次1–16组、最多32个选择器。输出目录使用平台工作目录results下的新目录。

projection支持：

| 类型 | 观察值 |
|---|---|
| tensor | 模块指定输入 |
| layer_norm_variance | nn.LayerNorm原始dtype下的variance＋eps |
| rms_mean_square | 指定输入最后一轴的mean(x²)＋声明epsilon |

RMS投影是显式配置的计算统计，不自动证明任意自定义模块采用了同一公式。当前观察点为模块hook；纯功能算子FX节点的独立观察仍待接入。

domain为现有公开区间时，报告域内／域外状态；None表示仅收集统计。观察结果不自动修改近似区间、参数、模型或reference，也不拟合多项式。记录用途明确为calibration，test／validation角色在此入口拒绝。

## 本批验收

- 29个不同控制检查通过，失败／错误／skip为0。覆盖dtype、原输出一致、关键字、模块复用、别名输入隔离、预算、hook清理、报告篡改、信任与任务绑定拒绝。
- 三个独立原生观察案例通过，共9组输入：float32 GELU输入、float64 LayerNorm variance＋eps、float64 RMS mean-square＋eps。它们是观察器及ABI测试，未计为Agent模型语义通过。
- 两份已有真实Agent任务完成绑定观察，共6组输入；原候选、程序包与原任务保持，原验收等级单列。
- 五个原生worker实际执行；API、重新编译、密态执行均为0。
- 当前SDK保持929e8131…；旧观察报告及源码绑定保留。

首轮清单遗漏v38 call字段，在启动worker前被拒绝；随后遗漏只读模型输出契约依赖，导入失败。两个失败现场保留，修正后通过。没有删除检查或更改测试门限。

隔离、网络空间、4GiB内存／CPU150秒／文件16MiB／128句柄限制保持。原始dtype、注册状态、随机状态及重复forward检查继续执行。观测统计没有进入生成提示。

## 后续

先将观察记录绑定到声明校准数据集并汇总，补功能算子节点范围。校准统计与原函数／近似图误差、CKKS误差分别记录；区间和参数须在生成前冻结。当前五个小案例不构成完整模型或CIFAR-10精度认证。

[机器报告](agent-source-calibration-r1625.json)。容量、通用算子、Bootstrap、CUDA与完整ResNet20保持独立进度。

仅ARM Ubuntu开发，x86及既有修改保留；后端、参数、模型、reference、数值门限和网络保持，安装／下载／commit／push为0。
