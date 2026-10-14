# 原始 FX 节点校准

新增原始 FX 图的检查和观察入口，适用于直接调用 PyTorch 函数或 Tensor 方法的静态模型。统计来自原始计算，覆盖节点输入和输出；原 dtype、权重与计算保持。

## 接口

~~~python
from task_source_calibration import TaskSourceCalibration
from source_calibration_contract import FX_FORMAT
from benchmark_graph import digest

calibration = TaskSourceCalibration(app)
inventory = calibration.describe_fx(
    task_id, samples, inventory_directory,
    expected_package_binding=program_binding,
    input_dtypes=input_dtypes,
    trust_local_code=True,
)
graph = inventory["calibration_report"]["fx_graph"]
selection = {
    "format": FX_FORMAT,
    "purpose": "calibration",
    "mode": "observe",
    "graph_sha256": graph["graph_sha256"],
    "selectors": [{
        "id": "gelu_input",
        "node": "gelu",
        "op": "call_function",
        "target": "torch.nn.functional.gelu",
        "point": "input",
        "argument": 0,
        "projection": {"kind": "tensor"},
        "domain": [-1.0, 1.0],
    }],
}
frame = calibration.observe(
    task_id, samples, observation_directory,
    expected_package_binding=program_binding,
    selection=selection,
    expected_selection_sha256=digest(selection),
    input_dtypes=input_dtypes,
    trust_local_code=True,
)
~~~

node、op、target 从 inventory 中选择，图哈希必须匹配。point=output 时 argument=None；输入支持位置索引和显式关键字。describe 返回图信息，observe 返回统计。两步均在原有隔离进程内执行，由宿主显式安排。

[校准数据集汇总](agent-calibration-dataset-r1629.md) 同时接收模块与 FX 回执。数据、选择器、源代码、实现及证据身份一致时，可复用已完成片。模块 hook 的旧接口保持。

原始 forward 与 FX 结果逐项核对，dtype 和 shape 也须一致。检查注册状态、随机状态、图及评估参数不变。域外值记录为 outside；统计不会自动修改近似配置。

当前使用 symbolic FX。数据相关分支等无法跟踪的源码返回 calibration_fx_trace_rejected；图或目标漂移返回 calibration_fx_graph_changed。这些状态描述校准能力，不能直接推断 DSL 是否可表达。更多静态写法仍需扩展。

## 验收

| 范围 | 结果 |
|---|---|
| 当前单元／协议／兼容检查 | 71 passed，0 failed／error／skip |
| GELU 函数输入 | 3 个独立上下文 |
| RMS 分母实际操作数 | 3 个独立上下文 |
| 倒平方根方法输出 | 3 个独立上下文 |
| 历史真实 Agent 任务绑定 | dot、矩阵乘、标量中间值，3/3 |
| 主批原生观察 | 12/12，27 个 worker |
| 协议及 dtype 补验 | 两轮各 2 个 worker，全部通过 |
| 本阶段原生 worker 总数 | 31 |
| 新 API／编译／密态调用 | 0／0／0 |

主批包含 78 组计划观察输入；inventory 和补验重复运行不增加独立样本数。九个独立案例只验证观察器和 ABI，不计为新 Agent 模型通过。三份真实 Agent 任务保持原候选、程序包及等级，验证 16＋1 分片和首片复用。

27 份原生记录通过当前协议复核，保留原生成源码与绑定。新校准代码使用新身份，旧回执不会被改写成当前通过；生成／编译执行 SDK 929e8131…保持。

初始图编码拒绝了 PyTorch 的 immutable dict/list 容器，产生 20 个控制 error。修正结构编码后通过，原日志保留。协议额外拒绝原始输入、答案、凭据或环境字段；forward 比较增加 dtype／shape 检查。

安装、下载、后端、参数、网络、reference 和数值门限保持；提交／推送为 0。[机器报告](agent-fx-calibration-r1633.json)。

## 后续

将原始 FX 观察点与注册近似的 source location 对齐，在生成前冻结区间和参数。原生数值 reference worker 当前按平铺位置参数调用；结构化 call 的兼容范围需要专门验收。原模型精度、容量、算子组合、Bootstrap、CUDA 和完整 ResNet20 继续分别推进。
