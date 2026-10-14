# 生成前校准与近似配置冻结

新入口使用可信本地源码、公开权重和静态输入输出签名，在生成 DSL 前检查近似区间。校准阶段无需已有 Agent 任务、计算图或确定性 DSL 答案。

## Python 接口

~~~python
from source_approximation_preparation import inspect_source, observe_source, freeze
from source_calibration_signature import FORMAT
from benchmark_graph import digest

signature = {
    "format": FORMAT,
    "outputs": [{"name": "out", "shape": [2]}],
}
inventory = inspect_source(
    manifest_path, samples, inventory_directory,
    expected_manifest_sha256=manifest_hash,
    source_signature=signature,
    input_dtypes={"x": "float64"},
    output_structure={"kind": "tensor", "name": "out"},
    trust_local_code=True,
)
~~~

根据 inventory 的图和节点选择输入观察点，再调用 observe_source()。selection 使用 [FX 校准格式](agent-fx-calibration-r1633.md)，source_signature 和原清单保持一致。

~~~python
frozen = freeze(
    manifest_path, freeze_directory,
    expected_manifest_sha256=manifest_hash,
    source_signature=signature,
    calibration_evidence=[{
        "directory": str(observation_directory),
        "report_binding": observation_binding,
    }],
    configuration=explicit_approximation_configuration,
    trust_local_code=True,
)
configured_manifest = frozen["configured_manifest"]
~~~

configuration 沿用现有 poseidon-pytorch-approximations-v1；函数节点与 defaults 选择遵循原 Resolver。调用者提供方法、区间及系数或迭代次数。接口检查区间包含所有已声明校准观察，域外值使冻结失败，不自动拟合或扩大区间。

输出目录保存原源码和公开权重，以及仅改变 approximations 字段的配置清单。它可直接交给现有可信 PyTorch 接入；原始 source location、配置和证据绑定单独保留。

当前冻结范围为已登记的直接函数／Tensor 方法输入。模块路径配置与复合 helper 的内部位置映射返回明确的 unsupported；这部分仍需扩展。校准数据的角色由调用者声明，完整模型精度与 CKKS 数值验收分别执行。

## 验收

| 项目 | 结果 |
|---|---|
| 控制及兼容检查 | 105 passed，0 failed／error／skip |
| GELU、Sigmoid、rsqrt | 各 3 个上下文，9/9 |
| 原始源码校准 | 18 个隔离 worker |
| 配置后的可信前端接入 | 9 个 worker，独立有限参考通过 |
| 实际 source location 与 profile 对应 | 9/9，复核时新执行为 0 |
| 新 API／编译／密态调用 | 0／0／0 |

每例先用 3 组输入采集统计，再冻结调用者指定的配置并接入真实 PyTorch 前端。已有源码和权重逐字节保持；实际前端记录的节点、函数及 profile 哈希与冻结记录一致。前端构造的 16 组独立参考检查属于明文接入验证，不计为密态验收。

本批测试系数用于接口回归，原函数精度尚未认证。当前新入口只有显式 Python 接口，CLI／应用准备流程的接线、Poseidon 与真实 Agent 测试仍待完成。

[机器报告](agent-approximation-freeze-r1642.json)。生成／执行 SDK 929e8131…保持，校准实现使用新绑定；旧报告不改写。安装、下载、后端、参数、网络、reference、数值门限、x86 配置、提交及推送保持。

后续补齐模块／复合算子映射和应用精度，同时继续容量、通用算子、Bootstrap、CUDA、整模型与 ResNet20。
