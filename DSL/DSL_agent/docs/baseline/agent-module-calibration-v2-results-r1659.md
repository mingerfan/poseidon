# 模块路径近似与生成前校准

SourcePreparation 新增请求格式 poseidon-source-preparation-request-v2。命令入口保持 scripts/agent_calibration.py，inspect／observe／freeze 与旧版一致。

v2记录原始 FX 节点的模块路径、实际模块类型及相关属性。近似按节点、模块路径、算子默认值依次选择；同优先级冲突、未知路径、未覆盖的域和漂移均拒绝。支持 ReLU、GELU、SiLU、Sigmoid、Tanh 的输入点，以及嵌套模块内已登记函数的来源。Linear、Flatten、Identity、Dropout可保留公开模块信息；其它复合模块的内部观察点继续单列。

## 使用

把准备请求的 format 设为 poseidon-source-preparation-request-v2：

~~~sh
python3 scripts/agent_calibration.py inspect --request /absolute/inspect.json --output /absolute/work/results/inspect --trust-local-code
~~~

observe 采用 inspect 返回的图哈希与节点信息，selection.format 为 poseidon-source-fx-calibration-selection-v2。freeze 请求的 configuration.modules 可使用：

~~~json
{
  "path": "activation",
  "function": "gelu",
  "profile": {
    "method": "polynomial",
    "basis": "power",
    "domain": [-4, 4],
    "coefficients": [0, 0.5, 0.25]
  }
}
~~~

以上是接口示例，系数与区间由调用者显式提供。冻结只改配置清单，原源码和权重保持；生成、编译和数值验收仍使用正常应用入口。旧 v1 请求及报告格式保持。

## 本批结果

| 项目 | 结果 |
|---|---|
| 不重复计数的控制与兼容检查 | 211 passed，当前 failed／error／skip 0 |
| 原始来源校准与捕获 | 9 个上下文通过 |
| 配置选择与前端实际来源 | 逐项完全一致 |
| 正常源码提交的规则程序 | 3/3通过 |
| 真实编译／Poseidon密态调用 | 3／9 |
| 首次数值 | 12组40值，max abs2.899e-9 |
| 新输入与重复执行 | 12组40值，max abs5.968e-9 |
| 包导出、重载、复用 | 3/3 |
| 本功能付费 API | 0 |

9个来源上下文包括GELU模块3项、Sigmoid模块3项、嵌套GELU函数3项；它们保留各自的代码和SDK绑定。实际启动26个校准worker、13个捕获worker（包括保留失败），正常应用再捕获3次。只计为源接入与规则程序验收，v2的新Agent生成尚待单独计分。

来源丢失发生在Agent规范化图的再次tracing。现仅在操作、参数依赖及逻辑输入目标一致时继承原模块来源；图的计算保持。首轮3个嵌套失败、复验1个直接调用失败、单位检查的初始重复收集和夹具错误均保留。成功项未替换模型、profile、reference或数值门限。

LayerNorm／Softmax等复合模块的内部近似点仍为未开放范围。原函数应用精度未认证；数值比较对象为冻结近似计算。

[机器报告](agent-module-calibration-v2-results-r1659.json)。当前SDK316b4fcb…；后端库、参数、网络、x86配置保持。安装、下载、commit、push、PR为0。后续按现有授权冻结代表生成请求，再继续复合近似点、原模型精度、算子、容量、bootstrap、CUDA及整模型目标。
