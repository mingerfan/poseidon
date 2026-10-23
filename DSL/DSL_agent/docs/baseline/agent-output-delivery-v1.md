# 结构化输出接口

`TaskOutputAdapter` 为已有应用任务提供输出结构绑定、收据重载和本地执行。任务提交、生成、修复、预算及取消沿用 `Application`。

## 输出形式

| 形式 | 结果 |
|---|---|
| Tensor | 按声明形状返回数值列表 |
| 标量 Tensor | 由单元素输出恢复为一个数值 |
| tuple / list | 保持容器类型及元素顺序 |
| 字符串键 dict | 保持键与结构 |
| 静态 None | 恢复占位值，不增加密文输出 |

宿主在模型的命名输出上选择结构。张量叶子的名称和顺序与模型声明一致。支持 1–4 个张量输出，元素和 rank 上限由任务资源配置决定；结构深度上限 32、节点上限 1024、单容器最多 64 项。

这是已完成任务的交付适配器。当前可信 PyTorch 生产捕获仍使用原输出契约；新静态 `None` 的自动捕获与原地操作来源接线另行验收。

## 使用

在已有的固定 Linux worker 中，把 `scripts/baseline` 加入模块路径。以下示例使用已完成任务，模型输入名为 `x`，形状为 `[2]`，输出名为 `out`。

~~~python
from task_output_adapter import TaskOutputAdapter

outputs = TaskOutputAdapter(app)
tree = {
    "kind": "tuple",
    "items": [
        {"kind": "tensor", "name": "out"},
        {"kind": "none"},
    ],
}
bound = outputs.bind(task_id, tree)
if bound["status"] == "succeeded":
    receipt = bound["receipt"]
    trusted_binding = receipt["binding"]  # 宿主单独保留

    checked = outputs.result(
        task_id, receipt, expected_binding=trusted_binding
    )
    executed = outputs.execute(
        task_id,
        [{"x": [0.125, -0.25]}],
        receipt,
        expected_binding=trusted_binding,
    )
    if executed["status"] == "executed":
        values = executed["result"]["structured_outputs"]
~~~

`receipt` 是 JSON 数据，可由宿主持久保存和重新加载。外部保留的 `trusted_binding` 用于核验；数据中的哈希字段本身不是外部签名。

`bind` 和 `result` 检查任务及程序包身份。`execute` 调用原 `Application.execute`，每次接受 1–16 组输入。重复调用复用已有 DSL 与编译产物。

## 返回与绑定

返回包含 `status` 和原任务 `task`。交付通过后增加 `receipt`；执行通过后增加 `result`，其中 `execution` 保存原执行报告的 JSON 数据，`structured_outputs` 保存恢复后的容器。

任务处于排队、取消、不支持或其他失败状态时，适配器保留该状态。收据、包、任务或输出身份不匹配时返回交付层错误，原任务和程序包保持原状态。

收据绑定任务 ID、来源任务、任务元数据、请求、模型、输出结构、程序包、验收等级及适配器版本。程序包复用后，新任务有自己的收据。

`compiled` 和 `numerical` 保持原含义。原包的四组数值验收与新输入的执行分别记录；新输入的数值比较和应用精度认证由其相应验收流程负责。数组恢复为独立数值快照。

## 当前验收

53 项输出契约测试、24 项任务适配器测试通过；6 类本地输出上下文、96 组明文差分通过。组件测试使用实际任务控制层和注入的程序包/执行夹具。新接口的 Poseidon 密态及真实 Agent 验收待执行。

[阶段结果](agent-output-delivery-acceptance-r1075.md) · [组件接口](application-agent-component-v1.md) · [本地执行](application-agent-component-v1.md#能力预检与执行)
