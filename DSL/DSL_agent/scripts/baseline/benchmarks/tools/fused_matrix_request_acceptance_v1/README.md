# 融合矩阵生成请求验收

复用 V14 已保存的可信来源包，验证从来源到生成上下文的接入：

**来源与权重绑定 → 统一模型 → 布局计划 → 公开权重描述符 → 生成请求。**

检查九个 addmm、baddbmm、addbmm 模型，覆盖密文矩阵、公开右矩阵和公开偏置，
每例有三个逻辑输入、三个命名输出。通过现有组件准备接口生成请求，
并检查确定性重建、常量内容绑定、隐私字段、输入输出布局及能力预检状态。

## 运行

工具使用当前 ARM Ubuntu 的已配置 SDK 和 r1268 保留来源证据。
每轮选择新结果编号：

```sh
python3 -B scripts/baseline/benchmarks/tools/fused_matrix_request_acceptance_v1/run.py agent-fused-matrix-requests-rNEW
```

仅补验生成上下文隐私检查时，复用已准备的请求：

```sh
python3 -B scripts/baseline/benchmarks/tools/fused_matrix_request_acceptance_v1/run.py agent-fused-matrix-requests-rNEW --prepared agent-fused-matrix-requests-r1274
```

将 NEW 替换为尚未使用的数字。结果保存在现有 ARM work root。
执行采用独立网络命名空间、只读源码与依赖、固定资源限制，
保留源码快照、请求、布局、生成上下文和原始报告。

## 结果范围

验收层级为“生成请求准备通过”。布局是静态输入输出计划，
中间密文、深度、内存和耗时维持 unknown，编译、密态执行、
编译程序包及真实 Agent 生成单列验收。
生产入口继续使用现有契约。

`audit.py` 将原始失败与定向补验合并，保留原始状态。
`acceptance.json` 记录当前验收绑定和具体证据。
