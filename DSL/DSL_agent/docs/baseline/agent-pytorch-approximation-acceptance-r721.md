# PyTorch 逐节点与模块近似验收

2026-10-05，ARM Ubuntu。新增 `poseidon-trusted-pytorch-v5`；执行后端为 Poseidon CPU。

近似选择采用节点 → 模块 → 算子默认值，记录完整配置、FX 来源和所选参数。共享模块的多次调用、函数式激活和 Softmax 的近似组件使用相同入口。旧 v1–v4、JSON、CLI 和组件接口保留。

## 验收结果

| 案例 | 规则候选 | 程序包复用 | 来源 |
|---|---|---|---|
| modules：Sequential ReLU→LeakyReLU | 通过 | 未安排 | 模块选择 |
| nodes：共享 ReLU 两次调用与函数调用 | 通过 | 重载、3 新输入×2、缓存复用 | 节点／模块／默认值 |
| nested：两个子模块与公开形状循环 | 通过 | 未安排 | 函数所属模块 |
| functions：GELU＋SiLU 与 Sigmoid×Tanh | 通过 | 重载、3 新输入×2、缓存复用 | 算子默认值 |
| norms：两个 LayerNorm，分别一／两次 rsqrt 迭代 | 编译失败 | 未完成 | 两个模块使用不同迭代参数 |
| compound：函数式与模块 Softmax | 通过 | 未安排 | 节点／模块分别覆盖 exp／reciprocal |
| functions_nodes：GELU 节点覆盖变体 | 通过 | 重载、3 新输入×2、缓存复用 | 第三个节点选择上下文 |

`functions_nodes` 是原模型的配置构造变体；6 个来源模型、7 个规则任务，6 passed／1 failed／0 skipped／0 not_run。

- 最终联合测试 226 passed，0 failed/errors/skipped。
- 7 个来源任务各完成 16 组原模块、独立数学解释器及 PyTorch reference 检查，共 112 组。公开 v5 CLI 另复核同一 nodes 模型的 16 组参考，模型哈希一致。
- 通过项完成真实 Hecate tracing、Earth／CKKS 编译、RuntimePlan／OperatorSpec／bundle 检查、Poseidon CPU 密态执行、解密和逐项比较。
- 规则候选含包重载和新输入执行共 54 组／164 值，最大绝对误差 1.4210701070743736e-9。
- 三份程序包完成重载与缓存复用；独立检查 60 个结构化新输出值、157 个程序包文件哈希。新输入执行的生成／编译调用为 0。
- 三种选择层级均有至少三个已执行上下文：默认值 4、模块 4、节点 3。

真实后端范围以表中六个来源模型及其构造变体为准。其他新增函数／Tensor 调用已完成捕获和独立参考测试，逐项后端及真实 Agent 验收继续保留为后续任务。

## 原 LayerNorm 失败与等价表达

原规则程序保留在 r712。固定编译配置推导的累积 scale 为 750，预算为 720，失败发生在 Earth 乘法类型推导；尚未进入密态执行。这是该表达的失败证据。

检查了五个人工等价表达。仅替换公开初值、单独调整初值乘法、单独调整两元素布局、单独调整第二步 Newton 均未解决。两元素中心化布局与第二步 Newton 乘法重排组合通过：4 组／8 值，最大绝对误差 1.370750310852153e-7。同一模型、公开权重、请求、近似迭代数及 reference 保持原样；256 组独立代数差分最大误差 2.220446049250313e-16。

这份结果单列为人工等价表达，原规则失败仍保留。后续在 Agent 自有 lowering 中研究通用的中心化归约和多项式乘法计划，并跨形状验证；当前未把人工源码插入应用工作流程。

## 绑定与资源

主批绑定 `d54bac112cb1601f61160e9b25187a21499fb05c3365c8cdd6e9447b79566fe5`；补验绑定 `8f0a825b7e5552c48e229214df3d00cce4e44cc0405d0b81421ab2beb751b7eb`。两份原计划、报告和候选保持原文件。

648 个后端文件及 6 个 reference／门限／产物检查模块与开工快照一致。当前 644 个 Agent 文件与补验冻结绑定一致；主批之后调整的两个测试文件保留原快照。实际执行仍采用 tc128、N=32768 和原参数，执行器仅持评估密钥。

本批 API／HTTP、安装、下载、SDK 重建、后端／参数／reference／门限变更、commit／push／PR 均为 0。数值门限为 `1e-5 + 1e-4*abs(reference)`。原函数近似误差与 CKKS 执行误差分开记录；本批配置用于接口回归，原模型应用精度尚未认证。

执行和测试累计约 2055.41 秒，新增产物约 43.74 MiB，全局约 22.43 GiB，余量约 46.33 GiB。编译≤2／链接≤1／密态串行，既有隔离与候选资源限制保持。

早期 FX 元数据格式、近似域及测试参数错误保留在 preflight；补验初次测试误读返回字典层级，在运行密态前停止，修正测试后 226 项通过。每次失败、诊断、原源码和绑定均保留。

## 使用与后续

[配置和示例](operator-decomposition-and-python.md#逐节点与模块近似) · [机器报告](agent-pytorch-approximation-acceptance-r721.json)。

后续：新增输入能力的真实 Agent 评测；原 LayerNorm 规则表达；更多静态索引写法；源码身份检查开销与组件依赖边界；权重去重。原 Attention 编译容量失败、bootstrap、密文分区、CUDA、x86 原生／空白安装及 ResNet20 分别保留。

原始产物：

~~~text
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approximation-preflight-r711
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approximation-offline-r712
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approximation-supplement-r720
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/candidate-replay-xzsmuvs2
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approximation-public-cli-r721
~~~

## 人工等价表达记录

以下代码属于单列的人工验证；公开常量绑定来自未修改的 norms 请求。SHA-256：`6dfdeba746a0c8077055de2ca253cff214dd75e8f8e9e09f2f1597ffe8afe147`。

~~~python
@hc.func('c')
def cap_period_sum_4(a0):
    v1 = a0 + a0.rotate(1)
    return v1 + v1.rotate(2)

@hc.func('c,c')
def golden(x0, zero_ct):
    delta = x0 - x0.rotate(1)
    a = delta * p4 * pv0
    va = cap_period_sum_4(a * a) + p6
    ga = (va * p7 * p7 * p2 + p9) * p7
    ya = a * ga
    b = delta * p5 * pv0
    vb = cap_period_sum_4(b * b) + p5
    gb = (vb * p7 * p7 * p2 + p9) * p7
    gb = gb * p9 + (gb * gb) * (gb * (vb * p2))
    yb = b * gb
    y = ya + yb
    return [y - y.rotate(1).rotate(2)]
~~~
