# MHA 组合输入接入（r1308）

当前状态：有序计算图接入和独立明文检查通过；Poseidon 编译验收仍受阻。v16 保留为开发草案，现有生产入口保持。

## 输入与实现

`nn.MultiheadAttention.eval()` 作为普通有序图节点接入，可与 Linear、残差和静态多输出组合。
使用显式 `poseidon-trusted-pytorch-v16`；旧接口与默认选择继续沿用。

本批涉及 packed／分别声明的 QKV 权重、self／cross attention、batch-first／sequence-first／rank-2 输入、
Boolean／有限浮点公开 mask、padding mask、bias_k/v、add_zero_attn，以及 `need_weights=False` 的静态 None 输出。
公开 mask 与原始权重均校验 dtype、内容、切片来源和哈希；None 不占用密文输出。
近似配置按原始 FX 节点和模块路径选择。原始模型、规范化图、分解图和来源记录均保留。

代码在 `scripts/baseline/benchmarks/tools/staged_mha_ordered_graph_v1/overlay/`，
通过现有 ordered capture／intake／reference／program 接口验证，未启用旧 MHA 专用 worker 路线。

## 本批结果

| 检查 | 结果 | 范围 |
|---|---:|---|
| 独立数学与组合控制 | 16/16 | 6 个原组合及 3 个单列浅配置 |
| 捕获与来源标准库控制 | 26/26 | 类型、mask、packed 参数来源与篡改拒绝 |
| 组件及实际 Torch 控制 | 40/40 | 输出树、选择器、数据流、状态保护和 provenance |
| 隔离来源捕获及参考 | 9/9 | 144 组输入、1,856 个输出值 |
| 局部表达处理控制 | 30/30 | 18 个二值 mask 控制、12 个既有 SSA 压缩控制 |
| 原规则程序编译 | 0/3 | tracing 通过，编译失败 |
| 周期 16 布局检查 | 0/3 | 2 编译失败，1 规则源码 AST 超限 |
| 局部 mask 处理后编译 | 0/2 | 第三项源码未变化，保留旧失败并未重跑 |

各行是独立验收范围，包含重复上下文，不累计成唯一模型数。冻结 Benchmark 未重新生成。

最终捕获已核验 113 项实现文件和 219 个产物哈希。
近似图与独立分解参考的最大差异为 `6.661338147750939e-16`。
六个原配置共 96 组／1,184 值，原函数与近似计算的最大观测差异为 `2.4733121106734757e-10`。

三项浅配置单列为编译可行性检查，保留原配置；
它们相对原函数的最大观测误差分别为 `0.2845255`、`0.0064200`、`0.7659458`。
这些近似误差不能用于原模型应用精度认证。

## 编译失败

固定 Dacapo 的 Earth 乘法检查使用 consumed level 与 scale，当前预算为 720。
原规则程序的失败值为 735、780、735。
选择现有 period-16 布局没有解除限制。

局部处理只删除连续乘同一已绑定完整周期 0/1 mask：
self 删除 8 次，cross 删除 10 次。
模型、权重、近似配置、请求及编译参数均保持。
修正后仍失败，分别为：

- self：`ci<45*12> × pl<45*12>`，Earth 累计值 765。
- cross：`ci<75*11> × ci<60*11>`，Earth 左操作数累计值 735。
- Boolean/padding：无此重复 mask，旧失败保留。

全局代数调度还会触及 AST 或规则运算上限；单独做既有 SSA 压缩无法解决这三项。
Earth 的 consumed level 与 RuntimePlan／Poseidon 数据模数索引含义不同。
这些日志只定位当前候选失败，尚不能判断所有等价表达都无法编译。

本批实际完成 7 次规则候选 tracing／编译尝试。成功编译、密态执行、解密、数值验收和程序包重载均为 0。
付费 API／HTTP 为 0。

## 已保留记录

初次 18/27 控制结果、后续 27/27、最终 40/40，以及两次捕获的原绑定均保留。
三轮后端结果和原始候选分别保存；局部处理属于规则基线，不记为 Agent 新生成。
历史 GELU 和原 76 项真实 Agent 结果保持原计分。

[机器报告](agent-mha-ordered-intake-results-r1308.json) 列出实际目录、哈希、逐项失败层和绑定。
本阶段绑定：`2cd3cf560ba13d22a1cd83ab1680b689d59240c80a9b0045a008e4876c76e1d2`。

## 后续

先在 Agent 自有布局／规则生成层减少中间 gather、scatter 和公开乘法的连续深度，
控制表达膨胀，再验收受影响的原失败案例。后端库、参数和误差门限保持。

尚待验收：MHA 与 GELU／有序修改的组合、嵌套模块和混合选择器、float32 转换、大布局，
真实编译与密态执行、None／scalar 输出程序包重载，以及新接口真实 Agent 生成。
bootstrap、CUDA 和 ResNet20 保持各自独立状态。

源码、测试和文档均在 ARM Ubuntu checkout；654 个后端／reference／门限文件、685 个生产阶段文件、
x86 配置均核验保持。新增安装、下载、参数变更、提交和推送为 0。
本批执行已结束，源码没有运行中的冻结批次。

下一条只读命令：

```sh
python3 -B scripts/baseline/benchmarks/tools/staged_mha_ordered_graph_v1/inspect_current.py
```
