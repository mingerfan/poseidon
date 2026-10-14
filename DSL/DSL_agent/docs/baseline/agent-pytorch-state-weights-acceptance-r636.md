# PyTorch 外部权重接入验收

ARM Ubuntu 的六类外部权重模型通过 Poseidon CPU 编译、密态数值比较和程序包重载。新增清单 `poseidon-trusted-pytorch-v3`；模型工厂定义结构，完整 `state_dict` 由经过名称、shape、dtype、大小和 SHA-256 检查的 NPY 文件加载。

本批使用规则候选，付费调用为 0。此前批准的 14 项真实 Agent 测试已在 [r626](agent-generalization-live-acceptance-r626.md) 完成；本批的 v3 入口尚待新的真实 Agent 验收。

## 结果

| 项目 | 结果 |
|---|---|
| 静态与组件测试 | 173 passed，0 failed/error/skipped |
| 独立明文参考 | 6 类模型，每类 16 组；原模块、数学解释器及 Torch 参考一致 |
| 编译和密态验收 | 6/6，24 组、136 个输出值，最大绝对误差 1.76722e-8 |
| 程序包重载 | 6/6，24 组、136 个输出值，最大绝对误差 4.63155e-8 |
| 新输入重复执行 | 3 类模型，9 组新输入各执行两次，132 个输出值，最大绝对误差 7.03548e-9 |
| 用户 CLI | v3 导入及参考检查通过 |
| 付费 API | 0 |

六类模型为平方激活 MLP、Conv→AvgPool、残差 Linear、BatchNorm 多输出、带公开增益的矩阵融合和形状变换→Linear。外部随机权重与构造器初始值不同；BatchNorm 保留整数统计计数器。MLP、BatchNorm 和矩阵融合另做新输入重复执行。

全部数值验收采用 `1e-5 + 1e-4*abs(reference)`。执行器仅获得评估密钥；验收端完成加密和最终解密。重载及新输入执行均为 0 次生成、0 次编译。

## 接口

- `pytorch_weight_package.export_state(module, directory)` 将 CPU 模块的完整状态写为确定性 NPY 数据及声明。
- v3 清单新增 `weights`，以原始 state_dict 键登记文件、dtype、shape、字节数和哈希。
- 允许 float32/float64 权重及 int32/int64/bool 公共状态；载入时必须与工厂模块的 dtype 一致。
- 浮点计算按显式 dtype policy 建立 float64 参考。原生 float32 应用精度仍为未认证。
- 状态文件在隔离捕获前验证，以只读快照加载；捕获后复核并保存来源。图常量与实际加载状态的对应关系通过内容哈希检查。
- 生成请求只获得公开模型及选定权重描述符；来源源码和捕获记录留在可信侧。
- 程序包沿用现有 v5/v6 接口，保留 Tensor、tuple/list、字符串键 dict 的输出结构。

v1/v2 清单、旧 JSON、请求哈希和 CLI 参数保持兼容。依赖身份变化产生新绑定；旧来源报告需要按当前捕获实现重新核验，历史文件保持原样。

## 原记录与复用

r629 的两项错误来自测试脚本使用了错误返回键和不支持的 options 字段；r630 修正后 79 项通过。r633 另有 94 项组件及程序包测试通过。

r631 完成六项真实编译和数值验收，但开发工具缺少离线 Provider 的 `agent_calls` 字段，六项应用任务在收集结果时返回环境错误。r632 核验原请求、候选源码、代码和依赖身份一致，复用这六份资格检查证据，再完成全部导出、重载和新输入验收。原失败报告、验证结果和日志全部保留。

原始证据位于 ARM work root：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-state-weights-numerical-r631` 和 `agent-state-weights-numerical-r632`。机器可读报告记录请求、源码、程序包及旧证据绑定，仓库报告保留汇总指标。

## 范围与后续

本批共六个小模型，覆盖 float32/float64 权重、整数缓冲区和结构化输出。int32/bool 状态的解析与拒绝边界有静态检查，尚未扩大为对应的密态算术支持。大权重、布局候选、原生 float32 应用精度、bootstrap、CUDA 与完整 ResNet20 保持原状态。

开发仍在 `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。582 个后端文件和三个 reference/数值模块核验未改；未安装、下载、调整参数、提交或推送。

[使用说明](operator-decomposition-and-python.md#pytorch-外部权重) · [机器可读报告](agent-pytorch-state-weights-acceptance-r636.json)
