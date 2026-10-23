# 高层模型生成上下文

源码入口选择 explicit-v21 后，生成请求包含经过宿主核验的原始数学图、近似参数、常量描述符及分解节点映射。展开后的统一图仍决定计算和验收。

## 使用

在应用 submit_source 的 options 中设置：

    generation_guidance: explicit-v21
    public_weight_policy: descriptors-math-roles-v2

现有输入格式、资源配置及数值门限保持。旧请求继续使用原版本和哈希。

上下文由模型 provenance 提取。权重通过描述符绑定，公开标量和多项式系数沿用现有 math_roles；私有测试和参考答案留在验收端。结构和哈希检查用于传输边界，宿主同时核对原图、分解图及来源。上下文上限 64 KiB，公开请求沿用原容量限制。

## 当前验收

| 验收 | 结果 |
|---|---:|
| 组件、传输及旧接口检查 | 101/101；失败、错误、跳过均为 0 |
| 正常 PyTorch 源码入口 | 6/6 |
| 历史真实 Agent 源码重放 | 1/1 |
| 卷积、BatchNorm、池化规则基线 | 3/3 |
| 程序包重载、新输入及重复执行 | 4/4 |
| 本批真实编译 / 密态调用 | 4 / 12 |

四个后端案例共比较 16 组输入、72 个输出值，最大绝对误差 3.318e-7；复用测试为 16 组、72 值，最大绝对误差 2.017e-7。均使用 1e-5 + 1e-4*abs(reference)。

历史源码保持与原 Provider 响应一致。其生成来源和原报告保留，当前编译与执行建立新绑定。三项规则候选另行计分。首批验收脚本误用模型 ID 选择重放来源的三个失败也保留；修正案例选择后仅补验这三个案例。

上述模型含显式近似计算，原函数应用精度尚未认证。

## 真实 Agent 批次

6 项请求已冻结：LayerNorm 的模块基础、模块输入变换、函数 rank-3 三例，以及卷积、BatchNorm、池化三个源码组合。

使用 DeepSeek deepseek-flash/high/stream，单次 1200 秒、384000 tokens；每例最多 3 轮修复及每次 3 次传输重试。全批上限 24 次生成、96 次 HTTP，API / 密态串行，编译 2、链接 1；全批 12 小时，每例 4 小时。

新增产物上限 512 MiB，总量 48 GiB、余量 4 GiB；每例 4 GiB，native 沙箱 4 GiB / CPU150 秒。货币上限未设置，重试可能重复计费。授权来自当前计划内测试的统一批准。批次已启动，当前调用账本为 1 次生成 / 1 次 HTTP；尚未结算。最终结果以批次 report 和调用账本为准。

Poseidon、Runtime、Dacapo、参数、reference、门限和网络均保持。新增依赖、安装、下载、提交和推送均为 0。

[机器可读记录](agent-generation-origin-context-r1680.json) · [冻结范围](agent-generation-origin-live-scope-r1680.json)

## 本批结算

真实 Agent 6/6通过，4首轮、2修复后通过；9生成/9HTTP、9编译尝试/7成功、19密态调用。六份程序包重载与复用通过。[完整结果](agent-generation-origin-live-results-r1681.md)。
