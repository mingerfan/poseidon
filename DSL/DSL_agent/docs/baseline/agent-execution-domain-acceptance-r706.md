# Agent 输入区间与 split 验收

2026-10-04 · r706 · ARM Ubuntu · `lhy-agent-dsl`

程序包执行接口已增加新输入的近似区间预检，Agent 自有规则 lowering 已补齐多输出 `split`。本批五个规则候选通过 Poseidon CPU 数值比较，一项 Attention 规则候选仍在编译层失败。新增真实 Agent 生成与付费调用为 0。

## 实现

`app.execute(task_id, samples)` 和 `dsl_bundle.py execute` 在加密前检查来源绑定的原图。输入先按原签名验证并复制，再以既定近似语义检查声明区间。成功返回 `input_domain_check`；越界返回 `approximation_domain_rejected`，包含首个违规节点、样本序号和公开区间。诊断不含输入值、reference、路径或凭据。

检查区间的状态为 `passed` 或 `not_declared`，范围仅限原图声明的近似区间。新输入的 CKKS 数值比较仍由调用方或验收工具完成，应用精度独立记录。CLI 越界返回同一诊断并以 1 退出，原已完成任务和程序包仍可复用。

`split` lowering 使用已有旋转和公开掩码。单个密文内的连续逻辑片段可直接作为视图；跨块或非连续片段逐项路由。视图的尾槽可含其他数据，归约沿用绑定的尾部掩码。未增加参数、公共常量、源程序或工作量预算。

修改仅涉及 Agent 执行适配器、规则 lowering、测试、开发工具和文档。Poseidon、CKKS Runtime、Dacapo/Hecate、reference、模型及安全参数保持原样。

## 验收

| 范围 | 结果 |
|---|---|
| 静态、组件与兼容测试 | 172 个去重测试最终通过，0 failed/errors/skipped |
| 输入区间与程序包执行 | LayerNorm、GroupNorm 通过；Attention 1 项编译失败 |
| split 真实后端 | 3/3 通过：`[9]` 的三段跨块拆分、`[2,3]` 的内轴拆分、`[1,2,3,2]` 的负轴拆分 |
| 程序包复用 | 两份归一化程序包重载、三个新输入各执行两次、缓存复用通过 |
| 超域拒绝 | 两项应用拒绝与两项 CLI 拒绝通过，均在创建执行产物与加密前结束 |
| 数值比较 | 40 组执行、248 个输出值，最大绝对误差 `3.189088633170911e-7` |
| 付费/API/HTTP | 0 |

40 组执行包含重载和重复执行。五个通过模型均使用规则候选，不计为当前 Agent 新生成成功。数值条件保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

原 Attention 的 16 组双参考与源码数学解释检查通过，最大 lowering 差异约 `1.39e-17`。该明文结果独立于密态执行状态。

## 保留失败

- r697：执行返回遗漏 `input_domain_check`，导致验收工具中断；已修正并加入成功路径测试，原报告保留。
- r698：容量规则基线缺少多输出 `split`；已补齐，原模型、权重和请求哈希保持。
- r700/r703：Attention 的真实 tracing 通过，Earth 编译累计 scale 分别为 750、735，均超过固定预算 720。当前原请求为 `f93f5c6e82faf177e05c66ffd6a3367347531332cae4172167fcb18f4a7b2c6d`，本次表达仍未编译、执行或解密。该失败不证明所有等价表达都不可编译。
- r702：连续 1024 元素 split 成为合法视图后，旧“应超预算”测试预期失效；原实例保留为正例，另用内轴非连续拆分检查原工作量上限。初始失败记录保留。
- 两次开发探针导入错误保留；修正后才启动对应检查，没有发生 API、编译或密态执行。

Attention 后续先检查 Agent 层的合法表达、布局与尾槽事实，保留固定参数和门限；后端改动另行审查。

## 绑定与资源

LayerNorm/GroupNorm 的执行绑定为 `cc21e006afe37a220721bcab9fee7000f8ce57696f9d4286821616547daea217`；当前 Attention 复验绑定为 `6365448a2d0d7df15d6441030335effecd49f6dc301f8cbdf4a2d6ebc825fca2`；split 执行绑定为 `751771a55711f456d3985db54390ee280463575bbdf21589d056dfd8477c7e5d`。各报告按自己的源码版本保留，没有合并或改写历史绑定。

636 个当前冻结 Agent 文件、465 个受保护后端文件及 5 个 reference/门限模块核验通过；107 个程序包文件哈希一致。累计记录执行约 2945.97 秒，低于 3600 秒；产物约 36.15 MiB，全局约 22.36 GiB，剩余磁盘约 46.41 GiB。

分支和 HEAD 保持 `lhy-agent-dsl` / `7588e48093631eaf8385506e3db058e7ca3cfe7e`。结束检查未发现本批运行进程。新增安装、下载、SDK 重建、后端修改、参数变更、commit、push、PR 均为 0。

## 后续

继续静态 shape 公开循环、直接标量输出、逐节点/模块近似配置、权重去重及组件依赖边界。新功能的真实 Agent 评测需要新的冻结范围和单独授权；r620 已结算，r675 为未执行历史绑定。Bootstrap、CUDA、x86 原生、空白安装和完整 ResNet20 仍单独待验收。

原 24 项 PyTorch 写法续验已在 [r695](agent-pytorch-spelling-acceptance-r695.md) 完成。当前报告为 [JSON](agent-execution-domain-acceptance-r706.json)，完整原始产物位于 Ubuntu `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/` 的 `agent-domain-*` 与 `agent-capacity-split-offline-r705`。
