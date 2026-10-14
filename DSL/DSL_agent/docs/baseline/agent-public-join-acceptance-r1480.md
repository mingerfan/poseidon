# PyTorch v34 拼接接入验收

已接入固定公开量与加密张量的 cat / concat / concatenate / stack，沿用 Poseidon CPU 后端。
来源格式使用 poseidon-trusted-pytorch-v34，生成指导 explicit-v14；默认 v18、latest v28。

| 验收 | 结果 | 实际调用 | 数值范围 |
|---|---|---|---|
| 隔离来源与独立参考 | 21/21 | 编译、密态、API 0 | 336 组 / 2,448 值 |
| 规则候选 | 21/21 | 21 编译 / 27 密态，API 0 | 96 组 / 720 值，最大绝对误差 1.03929549e-06 |
| 当前真实 Agent | 18/18 | 19 生成 / 19 HTTP / 19 编译 / 25 密态 | 84 组 / 672 值，最大绝对误差 2.94874066e-07 |

真实 Agent 首次通过 17 项，经过自动修复的案例：mixed_stack_0。两轨失败、跳过和未运行均为 0。
全部程序包完成重新加载和缓存复用；mixed_cat_2、mixed_stack_2、combination_5 另完成三组新输入及重复执行。
候选逐字来自 Provider 原始响应。规则程序与真实生成分别计分。

## 已确认

- 拼接序列为 1–8 个静态 Tensor；cat 要求非拼接轴一致，stack 要求相同 shape。
- 覆盖公开 dtype 提升、标量 stack、负轴，以及 mask、拆分、公开变换、Linear、残差和结构化输出组合。
- 公开/加密混合拼接通过 Agent 层显式规范化复用现有计算接口，额外密文运算和状态身份参与绑定。
- 验收包含 tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、真实密态执行和逐项比较。
- 误差门限保持 1e-5 + 1e-4*abs(reference)，两轨独立数值审计通过，HTTP 全部结算，实际 owner 退出。
- 历史五项输入拒绝和开发工具索引/准备错误记录保留；模型、reference、后端及网络保持。

## 基于证据的推断

本批注册的静态拼接能够通过既有后端接口执行，现阶段无需修改 Poseidon、Runtime 或 Dacapo。

## 尚未验证

任意 Python、动态 shape、最大 rank/物理容量、原模型应用精度、真实 bootstrap 接线、CUDA 实机和完整 ResNet20。
18 个新来源上下文与 3 个旧兼容上下文不增加冻结 Benchmark 的 1,200 模型结构计数。

## 后续

继续整合常用来源配置、较大权重和物理布局，再按通用算子清单验收未见组合。
测试沿用用户总授权；全局 48 GiB、每例及每批 4 GiB、磁盘余量 4 GiB，API 与密态串行、编译 2 / 链接 1。

SDK：0a58989024185fead91aa91e5b8079e1bacc3eeee591fa58dbff60e5a2f79968。阶段报告绑定：2b5df37707aa951185d1d090b82d41949afaca9ec082a1b9df734bbea2d3af2f。
[机器可读结果](agent-public-join-acceptance-r1480.json)。
