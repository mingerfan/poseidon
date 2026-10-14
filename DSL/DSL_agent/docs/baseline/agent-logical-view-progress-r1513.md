# 静态诊断与 PyTorch 视图接入

更新时间：2026-10-13T00:43:08Z。ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。
当前源码／执行身份：`4ab8e18103ca0eac8e7b5541e02bbcd7644c37a47faac6e801fa0761dc4dbb3f`。

## 本次改动

候选静态检查现在区分公开控制流、非法语法和 helper 名称冲突，返回源码行列与受控修复信息。接受范围保持原样。
25 项专门检查通过；相关回归 81 项通过、3 项跳过。跳过项为单元测试中未执行的 frontend golden。

新增显式 PyTorch 来源版本 `poseidon-trusted-pytorch-v35`。它继承 v34，在捕获后的逻辑图中识别完整轴置换；
逐坐标核验后，把切片／拼接序列还原为 `permute`，并保留原图、分解身份、坐标证明及独立参考。
单个视图的证明范围为 4,096 元素，总证明工作量为 262,144 单元。
默认来源 v18、latest v34，以及旧 schema／请求哈希规则保持；12 份历史请求按原字节重建。

## 已完成验收

| 范围 | 结果 |
|---|---|
| 新入口及相关组件检查 | 222 passed，0 failed，0 skipped |
| 三个真实隔离源码捕获 | 3/3 接收，48 组／18,880 值，数学与 Torch 门限 1e-12／1e-12 |
| 逻辑图节点 | 262→13、265→16、91→18，输入／权重／输出声明保持 |
| 未修改历史 Agent DSL 的诊断回归 | 2/2，2 次编译／4 次真实 Poseidon 密态执行，包重载、缓存、新输入与重复执行通过；API 0 |
| v35 规则密态基线 | 1/3 通过；2 项失败；真实 Agent 生成另计 |

三个 v35 上下文为 `rank_axis_8`、`rank_linear_16`、`rank_permute_32`。
它们来自已有模型，只增加来源／准备上下文，不增加 Benchmark 的独立模型数。
规则通过项完成四组输入逐项比较、真实 RuntimePlan／OperatorSpec／bundle 检查及程序包复用。

规则失败保留：
- `rank_axis_8`：规则发射器达到 AST 节点上限，尚未进入真实编译。
- `rank_linear_16`：生成的规则表达累计 scale 750，超过固定编译预算 720。已完成 tracing，编译失败，密态执行 0。

上述结果没有证明所有等价 DSL 表达都不可行。编译器、Runtime、Poseidon、参数和 reference 保持。

## 真实 Agent 批次

此前 r1504 的 13 项已经结算：8 passed、5 failed，26 次生成／26 次 HTTP 全部结算。
8 项数值通过的独立审计覆盖 40 组／8,288 值，最大绝对误差 7.404688832923778e-07。
五项失败分别是四项 Provider `stream_limit`，及一项构造／数值修复预算耗尽。
收到的 Provider 诊断不足以区分总流大小与单行大小；本轮保留原接收限制和网络设置。

当前 r1512 用 v35 请求重测上述三个模型。快照时 2 次生成／2 次 HTTP 已预约，
第一项正在生成，原批次尚未结算。当前进程身份与冻结源码已核对。
范围：最多 12 次生成／48 次 HTTP，DeepSeek flash/high/stream，单次 1,200 秒／384,000 tokens，
每例最多 3 轮修复和 3 次传输重试，整批 12 小时、每例 4 小时；API 与密态串行。
批次结束后自动运行一次只读混合结果审计；新增 API／编译／密态调用均为 0。

[离线冻结范围](agent-logical-view-offline-scope-r1511.json) · [真实生成冻结范围](agent-logical-view-live-scope-r1512.json)。
真实调用结果以该批次终态、原始响应、调用账本及独立审计为准。

## 剩余工作

完整大来源记录的应用接线仍待完成：`wide_tail_linear` 与 `weights_linear_10k` 超出既有 4 MiB 来源存储；
历史图入口通过与来源 DAG 草稿通过均保留各自状态。其他旧失败也未因本次测试自动变成通过。
后续在当前批次和审计结束后处理完整来源存储、剩余表达问题及通用组合验收。
Agent bootstrap 链、CUDA 实机、应用精度和完整 ResNet20 各有独立验收状态。

当前结果目录约 36.78／48 GiB；每例／批 4 GiB，磁盘余量至少 4 GiB。
本轮安装、下载、网络变更、后端库修改、参数变更、commit、push、PR 均为 0。
本地修改及未跟踪文件保留。

下一条只读命令：

```bash
python3 -B -m json.tool /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-logical-view-live-r1512/report.json
```

同时查看对应 task 的 `state.json`，以识别正在执行但尚未写入分片报告的调用。
