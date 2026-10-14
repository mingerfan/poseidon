# 通用输入与坐标指导进展

旧 explicit-v16 批次已结算：9 项真实 Agent 测试，8 项通过、1 项数值失败。13 次生成和 13 次 HTTP 全部结算；13 次编译、19 次 Poseidon CPU 密态调用。通过范围共 44 组输入、6,444 个输出值，最大绝对误差 4.19456060996e-8。三份包的新输入、重复执行和缓存通过。

| 工作 | 结果 |
|---|---|
| 当前组件与旧接口检查 | 105 passed，0 failed，0 skipped |
| explicit-v17 离线回归 | 2 份原规则程序、1 份原人工 public 程序通过 |
| 当前真实编译与密态执行 | 3 次编译、9 次密态调用；3 包重载、新输入、重复和缓存通过 |
| 独立数值审计 | 24 组输入、2,456 个值；最大绝对误差 1.57796300027e-6 |
| 当前 v36 来源捕获 | 原源码与原权重，3/3 隔离捕获及双参考通过；48 组、4,000 个值 |
| 新真实 Agent 批次 | 4 项已启动；结果按终态审计结算 |

## 修改

Agent 的 explicit-v17 为已登记的 BSGS 路由给出明确坐标关系：先在 baby 坐标中乘预旋转向量，再对乘积做 giant 旋转。原 native/public 契约、输出绑定、模型、权重和布局保持，静态、验证适配器和应用反馈共用指导。旧版本请求重建检查通过。

旧数值失败的最后候选，公开符号检查显示广播 216 个位置全部正确，置换有 105 个位置的系数不一致。检查只使用公开模型、权重与源码；这个诊断不能代替当前 Agent 新生成验收。

旧 PyTorch 捕获记录在当前实现身份下被拒绝。原记录保留，已重新隔离捕获同一批源码和 NPY 权重，并建立当前来源绑定；未绕过完整性检查。

## 正在执行

原 rank_axis_8 失败案例，以及完整 v36 来源入口的 wide_tail_linear、weights_linear_10k、rank_permute_32。三个来源案例分别覆盖 2,052 个输入元素、10,100 个非重复公开权重、rank 32 置换与命名多输出。

使用同一生成模型，最多 16 次生成／64 次 HTTP，12 小时／每例 4 小时；API 与密态串行，编译≤2、链接≤1，新增产物≤4 GiB、总量≤48 GiB、磁盘余量≥4 GiB。既有 native 限制保持。按计划内总测试授权执行，终态后自动做一次只读审计。

截至本报告快照，原 rank_axis_8 的新尝试发生 Provider 失败，未进入编译或密态执行；其余案例继续。网络原因尚未确定，网络配置保持。

## 使用与后续

开发与执行在 ARM Ubuntu `/home/lhohy/Code/Poseidon`，当前分支 `lhy-agent-dsl`。当前 SDK 是 `1a825a4e0d45ea7f11fd2f2a5cf760c229b8f159e6930ac2761cb11f65538297`；新指导显式选择 `generation_guidance=explicit-v17`，旧默认值保持。

本阶段没有修改 Poseidon、CKKS Runtime、Dacapo、参数或数值门限，没有安装下载或提交推送。x86 配置保持，当前真实执行证据来自 ARM CPU。

下一条只读命令：

~~~sh
python3 -B -m json.tool /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v36-complete-source-live-r1558/report.json
~~~

源码冻结到主进程与后置审计均结束且 HTTP 全部结算。后续继续容量与算子组合、原函数应用精度、Agent bootstrap、CUDA 实机和完整 ResNet20；本次通过范围分别计分。


## 传输限制诊断

首例的新尝试在 HTTP 200 的响应体阶段返回 `stream_limit`。固定接收器限制流响应总量 32 MiB、单行 1 MiB；当前收据未区分触及哪一项。已接收 107,367 个 SSE 事件，未取得完整候选，编译和密态调用为 0。调用已结算为 failed／non-retryable。此记录指向接收上限，不能归因为网络中断；没有修改网络或接收限制。[机器记录](agent-stream-limit-diagnostic-r1560.json)。


## 完整来源阶段进展

weights_linear_10k 首轮真实 Agent 通过：可信本地 PyTorch 工厂与 10,100 个非重复公开权重，1 次生成／1 次 HTTP、1 次编译／3 次密态调用。首轮 4 组／400 值，另 3 组新输入和 1 组重复输入，共 8 组／800 值，最大绝对误差 1.73005922144e-7；包重载和缓存复用通过。

当前批次已完成 1 passed／2 failed，最后的 rank_permute_32 仍在生成。两个失败分别是原图的 stream_limit、wide_tail_linear 的数值比较；后者 4 次生成中 2 次通过真实编译和密态执行，数值失败保留。整批独立收尾审计尚未完成。[快照](agent-v36-live-progress-r1561.json)。
