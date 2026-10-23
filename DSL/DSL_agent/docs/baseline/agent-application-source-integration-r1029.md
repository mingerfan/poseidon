# 统一模型来源组件接入 r1029

已将三类模型来源接入现有应用组件：JSON 图、受限 Python 清单、显式可信的本地 PyTorch 清单。来源导入和 DSL 生成共用任务、取消状态与总预算；旧 submit、submit_import 和 CLI 保留。

## 已确认结果

| 检查 | 结果 | 验收范围 |
|---|---:|---|
| 最终组件、来源、包、缓存、身份及兼容测试 | 185 passed，0 failed/errors/skipped | 包含注入后端与合成产物测试 |
| 当前上下文 worker 的实际隔离导入 | 6/6 | 三类来源，96 组独立参考 |
| 调用方修改文件后的快照保持 | passed | 实际可信 PyTorch 导入 |
| 实际导入进程取消与清理 | passed | 无遗留进程 |
| 当前新编译、密态执行、解密、真实 Agent 生成 | 0 | 尚待后端验收 |
| 付费 API、安装、下载、SDK重建、Git提交/推送 | 0 | 本批未发生 |

导入进程实际核验 AS=4GiB、CPU=150/155秒、FSIZE=16MiB、NOFILE=128、CORE=0。独立参考通过只证明导入和分解；程序包的合成文件测试只证明绑定与序列化。

此前获准的 r870 静态 split 批次已在 [r878](agent-pytorch-split-acceptance-r878.md) 完成：24/24真实编译及密态数值通过，三份包复用通过，maximum absolute error 5.9262181784092149e-7。该历史绑定保持；本次组件变更没有借用其结论。

## 实现

新增 application_model_source.py 和 model_source_provenance.py。现有 Application 增加 submit_source，run 自动进行来源接入，再进入原生成与验证循环。提交冻结列明的源码和 NPY 权重，状态只公开接入阶段及必要摘要。

修改应用后端、上下文适配器、程序包和来源清单。v5/v6 包可包含 artifacts/model-source.json，绑定来源文件、模型、参考和控制层身份；生成来源与后端验收仍分别记录。生产路径使用原组件和 Poseidon 接口，开发审计工具留在 benchmarks/tools。

接入审查补齐了 ContextBackend.export 对来源记录的透传。新增缓存回归复现了预算变化导致重复生成的问题，修复后按语义来源身份复用；每个任务的完整快照和预算绑定继续核验。首次失败日志和修改前工作文件保留。

开发在 ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。Poseidon、Runtime、Dacapo/Hecate、补丁、SDK、参数、reference、门限和x86配置保持；已有修改均保留。git diff --check 通过。

## 当前绑定与后续

来源接口当前实际报告：6cef6deff7d70a758c55d262d8a7924ff0305c7d48f95870dad372911bc58162。完整日志、任务快照、测试和原失败记录位于 ARM work root 的 agent-model-source-integration-r1012。

新的 [r1025 离线方案](agent-application-source-offline-proposal-r1025.md) 保留原112项，加六个来源接口上下文，共118项及18份包复用检查。690份运行源码、5份上下文源码、3303份案例/证据文件和4份驱动已核验；未批准启动在进入Nix前拒绝。当前没有批准收据或后端目录。

新提案绑定 f958c0074c4554f6254b29a68064edc285db51fefd1166abe3aa2704b789386d。旧 r991 已因生产源码变化过期；初始 r1021 的未批准启动草稿修正了路径定位，原文件和绑定保留。原模型、权重、reference、请求和规则候选未改写。

## 尚未验证

当前来源组件的真实编译、Poseidon数值验收、包重载和新输入复用待执行。新入口的真实 Agent 生成另需冻结和付费授权。

八项原编译失败保留：attention_core_0–2、channel_broadcast_mix_0–2、norm_depth_branches_1/2。原日志定位为 Earth 累计scale容量检查；尚没有新编译结果，也不据此判定数学模型不可表达。

更大物理布局、更多PyTorch写法与未见组合、原函数应用精度、密文子图交接、bootstrap、ResNet20、CUDA、x86原生和空白机器安装继续分别验收。本次没有完成整个通用化计划。

[组件接口](application-agent-component-v1.md#模型来源提交) · [Agent README](../../scripts/README.md) · [机器记录](agent-application-source-integration-r1029.json)
