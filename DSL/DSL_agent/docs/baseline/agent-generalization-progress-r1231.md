# Agent 通用化进展（r1231）

Ubuntu 的有效连接已确认，开发仍在 /home/lhohy/Code/Poseidon 的 lhy-agent-dsl，HEAD 为 7588e48093631eaf8385506e3db058e7ca3cfe7e。

## 已确认

原批准的76项真实Agent测试全部关闭：56通过、20失败，未运行／跳过／运行中均为0。累计145次生成、145次HTTP尝试，原额度已结算；本轮的新接口验收调用API为0。56份通过包对应392组输入、2814个输出值，原误差最大约1.021e-6；这是原冻结代码下的证据。

关闭失败复核为12数值、3Provider、3静态、2编译。终态以loop和已结清账本判断，两个历史数值标签实际是Provider终止。历史文件与绑定保持原样。

V12有序张量／别名输入和explicit-v11公开绑定、结构化反馈已经接入。最初25文件的草案检查251通过；实际tracing暴露了Poseidon沙箱缺少可信模块的问题，随后补齐两个模块及程序包身份绑定。当前共27个Agent文件，18个已有文件有对应恢复副本，9个为新增文件。补丁后的11项依赖／隔离／身份测试通过，使用Ubuntu Python3.10.12及锁定NumPy1.25.2；真实后端使用既有Nix Python3.10.14、Torch2.0.1和NumPy1.25.2。

| 当前源码模型 | 4组资格验收 | 3组新输入 | 包重载／缓存复用 | 最大绝对误差 |
|---|---|---|---|---:|
| Linear＋局部视图写入／快照 | 通过 | 通过 | 通过 | 3.026e-9 |
| LayerNorm近似＋原地更新 | 通过 | 通过 | 通过 | 2.191e-7 |
| 公开权重split→Linear | 通过 | 通过 | 通过 | 3.393e-8 |

三例合计21组输入、119个输出值；3次真实编译，6个密态验收阶段。已检查Earth／CKKS IR、RuntimePlan、OperatorSpec、常量bundle、Poseidon CPU执行、逐项误差及85个程序包文件哈希。新输入执行和缓存复用的生成／编译调用为0。每例均由真实隔离worker捕获来源，当前候选属于规则基线；新版本真实Agent生成尚未评测。

固定门限仍为 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。最初三例tracing失败的r1225、修复后的r1229、独立审计r1230及所有原失败均保留。续验沿用原30分钟截止时间和4GiB产物边界。

## Attention 的新证据

最后一轮已经完成真实密态执行。密态输出与候选源码数学解释的最大差异约1.071e-8；候选数学解释与冻结模型reference最大差异为1.0。因此已测输入上的最后一轮属于程序计算语义不一致。前三轮仍保留为编译失败。这个诊断没有新生成、编译或密态执行。

## 关闭失败与方向

| 原任务 | 终态失败层 | 生成次数 |
|---|---|---:|
| pytorch_functional_layer_0 | numerical_comparison | 4 |
| pytorch_functional_layer_1 | numerical_comparison | 4 |
| pytorch_functional_layer_2 | numerical_comparison | 4 |
| pytorch_functional_group_0 | numerical_comparison | 4 |
| pytorch_functional_variance_1 | numerical_comparison | 4 |
| matrix__public_vector_r_1 | provider | 3 |
| instance__functional_input_0 | numerical_comparison | 4 |
| instance__functional_input_1 | static_check | 4 |
| instance__functional_input_2 | provider | 3 |
| instance__module1_input_0 | numerical_comparison | 4 |
| instance__module1_input_1 | compiler | 4 |
| instance__module1_input_2 | static_check | 4 |
| instance__module2_input_1 | provider | 1 |
| instance__module2_input_2 | numerical_comparison | 4 |
| instance__unbatched1_1 | numerical_comparison | 4 |
| instance__unbatched1_2 | numerical_comparison | 4 |
| instance__unbatched2_0 | numerical_comparison | 4 |
| instance__unbatched2_1 | static_check | 4 |
| instance__unbatched2_2 | compiler | 4 |
| attention_decode | numerical_comparison | 4 |

数值失败优先检查常量绑定、归约广播、输出及缓存更新；已定位案例不能靠放宽门限处理。静态失败使用新增错误码和源码位置反馈。Provider失败留待新批准批次重测。编译失败保留原表达及预算诊断，先研究Agent层等价表达，不直接认定编译器bug，也不认定模型不可表达。

## 尚未验证

V12其他输入／修改组合的密态结果、新接口真实Agent生成质量、v11常用调用与有序别名的完整组合仍待补充。float32转换的原模型应用精度另计。算子组合、物理布局容量、bootstrap、CUDA实机、更多整模型和ResNet20继续按原计划推进。

本轮保持648个后端文件及6个reference／门限文件哈希。保留x86配置、全部已有修改和旧证据；新增安装、下载、后端库／参数变更、commit和push均为0。

## 入口与记录

应用仍使用 Application.submit_source、status、run、result、execute；程序包可重复使用。模型清单选择poseidon-trusted-pytorch-v12，资源配置可选择general-tensors-v1，生成指导explicit-v11为显式选项。

独立后端审计：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-joint-post-cutover-audit-r1230/report.json。
下一步准备三例新接口真实Agent范围和预算，再取得单独批准；已关闭的76项不会自动重开。

## 新入口执行准备

三例新入口评测的执行工具已接入原 Application。21 项离线检查全部通过，涵盖授权绑定、调用上限、未知调用结果和独立子进程清理；API、编译、密态执行均为 0。原始检查记录为 agent-v12-live-controls-r1235，首次参考闭包检查失败记录 r1233 保留，修正为分别核验生产闭包及验证闭包。

三例真实 Agent 提案 r1232 的模型、请求、后端和预算绑定保持不变，当前仍待用户批准。生产源码 704 项保持冻结；开发工具在单独的 source_live_acceptance_v1 目录。
