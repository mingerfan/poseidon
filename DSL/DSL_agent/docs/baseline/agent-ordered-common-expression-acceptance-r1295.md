# 布尔掩码 Attention 离线验收

2026-10-08，ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

## 结果

原布尔掩码 Attention 图采用新的等价规则表达后，通过 Hecate tracing、真实编译、
RuntimePlan／OperatorSpec／常量 bundle 检查、Poseidon CPU 密态执行和逐项数值比较。
一次编译、一次密态验收；四组固定输入、48 个输出值全部通过，失败和跳过均为 0。
这份候选来自规则 lowering，`agent_generated=false`。

| 指标 | 数值 |
|---|---:|
| MAE | 2.2827495781526828e-9 |
| 最大绝对误差 | 5.07544949818195e-8 |
| 非零 reference 最大相对误差 | 6.029800792545785e-7 |
| Cosine similarity | 0.9999999999999999 |
| 数值门限 | `1e-5 + 1e-4 * abs(reference)` |
| 整批耗时 | 206.13 秒 |
| 新增产物 | 约 5.25 MiB |

独立 reference 已重新计算；51 个原始证据文件和 30 个程序包文件哈希匹配。
RuntimePlan 包含 241 条指令，涵盖密文／明文乘法、rotation、加法、modswitch、
rescale 和 relinearization。3 个逻辑加密输入，另有固定 zero 参数，3 个输出密文。
参数沿用 Poseidon w45：N=32768、Q13/P1、tc128，本例没有 bootstrap。

## 原失败与新结果

原候选在 Dacapo 累计 scale 735>720 处编译失败；原源码、模型、公开权重、
reference、失败报告和旧绑定保持。新候选只调整 DSL 表达，按当前后端建立独立绑定。

这次结果说明该案例存在可在现有后端执行的等价规则表达。原失败候选仍记为失败，
新候选记为人工选择的规则变体。历史 76 项真实 Agent 的 56 通过／20 失败计数保持。
本次比较对象是已冻结的近似计算 reference；原始 Softmax 应用精度另行验收。

## 程序包

程序包采用 `poseidon-dsl-bundle-v6`，验收等级 `numerical`，已导出并完成只读重载检查。
绑定：`95de09d72bc9fe3e94e7b456998f07a86dde105cc52b7ba72576385fcdcf14fe`。

本轮范围只有四组固定输入。新输入执行和缓存执行未运行。
原始证据与程序包均保存在 Ubuntu：

- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/candidate-replay-rsh9wadc`
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-ordered-common-expression-runtime-r1286/program`

## 状态与后续

生产 704 文件、后端 648 文件、reference／门限 6 文件和 x86 配置核验一致。
联合 706 文件版本仍为 `awaiting_backend_qualification`；本轮是现有生产图重放，
来源接入 V13/V14 和联合版本尚无本轮后端验收。

下一步补齐联合版本下有序张量与融合矩阵的后端、程序包和新输入验收，再安排安全切换。
r1232 真实 Agent 三例仍按独立付费范围等待批准。其余通用算子、容量、近似／bootstrap、
整模型、ResNet20 和 CUDA 目标继续保留。

API／HTTP／安装／下载／后端修改／参数修改／commit／push 均为 0；当前无运行批次。
本轮只新增批准记录、验收产物和接续文档，生产源码保持。
