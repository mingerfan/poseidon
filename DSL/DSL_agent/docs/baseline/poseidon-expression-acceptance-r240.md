# Poseidon CPU 表达验收 r239/r240

已确认：原 RMSNorm 近似计算已有人工等价程序通过真实 tracing、Earth/CKKS 编译、RuntimePlan/bundle 检查、Poseidon CPU 密态执行、解密和四组输入比较。保留三次 Newton 迭代、增益、原模型和 reference。没有修改编译器、Runtime、库、参数或门限。

| 验收 | 结果 |
|---|---|
| 相关静态/组件回归 | 42 passed，0 failed，0 skipped |
| RMSNorm：4 种表达 × w45/w40 | 5 passed，3 compiler failed |
| Attention：2 种表达 × w45/w40 | 0 passed，4 compiler failed |
| 当前绑定 RMSNorm | 16 个输出值，最大绝对误差 4.71756e-9 |
| 当前 RMSNorm 程序包重载 | 16 个输出值，最大绝对误差 2.17552e-9；生成/编译调用均为 0 |
| 付费 API、下载、安装 | 0 |

变体和水线配置不增加模型数量。候选均为人工程序，不计为真实 Agent 成功。原 scalar 基线和失败记录保留；默认规则基线尚未采用这些表达。验收针对冻结的近似计算，不证明原函数的应用精度。

基于证据的推断：packed 求均值、提前合并增益及提取独立公共系数能减少 RMSNorm 的串行开销。不能把原失败认定为该模型不可表达。

Attention 保留投影、真实 RoPE 系数、六阶 exp、三次倒数迭代和全部 K/V 输入输出。明文表达与数学解释器、Torch reference 在 16 组探针上一致；两种表达在 w40/w45 下仍被 Earth 累计 level/scale 检查拒绝，未进入密态执行。是否必须 bootstrap 尚未证明。编译诊断与所有绑定见 [JSON](poseidon-expression-acceptance-r240.json)。

运行器首次抽取公共入口时出现默认参数 NameError，在 Attention 后端执行前修复；随后 42 项测试通过。r239 原运行器快照保留，旧计划字段哈希命名差异在 JSON 中注明，没有改写历史计划。

程序包：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-expression-acceptance-r240/rmsnorm-program`

独立 bootstrap r233 批准已由 r234 执行结算：有符号输入超限。ratio128 是 [r235 新提案](poseidon-native-bootstrap-proposal-r235.json)，仍待批准，没有构建或运行。Poseidon 上真实 Agent 验收也须另行批准。当前无 commit、push、PR；已有修改和 x86 配置保留。
