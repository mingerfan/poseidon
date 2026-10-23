# 原地操作与视图别名草案验收

本批完成 Agent 输入层的存储别名分析和不可变图转换草案。生产入口和 r1042 待批准绑定保持。

## 改动

按源程序顺序维护存储版本。共享分支、transpose、静态 slice/split 等视图在原地修改后读取新版本；先前已经算出的独立结果保持原值。

`view` 需满足步幅兼容性；`reshape` 在不兼容时创建副本。`clone` 保持稠密布局并使用独立存储，`contiguous` 区分直接返回已有连续视图和新副本。部分写回使用现有静态 gather/concat 图算子，结果由正常分解管线处理。

支持原地 ReLU、add/sub/mul，静态 shape/axis 和 tuple/list/字符串键字典输出。公开权重保持只读；非法重叠、错误 view、超容量和未登记调用返回受控错误码和源操作序号。

本批原地 ReLU 保留原函数参考，使用显式多项式 `0.5*x+0.25*x^2`、有效区间 [-1,1] 验证近似计算。近似模型与原模型的精度分别记录。

## 结果

| 检查 | 当前绑定结果 |
|---|---:|
| 测试方法 | 34 passed，0 failed/errors/skipped |
| 已知本地 FX 上下文 | 24/24 |
| 明文输入组 | 384 |
| 逐项输出值 | 6,304 |
| 原函数独立参考与 Torch 最大差异 | 0 |
| 近似独立参考与分解最大差异 | 5.551115123125783e-17 |
| 布局接纳/拒绝交叉检查 | 96 项一致：80 接纳、16 拒绝 |
| rank 4、rank 32 逻辑边界 | 各一例通过 |
| 本批近似相对原函数最大差异 | 0.11859375 |

24 个上下文包含三种原地 ReLU 写法各三个上下文、add/sub/mul 各三个上下文，以及 clone、复制/共享 reshape、contiguous、重复写入和 split 写回。每例保留四类基础输入和十二组确定性随机探针。

96 项布局检查另计，覆盖不同 shape、transpose/permute、步进与部分 slice、view/reshape/clone/contiguous 的组合。两个高 rank 边界仅证明逻辑图和明文语义；密文布局容量仍待检验。

r1065 的 30 个测试方法中有两个错误，均来自 split 静态列表被 FX 转成固定 `immutable_list`。适配器按精确类型接纳该容器，并补充维度列表与结构化输出检查。原失败与源码保留；r1066、r1067、r1068 各自绑定，没有覆盖旧记录。

当前 r1068 绑定：
`cf51eac2f03d2366bf195ccdf954b850c8185bdbd5741ed5b6b574b9e5ab8fd4`。

## 判断与边界

**已确认：**上述源调用、别名更新、复制隔离和拒绝规则有固定 Torch 与独立参考证据。34 个测试全部通过。旧除法/MaxPool、GELU、MHA 草案绑定以及 5952 个生产冻结文件保持。

**基于证据的判断：**这些常见原地写法可在 Agent 输入规范化层转换为合法不可变数学图，后端库保持现状即可继续接线。公开 gather 的分解成本需接受原资源检查。

**尚未验证：**生产隔离捕获与输入版本接入、实际密文布局、Hecate tracing、编译、Poseidon 执行、解密、程序包重载及真实 Agent 生成。本批测试多项式的应用精度未认证。

逻辑输入依照当前资源配置；本批主要模型 rank 1–2，补充布局与 rank 边界分别检查。来源假设为独立连续的 CPU float64 输入，任意跨输入存储共享、channels-last、任意 as_strided、copy_/resize_/set_ 或未登记副作用保持拒绝/待支持状态。

所有改动位于 Ubuntu 新建的 Agent 自有开发工具目录及文档。Poseidon、CKKS Runtime、Dacapo/Hecate、参数、旧 reference、门限和 x86 配置保持。API、安装、下载、编译、密态执行、解密和 Git 发布均为 0。

冻结 CKKS 门限：`1e-5 + 1e-4*abs(reference)`。本批明文差分：`atol=rtol=1e-12`。

## 文件与后续

源码：`scripts/baseline/benchmarks/tools/static_alias_source_draft_v1/`。
原始证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-alias-source-draft-r1068/`。

[机器记录](agent-alias-source-draft-acceptance-r1069.json) · [后端批次提案](agent-common-pytorch-offline-proposal-r1042.md)。

后续把除法/MaxPool、GELU、MHA 和别名处理统一接入显式输入版本，贯通可信隔离捕获、结构化反馈与输出结构，再对新增或受影响范围进行后端和真实 Agent 验收。r1042 的 151 项、25 份程序包复用仍待当前授权；r870 原 24 项已在 r878 关闭。

容量、bootstrap、密文交接、应用精度、ResNet20 和 CUDA 保持独立未完成状态。
