# Poseidon 迁移与容量阶段报告

2026-10-01，ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

## 核心链路迁移状态

**核心执行链已迁移到 Poseidon CPU。** 活动 CLI、应用组件和验证适配器采用 Hecate → dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU。真实 tracing、编译、I/O绑定、密态执行、解密、独立数值比较及程序包重载已有验收。活动入口拒绝旧 SEAL 配置，旧记录保持原绑定。

原 Agent 源码可独立验收，重新生成不是迁移条件。结果按来源计分：

| 来源 | Poseidon结果 | 范围 |
|---|---|---|
| 历史真实 Agent 原源码 | 27份通过，27包重载通过 | r371的14份+r375的13份；非唯一数学模型数 |
| 当前规则基线 | 本批21/21 | 18容量模型、1自动布局变体、2旧Linear/MLP |
| 人工等价表达 | 既有RMSNorm、Attention等通过保留 | 原r240/r242/r322身份；本批未追加 |
| 当前新生成真实Agent | 本批0项 | 生成及新增能力独立验收 |

核心迁移不代替完整历史兼容、bootstrap或整网目标。[r377](poseidon-historical-helper-acceptance-r377.md)记录新增原响应与helper来源核验；[r373](poseidon-migration-status-r373.md)保留此前核心验收。

## 兼容性回归状态

历史原源码覆盖Linear、真实Linear→square→Linear、多输入多输出、P8/16/32/64/128/256 packing、尾块、rotation、函数、对象数组、公开循环、展开参数、别名、增强赋值及代表性helper。27程序首轮108组/772值，重载另计相同数量；不宣称全部schema、shape或DSL组合兼容。

容量源码修改后，94项静态/组件/旧接口测试通过，失败与跳过均0，其中27份既有请求精确重建。旧Linear与Linear→square→Linear还完成当前编译和密态数值回归。历史575/518文件结果按原身份保存，不重标为当前579/522文件结果。

## 1024元素容量

新增 `poseidon-model-graph-v2`，要求模型内绑定 `resource_profile: "medium-v1"`。布局采用 `unified-capacity-chunked-inputs-v2`。旧v1、schema1–5、默认256元素及请求规则保留。

medium输入/输出各最多1024元素、各最多4逻辑张量；物理输入/输出分别最多4块，周期仍是4/8/16/32/64/128/256。可信encrypted-zero辅助输入沿用。资源身份、张量形状、偏移、有效尾部与输出selector不可变绑定。全归约在公开偏置之后屏蔽尾部，排除非零填充。

新集独立于冻结1200模型：6分区×3形状上下文=18模型，另有自动P128布局变体与2旧回归。

| 容量分区 | 三上下文实际Poseidon结果 |
|---|---|
| negate / 公开add / square | 通过；最大输入769 |
| 全mean、偏置和尾块 | 通过；最大输入1024 |
| 密文均值广播与中心化 | 通过；最大输入769 |
| 双输入加减乘组合 | 通过；最大总输入768 |
| 对齐分块concat | 通过；总输入/输出各1024 |
| C-order reshape | 通过；输入/输出各1024 |

18模型各做16组独立数学与CPU float64 Torch参考探针。密态执行沿用四组输入和原独立reference。

规则基线还实现sum、power、多项式、stack、flatten，但新v2三上下文密态验收尚未补齐。非均匀公开张量、部分轴归约、非对齐拼接、跨块Linear/Conv/Pool、全张量rotate继续开发。缺少规则答案不等于DSL或Agent不能表达；请求准备不依赖规则答案成功。

中间张量4096、原图256节点、展开1024节点和权重预算属逻辑分析上限；实际请求仍受128KiB、256常量注册项、源码/工作量、四块I/O和深度限制。ResNet20配置仍是逻辑准入，4096元素物理I/O未开放。v2 helper适配和定向基线尚未验收。

## 实际测试

| 层 | Passed / Failed / Skipped | 工作 |
|---|---|---|
| 静态/双参考/组件/兼容 | 94 / 0 / 0 | shape、篡改、边界、身份与请求 |
| 当前规则候选 | 21 / 0 / 0 | tracing、编译、密态、解密、逐项比较 |
| 六份容量包重载 | 6 / 0 / 0 | 直接执行产物；生成/编译0 |

首轮84组/36,420值，最大绝对误差 `2.720171492e-8`；重载24组/15,884值，max `2.567749062e-8`。门限不变：

~~~text
abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)
~~~

失败保留：r381的21项在tracing启动因可信模块漏挂载失败，0编译/密态。补齐沙箱可信模块允许列表后，r382/r383新绑定全部通过，网络与资源限制未改。r384在测试开始前Nix离线metadata返回1，原日志保留；底层原因未确认，串行r385完成94项。失败未删除或算作通过。

## 后端能力缺口与下一步

- 原12深多项式失败和容量诊断保留；未改编译器或Runtime。
- 真实bootstrap独立试验执行过，但有符号输入误差约1.04e-4未达门限。r235 ratio128试验等待批准、未启动；Agent Runtime尚未开放bootstrap。
- 接下来补非均匀权重、跨块Linear/空间计算和非对齐连接，每个新分区补三个上下文。
- PyTorch导入到v2容量图需实际验证。MLP/RMSNorm/Attention的新生成用于生成及自动修复验收；r368已批准未启动，执行前核验当前源码冻结。
- 容量新功能真实Agent评测另冻结案例、请求和预算申请批准。ResNet20物理布局、权重/数据、近似精度和整网密态尚未完成。
- x86原生、空白安装、CUDA暂缓，保留配置。

## 绑定与复现

当前执行579/应用522文件哈希与r382/r383/r385一致。依赖与二进制保持Poseidon API `62dbe225`、Runtime `23c8b942`、dacapo-modified `edcebf5d`及批准补丁。LLVM/MLIR18.1.2、GCC13.2.0、GMP6.3.0、Python3.10.14、NumPy1.25.2、ARM Torch2.0.1；完整SHA见[机器可读报告](poseidon-capacity-acceptance-r386.json)。

修改限于ARM checkout的Python布局、基线、准入、测试和文档，保留已有修改。r378/r386有恢复快照。Dacapo、Runtime、库、参数、reference及门限未变。编译≤2/链接≤1/密态串行，容量主产物约28.8MiB、执行约82秒。安装、下载、付费、commit、push为0。

下一条离线容量自检：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/agent.py --backend local candidate --   --case scripts/baseline/cases/capacity-v2/capacity_mean_tail_2.json   --unified-chunk-period 256 --self-test   --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

原证据位于work root的results/poseidon-capacity-static-r385、poseidon-capacity-minimal-r382、poseidon-capacity-remainder-r383。poseidon-stage-report-r386保存源码与修改前文档。结果限定这些绑定与输入，不是形式化等价或完整Agent能力完成。
