# E5a：Agent 放置自举，编译器只做 scale 管理（2026-11-06，用户选方案 A）

## 0　目的

论文动机二改为：编译器在展开后的运算图上搜索自举位置，程序一大就超时或崩溃；PT-Agent 在生成程序时决定自举，
交给编译器的程序已经带好自举，编译器只做 scale 管理。现状与这个说法不符：
- 6 个冻结程序都是 `refresh="compiler", placement="depth-dp"`，自举由编译器的 depth-dp 放置，代价模型复现它。
  depth-dp 是团队移植时加进 DaCapo 的，上游 `--dacapo` 只有 optimized（M7：BERT/Qwen 3600 s 未完成；M7b：4 个 CNN 段错误）。
- 上游 DaCapo 自带 `--pars` 流水线（上游 `tools/optimizer.cpp:435`，"Perform Scale Management on bootstrapped models"）：
  保留程序里的 bootstrap，只做 ProactiveRescaling 和 EarlyModswitch。本构建的 pars 在此基础上追加 RuntimePlan 导出
  （构建源码 `src/tools/optimizer.cpp:629`）。
- `refresh="agent"` 的通路已经存在：`RefreshBackend` 同时包在代价模型和导出包的 trace 外面（`cost.py:120`、
  `export/templates/_package/runtime.py:80`），包的编译模式为 pars（`export/package.py:279`）。但它是惰性策略
  （乘法操作数 level 不够时才自举），R6b 在 ResNet-20 上为 461 次，depth-dp 为 383 次。

本轮只做第一步：让 Agent 端用代价模型里已有的 depth-dp 计算放置，写进程序交给 pars 编译，在 6 个冻结程序上验证
"预测 = 导出 = 编译"、编译时间，以及与编译器放置的自举数对比。不改提示词、反馈、精调和冻结程序的正文，API 0。

E4c 的结果说明 depth-dp 本身放得差：DaCapo 自带的 ResNet-20（SiLU，10,089 个 Earth op，HEAAN 配置）上，
optimized 放置 43.07 s、19 次自举（与 DaCapo 论文 Table 4 相同），depth-dp 0.013 s、85 次。所以本轮的 depth-dp 计划
只用来搭好"计划 → 应用 → pars 编译"这条通路并验证它精确；替换表的格式与放置算法无关，下一轮（E5b）换成更好的
程序层放置算法时直接沿用。（团队 M2 记录里，Qwen 的 RMSNorm 小图上 optimized / depth-dp 为 3 / 17 次，差距同向。）

## 1　新的放置策略 `depth-dp-plan`

### 1.1　语法

- header 新增取值：`refresh="agent", placement="depth-dp-plan"`。parser 接受该值；typecheck 规定它只能和
  `refresh="agent"` 一起出现。
- 已有组合的行为全部不变：`refresh="agent"` 搭配其他 placement 仍走惰性策略；`refresh="compiler"` 仍由编译器放置；
  `manage="dsl"` 的约束不变。

### 1.2　计算放置（代价模型侧）

新增 `pt_agent/plan.py`（名字可调整），入口 `plan_refresh(program, weights, *, profile="qwen48") -> dict`：

1. 用 `RefreshBackend(policy="record")` 包住代价模型本来用的 DAG 后端（Transformer 程序为 `CostDag`，其余为 `DagBackend`，
   与 `cost.estimate` 的选择相同），执行一次 lowering。
   - `record` 策略给每次产生密文的调用编一个顺序号（输入、二元运算、取负、产生新值的旋转），记录该调用的操作数
     对应的内部节点，以及这次调用新产生的内部节点集合。
   - `CostDag` 的一次调用可能产生多个内部节点（`sub` 拆成 `neg` + `add`），也可能因为值编号命中而不产生新节点；
     `finalize` 会重新编号。映射按"调用产生的节点集合"和 `finalize` 的 remap 换算。
2. 在 finalize 后的 DAG 上调用 `placement.depth_dp`，参数与 `cost.estimate` 完全相同。
3. 扩展 `depth_dp`，让它额外返回：每个被自举的节点在哪个切点被自举，以及哪些使用者读取的是自举后的值
   （按深度顺序在该切点之后处理的使用者，包括输出）。`Placement` 增加字段，已有字段和数值不变。
4. 换算成调用顺序号上的替换表：`(使用者调用号, 操作数位置)` 的列表，加上需要替换的输出位置。
   结果映射到同一内部节点的所有调用（值编号命中的重复调用）都按同样方式替换。
5. 返回 `dict(boots, cuts, substitutions, outputs, calls, plan_s, lower_s, dp_s)`，`calls` 为本次的调用总数。

### 1.3　应用放置

- `RefreshBackend(policy="plan", plan=...)` 按同样规则给调用编号，在执行第 k 次调用前把替换表里标出的操作数换成它的
  自举值（每个操作数对象只自举一次，沿用 `_refreshed` 缓存），输出同理。
- `plan` 策略下不做惰性插入。乘法操作数的 level 低于 `boot_in_level` 时报错（放置不可行），沿用 RValue 的现有检查。
- 结束时调用总数与计划的 `calls` 不同就报错。这是两次执行顺序一致的检查，不涉及哈希。
- 代价模型：`cost.estimate` 对 `depth-dp-plan` 程序用 `plan` 策略包住 `CountBackend`，返回的自举数、延迟和 level 直方图
  来自实际执行，并附上计划的 `boots`，两者不一致时报错。`placement_exact=True`。
- Mock：bootstrap 在 Mock 中是恒等，`plan` 策略下的 Mock 输出应与 compiler 模式逐位相同。

### 1.4　导出包

- `export_package` 对 `depth-dp-plan` 程序：调用 `plan_refresh`，把结果写到包内 `config/refresh_plan.json`；
  `runtime.invoke` 读取它，用 `plan` 策略包住 trace 后端。编译模式为 pars（沿用 `package.py:279`）。
- 包里复制的 `refresh.py` 随之更新（`CORE_SOURCES` 已包含它）。
- 原有包格式和 `refresh="compiler"` 的包逐字节不变。

## 2　实验：6 个冻结程序

程序（正文不变，只把 header 改成 `refresh="agent", placement="depth-dp-plan"`，记录这一行改动）：
- BERT：`/home/lhohy/pt_agent_results/A1-a624-20261102/bert/replay-session-01/program.ptdsl`
- Qwen：`/home/lhohy/pt_agent_results/A1-a624-20261102/qwen/replay-session-01/program.ptdsl`
- mlp、cnn_plain、resnet20_silu、resnet20：`/home/lhohy/pt_agent_results/r8e7-reconfirm-20261022T122558/models/<model>/`

模型、权重、校准文件和 A1 / A1-optimized 的 compiler 模式相同（BERT/Qwen 的包和权重路径见
`A1-optimized-20261102/{bert,qwen}/Agent/REPORT.json` 的 `source_package`、`weights`；CNN 见 `cnn_driver.py` 的读法）。

每个程序按顺序执行（结果目录 `/home/lhohy/pt_agent_results/E5a-agent-placement-<开跑日期>/<model>/`）：
1. **计划**：`plan_refresh`，记录 `plan_s`、`lower_s`、`dp_s`、`boots`、预测延迟。在 Nix 外运行即可。
2. **一致性**：
   - 对同一程序的 compiler 模式运行 `cost.estimate`，记录它的 `boots_pred`、`latency_pred_us`，与计划比较。
   - 在 2 个校准样本上运行 Mock，`plan` 模式与 compiler 模式的输出要逐位相同。不读报告集。
3. **导出、trace、编译**：用与 A1 compiler 模式相同的 trace/compile 入口（`u4_native_continuation`、trusted_run、限制不变），
   编译参数取包内 `compile_argv.json`（pars）。记录 trace 和编译的耗时、峰值 RSS，`plan_stats` 的自举数和步数，
   以及按现有延迟表算出的计划延迟（`cost.plan_latency`）。
4. **清理**：`toolchain.prune`，再按 50 MiB 规则清理（同 E4）。

顺序：mlp、cnn_plain、resnet20_silu、resnet20、bert、qwen。某个程序失败时记录原因，继续下一个，不重试。

比较对象全部引用已有记录，不重跑：
- 编译器 depth-dp：A1-a624 验收（BERT/Qwen）和 A1-optimized-20261102 的 CNN depth-dp 记录（自举数、编译耗时）；
- 编译器 optimized：M7 / M7b（`A1-optimized-20261102`）。

## 3　测试

- 单元测试：
  - 小程序（mlp 和一个现有 Transformer 测试夹具）上，`record` 与 `plan` 两次执行的调用总数和替换位置一致；
  - `plan` 模式的 `estimate` 自举数等于计划自举数；
  - Mock 逐位相同；
  - parser/typecheck 接受新取值，拒绝 `refresh="compiler", placement="depth-dp-plan"`。
- 离线核对：6 个程序的计划自举数与 compiler 模式 `estimate` 的 `boots_pred` 对比，全部列出（不要求相等，见 §4）。
- 现有测试全部照常通过；惰性 `refresh="agent"` 程序（R6b）的预测数值不变。

## 4　判定与记录

- 每个程序记录三个自举数：计划、导出包实际插入的、编译后 RuntimePlan 里的。三者应当相等；不相等时写明差异来源
  （例如前端值编号合并了两次自举），不停止。
- 计划自举数与编译器 depth-dp 自举数可能不同：编译器在前端规范化后的 IR 上放置，切点附近的活跃值集合可能有差别。
  两者都列出，差值写进 E5a.md。
- pars 编译失败（例如 level 不足）时记录日志尾部和失败的 pass，列为阻塞，不改编译器。
- E5a.md 另附一段：构建与上游 optimizer.cpp 中 pars 流水线注册的文本差异（只读源码），供论文 Setup 使用。

汇总表（E5a.md 和 E5a.json）：

| 模型 | Earth op | 编译器 depth-dp：自举 / 编译 s | 编译器 optimized | Agent 计划：plan_s / 自举 | 导出 / 编译自举 | pars 编译 s / RSS | 计划延迟 vs depth-dp 延迟 |

## 5　不要做

- 不改提示词、反馈格式、精调规则、能力注册表和冻结程序正文（冻结 v1 及 A1 统一规则不变）；不调用 API。
- 不改编译器、上游脚本和前端；不提高限制；失败不重试。
- 不删除历史结果；不提交、不推送。
- 本轮不做新的放置算法、按语句缓存代价、会话耗时优化和新的 Agent 会话，它们留给后续轮次。

## 6　完成标准

- §1 的实现和 §3 的测试通过；6 个程序各有 REPORT.json（完成，或写明失败原因）。
- E5a.md、E5a.json 齐全；registry 中 round 为 `E5a`；最后一次残留进程检查为 0。
- STATUS.md 新增 E5a 一行，"与 spec 不一致之处"逐条列出。Nix 独占，串行。
