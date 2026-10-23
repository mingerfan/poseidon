# R9a：Motivation 研究（不调用 LLM）

前置：Agent 已冻结（tag `pt-agent-freeze-v1`）。本轮只做测量，**不修改 pt_agent 已有的任何模块**（只新增实验脚本和测试），不调用 LLM，不使用报告集（测试集第 0–7999 张）。目标日期：2026-10-23～24。

## 0. 目的

论文要证明"没有本 Agent 时"存在以下问题。现有证据大多只来自 ResNet20，本轮在 4 个开发模型上系统地测量（编号对应 PLAN §6.4）：

| 实验 | 要证明的问题 |
|---|---|
| E1 设计空间扫描 | P2：同一模型、同一编译器下，DSL 层决策（每层的近似方法、阶数、定义域）让延迟和精度变化很大，可行又快的区域很小 |
| E3 手写成本 | P1：专家手写 FHE 程序的代码量和人工决定的数量 |
| E4 定义域安全 | P4：训练图上观测的激活范围不能代表没见过的图；按训练图定义域，余量不同，精度和发散差别很大 |
| E6 预测准确性 | P6：真实编译太贵、简化代价模型会误导；我们的预测与编译一致 |

E2（Dacapo 7 个模型的编译负担）需要合成权重和服务器，不在本轮。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/experiments/r9a_motivation.py   新：§2–§6，子命令 e1 / e3 / e4 / e6 / all
pt_agent/experiments/r9a_sampler.py      新：§2.1 配置采样（纯函数，便于测试）
pt_agent/tests/test_r9a_motivation.py    新：§7
../docs/pt_agent/STATUS.md               改
```

结果目录：`pt_result("r9a-<时间戳>")`，每个实验一个子目录。每个配置或每个样本评估完，立刻向 `<实验>/rows.jsonl` 追加一行；重启时跳过 rows.jsonl 中已有的 key，以便断点续跑。registry 中 round 记为 "R9a"。

### 评估用的数据（统一约定）

- **E1 精度**：`heldout_set(200)`，即测试集第 9000–9199 张，用来快速、低成本地比较大量配置。
- **E4 精度**："未见集" = 确认集与留出集的并集，即测试集第 8000–9999 张，共 2000 张。Agent 已经冻结，本轮使用这两部分只是测量，不影响任何选择。
- 校准：沿用各模型现有的 calibration（训练图，`calibration_set` 加上该模型的归一化方式）。
- 所有 validate 调用都用 `stop_on_divergence=False`，发散的图按错误计。

## 2. E1：设计空间扫描

### 2.1 采样器 `r9a_sampler.py`

```python
DEGS = (15, 31, 63, 127, 255)
MARGIN_RANGE = (1.0, 1.6)

def act_ops(program): ...   # 展开后 IR 中 kind=="act" 的 op，按程序顺序

def apply_config(base_program, calib_ranges, config) -> Program:
    """config 为每个 act 一个 dict(method, deg, margin)，按程序顺序排列。返回 deepcopy 后修改过的程序：
    - cheb：attrs 设为 method="cheb"、deg=deg、degrees=None、domain=(min*margin, max*margin)
      （与 rule.plan 相同的公式）；
    - minimax_comp（只允许 fn=="relu"）：method="minimax_comp"、deg=None、degrees=(15,15,27)、
      domain=(-B, B)，其中 B = margin*max(|min|, |max|)。
    min、max 取该 act 输入张量的 calibration 值。"""

def random_config(acts, rng) -> list[dict]:
    """每个 act 独立采样：fn=="relu" 时以 0.25 的概率取 minimax_comp，否则取 cheb；cheb 的 deg 从 DEGS
    中均匀抽取；margin 在 MARGIN_RANGE 上均匀抽取，保留 2 位小数。"""

def anchor_configs(acts) -> dict[str, list[dict]]:
    """固定的"人工常用配置"：
    - uniform-cheb-{deg}-{m}：deg ∈ DEGS、m ∈ {1.2, 1.5}，共 10 个；
    - uniform-minimax-{m}：m ∈ {1.2, 1.5}，共 2 个，只用于含 relu 的模型；非 relu 的 act 用 cheb 63。"""
```

base_program 为 `parse(rule.plan(graph, calib, deg=63, margin=1.5))`。采样只修改 act 的 attrs，不修改结构。

### 2.2 运行

- 对 4 个开发模型（models-v2）：random 配置用 `numpy.random.default_rng(20261023)`，ResNet20 和 SiLU 各 150 个，cnn_plain 和 mlp 各 200 个；另外加上全部 anchor 配置，以及两个参照程序：R6c 规则程序（只用于 resnet20）和 R8e7 冻结选中的程序（每个模型）。
- 每个配置记录一行：
  ```
  key, kind ("random" / "anchor" / "reference"), config,
  boots_pred, latency_pred_us（cost.estimate，depth-dp）, max_depth,
  top1, exact_top1, acc_drop_pp, n_diverged, max_u（validate，heldout 200 张）,
  feasible_like = (n_diverged == 0 and acc_drop_pp <= 0.5)
  ```
- 某个配置 typecheck 或 lowering 失败时，记录 error 后继续下一个。

### 2.3 汇总（写入 `e1/summary.json`）

每个模型：
- latency_pred 的最小、中位、最大值，以及 max/min 倍数；
- feasible_like 的比例；
- feasible_like 配置中的最低延迟；
- 各 anchor 配置的延迟和是否 feasible_like；
- 参照程序（R8e7 冻结）的延迟在所有 feasible_like 配置中的分位数；
- Pareto 前沿（acc_drop_pp 对 latency_pred 的非支配点）的 key 列表。

## 3. E3：手写成本（不需要 Nix）

- **代码行数**：非空、非纯注释的行数，Python 注释以 `#` 开头，C++ 注释以 `//` 开头。统计以下内容：
  - Dacapo 专家 DSL：`third_party/dacapo/examples/benchmarks/{MLP,ResNet,AlexNet,VGG16,MobileNet,SqueezeNet}.py`，以及它们共用的 poly 库 `third_party/dacapo/python/poly/poly/{Func,MPCB,Poly}.py`（单独列一行"共享库"）；
  - Dacapo 的 PyTorch 定义：`third_party/dacapo/python/poly/poly/models/{Lenet,ResNet,AlexNet,VGG16,MobileNet,SqueezeNet}.py`；
  - 手写 C++ 的 ResNet20：`benchmark/resnet20_gpu/` 下所有 `.cpp`、`.cu`、`.h`、`.hpp`、`.cuh`；
  - 我们的 4 个开发模型：PyTorch 定义（models/zoo.py 中的类，resnet20 的定义取 models/resnet20.py 实际加载的那个文件）、R8e7 冻结程序的行数，以及它对应的 LLM 原程序（宏形式）的行数。
- **人工决定数**：在专家 DSL 源码中，用正则统计激活调用（`HE_ReLU|HE_SiLU|HE_Max|relu|silu`）和线性/卷积调用（`HE_Conv|HE_ConvBN|HE_Linear|HE_DwConv|HE_ReshapeLinear`）的出现次数：
  ```
  人工决定数 = 3 × 激活调用数（方法、阶数、定义域）+ 线性/卷积调用数（布局与算法）
  ```
  开发模型用同样的公式，激活数和卷积/线性节点数取自 ATen 图。
- 输出 `e3/loc.csv`，列为 model、artifact、lines、decisions；以及 `e3/summary.json`。

## 4. E4：定义域安全

对 4 个开发模型：

1. **范围比**：在未见集（2000 张）上用 `reference_forward` 计算每个 act 输入的 min 和 max：
   ```
   ratio_hi = max(unseen_max / calib_max, 1.0)          # 仅在 calib_max > 0 时计算
   ratio_lo = max(unseen_min / calib_min, 1.0)          # 仅在 calib_min < 0 时计算
   ratio = max(ratio_hi, ratio_lo)
   ```
   每个 act 记录一行。汇总：每个模型的最大值、中位数，以及 ratio > 1.1 的 act 比例。
2. **余量扫描**：所有 act 都用 cheb 63（含 relu 的模型另做一组全部 minimax_comp），定义域 = 训练图校准范围 × margin，margin ∈ {1.0, 1.05, 1.1, 1.2, 1.3, 1.5}，用 `apply_config` 构造程序。这里 margin 可以小于 1.1，因为不经过 rule.plan。在未见集 2000 张上运行 validate，记录 top1、acc_drop_pp、n_diverged、max_u。
3. 输出 `e4/ranges.csv`、`e4/margin_sweep.csv` 和 `e4/summary.json`。

## 5. E6：预测准确性

1. **新编译**：从 E1 中，每个模型选 8 个 feasible_like 配置，按 latency_pred 的 0、1/7、2/7、…、1 分位各取最近的一个，key 去重。每个配置写成 program.ptdsl，然后调用 `run._inside`，参数为 `stages=['check','count','emit','compile']`、`planner='rule'`、`refresh='compiler'`（与 r8e6_reconfirm 的调用方式相同），记录 plan 的 boots 和 plan_latency。
2. **已有的编译对**（不重新编译）：从以下 REPORT 中读取（预测、实测）对：R6b 的 4 个 case、R8e5 的 4 个冻结程序、R8e7 的 4 个选中程序、2 个基线。预测值统一用当前的 cost.estimate 重算。
3. 对每个程序，同时计算 `cost.estimate(..., placement_model="lazy")` 作为对照。
4. 输出 `e6/pairs.csv`（列为 program、boots_pred、boots_lazy、boots_plan、latency_pred、latency_lazy、latency_plan）和 `e6/summary.json`：depth-dp 与 lazy 两种模型各自的自举数完全一致的比例、自举数的平均绝对误差、延迟相对误差的中位数和最大值。
5. 每次编译完成后调用 `toolchain.prune`，避免占满磁盘。

## 6. 资源与顺序

- 内存：validate 使用 VAL_BATCH=20；Mock 进程同一时间只跑一个。
- 顺序：e3 → e4 → e1 → e6（e6 依赖 e1 的 rows）。`all` 子命令按这个顺序执行；中断后再次运行会按 rows.jsonl 接续。
- 预计耗时（本机 2 核）：E3 几分钟；E4 约 1–1.5 小时；E1 约 3.5 小时（两个 ResNet20 各约 1.5 小时）；E6 约 1 小时。合计约 6 小时。用 setsid nohup 启动，并写 runner.pid。

## 7. 测试（test_r9a_motivation.py，用 nixrun）

1. **采样器**：
   - 同一 rng 种子得到相同的配置；
   - 非 relu 的 act 从不取 minimax_comp；
   - apply_config 后，cheb 的定义域等于 (min·m, max·m)，minimax 的定义域对称且等于 ±m·max(|min|,|max|)；
   - 生成的程序能通过 parse 和 typecheck，并且与 base 的 canonical 只在 act 的 attrs 上不同。
2. **anchor 配置**：resnet20 共 12 个，mlp（含 relu）共 12 个；uniform-cheb-63-1.5 的程序与 `rule.plan(deg=63, margin=1.5)` 的 canonical 相同。
3. **代码行数统计**：一段含空行、`#` 注释和代码的小字符串，计数正确；C++ 的 `//` 注释同理。
4. **范围比**：构造的数组得到预期的 ratio；calib_max ≤ 0 时不计算 ratio_hi。
5. **断点续跑**：rows.jsonl 中已有 key 的配置不会被重复评估（mock 掉评估函数并计数调用次数）。
6. **数据约束**：E1 和 E4 使用的图像索引全部 ≥ 8000（报告集不被读取）。可以 patch `report_test_set`，让它一旦被调用就抛出异常。
7. R1–R8e7 的全部测试通过。

## 8. 汇报

- 测试输出。
- E1：每个模型的延迟范围（最小/中位/最大和倍数）、feasible_like 比例、可行的最低延迟、anchor 的表现、冻结程序的分位数。
- E3：代码行数和人工决定数的表格。
- E4：范围比的分布和余量扫描表。
- E6：depth-dp 与 lazy 的对照汇总。
- 结果目录路径。STATUS.md 新增 R9a 一行。

## 9. 不要做

- 不修改 pt_agent 已有的任何模块（冻结），只新增 §1 列出的文件。
- 不调用 LLM，不读取报告集，不下载，不训练，不生成新权重。
- 不为了得到某种结论而调整采样范围或筛选结果；所有失败照实记录。
