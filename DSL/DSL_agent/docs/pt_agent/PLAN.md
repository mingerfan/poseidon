# Poseidon DSL Agent：现状诊断、工程路线图、逐轮开发方式与 DAC 论文规划

> 本文件是路线图，不直接交给 codex 执行。codex 每轮只实现 Claude 写的 `docs/pt_agent/specs/Rn-*.md`（见 §3）。

## 设计原则：通用性优先（用户 2026-10-21 强调）

所有设计和修复都按"一类问题"来解决，不针对单个模型或单种写法打补丁。具体包括：前端按 ATen 核心算子映射；提示词的能力部分由注册表自动生成；模式标注放在数据里；kernel 支持一般的算子语义，不依赖特定模型的写法。凡是做不到通用的地方，在 PLAN 和论文中明确写为限制。

## 0. Context

目标：做一个组件化 Agent，输入明文模型，自动生成**性能感知**的 FHE DSL 程序，交给现有 Hecate/Dacapo 编译器（不改编译器）→ RuntimePlan → Poseidon 执行；同时在约 1 个月内产出 DAC 投稿。
已确认的决策：
- 设计一个高层 DSL，优化放在 Agent 层，编译器代码一行不改；
- Agent 冻结输出前允许使用内部工具循环（语法/类型检查、明文仿真、代价估计），真实编译与 CKKS 结果永不回传给 Agent；
- 按 R8f2 §8–§9，精调 `tune` 默认用 float64 张量参考解释器及三层缓存；面向 LLM 的 `evaluate`、冻结确认、独立精确检查和最终报告用槽级 Mock，`validate` 默认也保持 Mock。Tensor精调在最终程序上增加一次完整Mock复核，核对feasible/agree并用Mock统计构造对外反馈。反馈全文证明按§10替代流程执行；当前静态检查发现既有路径差异，处理口径与进展见STATUS。精调预算、搜索轮数和数值条件保持；本轮不打新 tag。
- LLM 预算放宽到数百次 DeepSeek 调用；
- 目标 DAC，约 1 个月交稿；
- 目前只有本机（ARM，7 GB 内存、2 核、25 GB 空闲盘），服务器可协调；
- **输入要通用**：尽量扩展，主入口支持常用 Python/PyTorch 模型（`nn.Module`），JSON 图和受限 Python 源码保留为辅助入口；
- **短期验收目标是 ResNet20 通过**（定义见 §4.0 M1），排在其他所有实验之前。

---

## 1. 现状诊断（带证据）

### 1.1 方法层面的根本缺陷：让 LLM 一次性手写整网底层 DSL
- 交付路径是一次 DeepSeek 调用，直接返回整份 Hecate 源码（`scripts/baseline/deepseek_provider.py:245`，`delivery_component.generate_once:194`），prompt 约 144 KB。
- 4 次正式请求全部失败（0/4）：M01、M02 用满 32768 token 推理预算，没有产出 DSL；M03 只有 352 字节，且 `range(1,144)` 超出 `MAX_RANGE=128`；R01（ResNet20）被拒，原因是 `Unsupported native loop syntax`（`native_public_loops.py:43-57`，`x if False else y` 属于 IfExp）。
- **R01 即使通过语法检查也是错的**：`agent-r01-generation-r2909/delivery/candidate.py` 里，LLM 在注释中自己写了"无法在字节预算内展开 10 万+ 权重""只计算线性头"；ReLU 用 `if False else` 占位，16 个 helper 都是 `c*0.0 + p*0.0`。LLM 把 Transformer 级别的转写工作量当作了自由文本生成，这条路线在完整网络上走不通。
- 仓库里其实已经有确定性的整图 lowering（`unified_graph_lowering.py`、`unified_capacity_lowering.py`、`fx_to_hecate.py`），但只作为 baseline，没有接入 Agent。Agent 没有可调用的 kernel 库，也没有工具。

### 1.2 "性能感知"目前基本没有
- 布局选择 `application_planning/layout.py:_score` 按 I/O 密文数、公开元素数等做字典序排序，不是 FHE 代价；`capacity_layout_costs.py` 自己也承认"不是 level 估计"；`resources.py:86` 的 `ckks_level_consumption=unknown`。
- 没有 BSGS/对角线矩阵乘的代价搜索，没有近似多项式的阶数/定义域选择，也不预测自举次数。
- 现有设定把 rescale/modswitch/bootstrap 全部交给 `--dacapo`。这和学长的论点正好相反：学长认为 DSL 生成阶段就该承担这部分优化。

### 1.3 DSL 本身的局限（可直接写进 Motivation）
- Hecate 前端（`third_party/dacapo/python/hecate/hecate/expr.py`）只有 `+ - *`、取负、`rotate`、`Plain(scale=)`、`bootstrap`。**没有 rescale/modswitch/level 语法**；`--dacapo` 流水线的第一步就是 `createRemoveBootstrap()`（`tools/optimizer.cpp:580`），会删掉用户写的 bootstrap。所以 DSL 无法表达刷新意图，全部优化都推给 MLIR 层。
- 冗长：Dacapo 自带的 ResNet20 DSL 是 `examples/benchmarks/ResNet.py` 136 行，加上 `python/poly` 库 1588 行（MPCB.py 752 行），约 1.7k 行；手写 C++ GPU ResNet20（`benchmark/resnet20_gpu/`）5240 行；Qwen DSL kernel 约 1106 行。
- **关键可用点**：`hecate-opt --pars`（`optimizer.cpp:630`，"Scale Management on bootstrapped models"）会保留用户写的 `hc.bootstrap`，只做 rescale/modswitch，然后导出 RuntimePlan。也就是说，Agent 可以自己决定刷新位置，编译器只承担轻量的 scale 管理，**编译器代码一行不改**。

### 1.4 编译器负担（已有数据，均来自 188Server）
- Qwen 2-token 单 block：Earth IR 165,169 个 op（2.09 GB）。原版 Dacapo optimized placement 第一次因 SIGXCPU 被杀（566 CPU-s），重试在 900 s 墙钟上限被杀；卡在 `BypassDetection`，调度了 74,657 次整图 clone。depth-DP 27 s / 3.67 GB；greedy 4,084 次自举，depth-DP 448 次。只看 RMSNorm：3 / 17 / 10 次（启发式差约 6×）。
  来源：`/home/lhohy/Code/Poseidon/test-results/qwen-dacapo-188-20261010/`
- Qwen 24 层：前端 451 s / 18.25 GB，编译 593 s / 60.6 GB，10,835 次自举，plan 4.82 GB JSON + 14.7 GB bundle（`test-results/qwen24-dacapo-188-20261010/`）。
- Qwen 2 层 decode：手工打包的 Linear/LM head + depth-dp 只需编译 25 s（`DSL_benchmark/.../evidence/compile-full-timing.json`），全保留时 RSS 382 GiB。这说明 DSL 层的打包决策会直接决定编译规模和内存。
- **缺口**：仓库里没有 CNN（ResNet20 等）在 Dacapo 下的编译时间数据，需要补（见 T1）。

### 1.5 工程与流程问题
- 规模失控：A 线 `scripts/` 1.1 GB、9427 个 py；`baseline/` 下 996 个核心模块、14.1 万行；`baseline/benchmarks/` 1.0 GB，是 7504 个快照副本（`function_construction.py` 出现 19 次）；17 个近乎相同的 `stage2_*_rNNN.py`。
- 文档/证据过载：docs 共 3295 个文件；`PARALLEL_DEVELOPMENT.md` 116 KB；`DELIVERY_CONTRACT_CURRENT.json` 约 700 个键，部分已过时；结果目录 18,235 个、57 GB（上限 64 GiB，已用 62 GB）。r2908→r2913 几轮只有 1 次真实生成，而且是契约失败。
- 规则互相冲突：AGENTS 写 48 GiB，合同写 64 GiB；AGENTS 禁止 SHA 校验，文档里却处处是 SHA 绑定。API 只批了 6 次，没法做任何统计实验。

### 1.6 后端依赖（不归你负责，但影响实验）
- CPU 原生 Boot 没有实现（`poseidon_cpu_api.cpp:625` 直接抛异常）。GPU 有原生 Boot（`poseidon_gpu_api.cpp:714`）。Qwen 的 CPU 604 次自举用的是 gy 服务器上一个独立的 bootstrap48 回调，不在仓库里。
- 结论：本机能对**不需要自举的模型**跑真实 CKKS；ResNet20 这类模型的端到端 CKKS 需要协调 GPU 或 gy 的 CPU 回调。

---

## 2. 新路线：两层 DSL + 工具化 Agent（编译器不改）

```
PyTorch / JSON 图（复用现有 intake）
  → Analyzer：形状、校准集取值范围、算子分类
  → Planner Agent（DeepSeek）⇄ 工具：parse/typecheck · CountExpr 代价/level · Mock反馈验证 · 张量参考解释器精调
      输出：PT-DSL 程序（高层、带性能注解，可读）
  → 冻结（Agent 声明完成，记录迭代次数）
  → Lowering：可信 kernel 库 → Hecate 图（hc API 直接 trace，不 exec LLM 代码）
  → hecate-opt：refresh=compiler → --dacapo --boot-placement=depth-dp
               refresh=agent    → 显式 hc.bootstrap + --pars
  → RuntimePlan → Poseidon（CPU：无自举模型；GPU/服务器：有自举模型）
  → 外部评测（三层误差，维持原容差）
```

职责：LLM 只做**决策**（布局、算法、近似方法/阶数/定义域、融合、刷新点、level 预算），并能用已有算子组合出长尾算子；**底层展开全部由可信 kernel 完成**，转写 10 万权重这类事不再交给 LLM。同一份 kernel 代码跑三个后端：HcExpr（真实 trace）、MockExpr（numpy 仿真）、CountExpr（计数 op/rotation/depth/level）。所以代价模型的计数按构造就是精确的，不需要手推公式。

### 2.1 PT-DSL 规格 v1（受限 Python 语法，ast 解析，永不执行）

```python
@fhe.program(slots=16384, refresh="agent", placement="depth-dp", l_max=..., boot_out_level=...)
def resnet20(x: Cipher[1, 3, 32, 32]):
    x = pack(x, layout="mux", gap=1)
    with region("stem", level_budget=3):
        y = conv2d(x, w=W["conv1.weight"], bn="bn1", stride=1, pad=1, algo="mpcb")
        y = act(y, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-12.0, 12.0))
    y = refresh(y)
    y = basic_block(y, p="layer1.0", stride=1)
    ...
    z = avgpool(y, k=8)
    return linear(z, w=W["fc.weight"], b=W["fc.bias"], algo="bsgs")

def basic_block(x, p, stride):     # 用户自定义宏，参数只能是 tensor/字面量/字符串前缀
    ...
```

- 语句：赋值调用、`with region(name, level_budget=int)`、`return`、顶层宏 `def`。v1 不支持 `for` 和 `if`。`W[...]` 下标只允许字符串字面量，或"字符串参数 + 字面量"拼接。
- 算子与优化旋钮：
  - `pack/relayout(layout=flat|mux|rep, gap, period)`
  - `conv2d(algo=mpcb|tap, bn=前缀|None)`
  - `linear/matmul(algo=bsgs|diag|rowsum, n1=None)`
  - `avgpool`、`add/sub/mul/scale/bias`
  - `act(fn=relu|silu|gelu|sigmoid|tanh|exp|inv|rsqrt, method=minimax_comp|cheb, degrees|deg, domain)`
  - `poly(coeffs, basis)`、`flatten/reshape/concat/slice`
  - `refresh(x)`：刷新意图。refresh=agent 时 lower 成 `hc.bootstrap`；refresh=compiler 时 lower 成恒等，但仍参与代价预测
  - `region`：level 预算注解，typecheck 时校验
- 检查项：
  - 形状与布局兼容（布局不同必须显式 `relayout`）；
  - 权重名和形状与图一致；
  - region 内预测深度 ≤ 预算；
  - refresh=agent 时，任何值的 level 都不能低于 0；
  - 覆盖性用数值检查：exact 模式下（非多项式算子用精确函数），MockExpr 输出与参考前向的差 ≤ 1e-9。

### 2.2 通用输入层设计（PyTorch 优先）

```
nn.Module + 示例输入 ──► 前端 A：torch.fx / torch.export（新，主路径）
JSON 计算图          ──► 前端 B：现有 model_import_cli（复用）
受限 Python 源码     ──► 前端 C：现有 prepare_pytorch_source / pytorch_model_capture.capture（复用）
                          │
                          ▼
            规范化 PT-Graph（约 40 个核心算子 + 形状 + 公开权重 + 静态属性）
                          │  decompose.py：复合算子 → 核心算子
                          ▼
     覆盖检查：每个节点 ∈ kernel 库 ｜ 可分解 ｜ 交给 Agent 组合（T13）｜ 明确报错并列出算子
```

- 核心算子集（PT-Graph op）：`conv1d/conv2d`（含 groups/depthwise、stride、pad、dilation）、`linear/matmul/bmm`、`add/sub/mul/div_const/neg`、`avgpool/adaptive_avgpool(静态)`、`relu/gelu/silu/sigmoid/tanh/hardswish/exp/softmax/rsqrt`、`batchnorm(推理)/layernorm/rmsnorm/groupnorm`、`reshape/view/flatten/permute/transpose/concat/slice/split/pad/channel_shuffle`、`sum/mean`、`constant`、`identity/dropout(eval→identity)`。
- 分解规则（`decompose.py`，确定性）：BN→affine（后续 IR 层折叠进前一个 conv/linear）；LayerNorm/RMSNorm→mean/var/rsqrt/affine；MHA→matmul+softmax；`adaptive_avgpool` 在静态形状下→avgpool；`max_pool`→用近似 max 的多项式组合（标记为近似）；`flatten/view/permute`→布局算子；dropout 与 eval 状态下的 identity 删除。
- 约束：固定形状、eval 模式、静态控制流；动态形状、数据相关控制流、训练模式在导入阶段明确报错，并给出原因和节点位置。
- 复用：现有 intake 已经覆盖的算子范围（README 中 Linear/Conv/ConvTranspose/归一化/Attention/gather/tensordot 等）全部保留，前端 B/C 的输出也转换到同一个 PT-Graph。
- 通用性度量（论文 RQ5）：用 torchvision 和 timm 的常用模型（ResNet18/34、MobileNetV2、ShuffleNetV2、SqueezeNet、VGG11、EfficientNet-B0 小分辨率、ViT-tiny 单 block）做"导入 + 覆盖"测试，报告能导入的模型比例和节点覆盖率。这只做图层面的检查，不需要编译。

---

## 3. 开发方式：逐轮 spec（Claude 设计 → codex 实现 → Claude 审查）

### 3.1 流程
1. **Claude 写 spec**：每轮一份 `docs/pt_agent/specs/Rn-<主题>.md`。写之前 Claude 先读要复用的源码，把确切的调用方式写进去。每份 spec 包含：
   - 文件清单（新建/修改哪些，精确到路径）；
   - 每个函数的签名、数据结构、关键算法（伪代码或公式，例如布局的 slot 映射、旋转步长）；
   - 要复用的现有接口及调用示例；
   - 测试用例和期望值（codex 照写）；
   - "不要做"清单和完成标准。
2. **codex 实现**：只实现对话中指定的那一份 spec。spec 没写的一律不做；有疑问或遇到报错，就停下来写进 STATUS.md 的"阻塞"一栏，不自行设计替代方案。
3. **用户转达**：codex 汇报后，用户在 Claude 会话里说"Rn 做完了"（或贴出阻塞内容）。
4. **Claude 审查**：看 `git diff`、跑本轮测试、对照 spec 检查，列出要返工的项（返工也写成小 spec），然后根据实际实现写下一轮 spec。

节奏：每轮控制在 codex 半天到一天能做完，目标每天一轮。PLAN.md 的路线图随审查结果更新，由 Claude 维护。

### 3.2 给 codex 的常驻规则

写在仓库根 `AGENTS.md` 开头的"当前主线：pt_agent"一节（2026-10-20 起生效，以那里为准）。要点：只读当前 spec 和 STATUS.md；只实现 spec；spec 没写的不做，有疑问写进"阻塞"栏；新代码只放 `scripts/pt_agent/`；不改 baseline、third_party、编译器；API 预算 300；结果写 `$PT_RESULTS`（`/home/lhohy/pt_agent_results`，仓库外）；过程记录不进 git。

2026-10-20 起三个工作目录已合并为 `/home/lhohy/Code/Poseidon`（分支 `lhy-agent-dsl`），Dacapo 13 文件补丁保存为 `DSL/DSL_agent/validation/dacapo-13file.patch`。

### 3.3 STATUS.md 模板（R1 由 codex 按此创建）

```markdown
# pt_agent 状态（最后更新：日期 / git rev）
## 轮次
| 轮 | spec | 状态(未开始/进行中/完成/阻塞/返工) | 改动文件 | 测试结果 | 与 spec 不一致之处 |
## M1 ResNet20（§4.0 的 1–5 条，各写 通过/未通过/待执行 + 证据路径）
## 阻塞（需要用户或 Claude 决定的问题）
## API 已用 / 300
## 最近 5 次运行（run_id、模型、结论）
```

### 3.4 给 codex 的每轮消息模板

```
本轮：DSL/DSL_agent/docs/pt_agent/specs/Rn-<主题>.md。只实现这份 spec。
完成后更新 STATUS.md 并按 AGENTS.md 的格式汇报。
```

---

## 4. 工作包路线图（设计粒度，细节在各轮 spec 中给出）

下面的 T0–T14 只用来界定范围和划分轮次，**不直接交给 codex**。其中标注"待定"或写着"找出""哪个能跑就用哪个"的地方，都由 Claude 在对应那一轮的 spec 里定下来。

### 4.A 轮次总览（按阶段，状态截至 2026-10-21）

| 阶段 | 轮 | 内容 | 状态 | 日期 |
|---|---|---|---|---|
| A 基础设施 | R1、R2、R2b | PT-DSL 的 IR/parser/typecheck；Mock/Count/Emit 三个后端；工具链（trace、eva/dacapo/pars 编译、本机 CKKS） | ✅ 完成 | 10/20 |
| B 前端与 kernel | R3、R4、R5、R5b、R5c、R5d | torch 导入 + 参考前向 + 校准；mux 卷积/BN 折叠/下采样；Chebyshev 激活、池化、全连接；常量内存修复；发散记录；可信运行器 | ✅ 完成 | 10/20 |
| C 端到端与代价 | R6、R6c、R6b、R6b2 | lowering、规则规划器、`pt_agent.run`；验证集选余量；代价模型（Count + 延迟表 + 惰性自举）、`--pars` | ✅ 完成 | 10/21 |
| D LLM Agent | R7、R7b、R7c | DeepSeek agent + 工具循环；结构化反馈；思考模式对照 → **M1 第 1–4 条首次通过（R7c B）** | ✅ 完成 | 10/21 |
| E 通用化与性能感知 | R8a | 开发模型库（4 个）、规则基线 | ✅ 完成 | — |
| | R8a2 | 通用 ATen 前端（make_fx + 映射表 + 依赖数据的控制流检查） | ✅ 实现完成（51 项测试通过；3 个模型的规则基线编译通过） | 10/21 |
| | R8a3 | 修正小模型训练配方（v1 下 MLP 发散 → v2 加梯度裁剪 + 发散检查），重新训练并跑完 R8a | ✅ 完成 | 10/22 |
| | R8b | 能力注册表、模式标注、示例库、通用提示词 r8-v1（**只实现和离线测试**，真实运行并入 R8c） | ✅ 完成 | 10/21–22 |
| | R8d | 组合 minimax ReLU（Dacapo HE_ReLU 系数 15/15/27）：原型显示 cnn_plain 的一致率从 974 提到 998，mlp 提到 1000；深度 17 vs cheb255 的 10 → Agent 多了"近似方法"这个旋钮；rule/run 支持 --act-method | ✅ 完成 | 10/21–22 |
| | R8c | （在 R8d 之后）代价模型与编译器一致（移植 depth-dp，原型已验证 383/383、288/288）；资源约束；自举输入检查；热点反馈；r8-v2（**只实现和离线测试**，真实运行并入 R8e） | ✅ 完成 | 10/21–22 |
| | R8e | **精调工具（数值旋钮的贪心坐标下降）**：LLM 负责离散的结构决策（近似方法、布局/算法、宏结构、刷新位置），精调工具在 LLM 给出的可行程序上逐个激活尝试更低阶数、更窄定义域，每一步用代价模型 + 验证集检查，可行保留、不可行回退；先用少量验证图粗筛，再用 1000 张确认；不调用 LLM，结果交回 LLM 进入下一轮。**4 个开发模型的 agent 真实运行** → **M1 第 1–4 条在 LLM + 搜索的 agent 下复核** | ✅ 完成 | 10/21–22 |
| | R8e4 | **可行性验证改用留出数据**（测试集第 9000–9999 张做留出验证，第 0–8999 张只做报告）；批量运行可接续、test 阶段保存进度；重算基线；4 个模型的真实运行重跑。起因：R8e 精调后的 ResNet20 验证集 99.4%，测试集只有 75.1%——验证图取自训练集，测试图激活范围最多宽 1.33 倍 | ✅ 完成 | 10/21–22 |
| | R8e5 | 精度可行条件改为"留出集 top-1 不低于精确网络 −0.5pp"（精调确认 ≤0.3pp，用户 10/22 决定；R7c B 在留出集上准确率 92.2% vs 精确 92.1%，但一致率仅 98.6%）；验证批量 50→20（127 阶程序会被系统杀掉）；两条依赖数据的流程测试改为脚本化；然后跑完 R8e4 的基线和 4 模型真实运行 | ✅ 完成 | 10/21–22 |
| | R8e6 | **独立确认集**（测试集第 8000–8999 张，只在冻结时用）：候选按预测延迟从低到高确认，取第一个下降 ≤0.5pp 的；报告集改为第 0–7999 张；对 R8e5 的候选离线补做确认（不调 API）。起因：R8e5 的 ResNet20 精调程序（189 次自举）留出集 +0.6pp、报告集 −1.4pp，属于在同一批留出图上反复挑选的选择偏差 | ✅ 完成 | 10/21–22 |
| | R8e7 | 确认阶段按结果判定（准确率下降 ≤0.5pp，发散的图计为错误；不再要求 max_u ≤0.90——规则基线在确认集上 max_u 也有 0.906）；修正验证中"一张发散整批计错、并跳出循环"的问题；重做 R8e6 的离线确认（不调 API） | ✅ 完成 | 10/21–22 |
| F 冻结 | — | tag `pt-agent-freeze-v1`（用户 10/22 批准）：提示词 r8-v4、能力注册表、示例库、反馈格式、精调工具、确认流程自此不再修改。冻结时的开发模型结果：ResNet20 226 次自举（规则基线 383），报告集 −0.46pp；SiLU 160 次，−0.66pp；cnn_plain 72 次；mlp 24 次 | ✅ 完成 | 10/22 |
| G 评测 | R9a/R9b | ✅ R9a 已完成（E1 751 配置、E3、E4、E6 40 对）。R9a（spec 已写：specs/R9a-motivation-study.md；不调 LLM 的 motivation 研究：E1 设计空间扫描、E3 手写成本、E4 定义域安全、E6 预测准确性；E2 编译负担的大模型部分上服务器）；R9b（spec 已写：specs/R9b-comparison.md）：基线（朴素 DSL 编译 T1、直接 LLM B1、LLM 写 Hecate + 测试修复 B1+、Dacapo 专家 DSL B4、规则 B3）+ motivation 数据 + **三组对照：只用 LLM / 只用搜索 / LLM + 搜索**（同一套旋钮与评估预算）+ 消融（代价反馈、验证反馈、热点、思考模式） | ⏳ | 10/28–31 |
| | R9a2 | 在 2000 张未见图上复核 E1 中比冻结程序更快的可行配置（E1 只用了 200 张，噪声大）；E6 按程序的刷新模式选择预测方法；E3 补 MLP 的人工决定数。不调 LLM，约 1 小时 | ⏳ 待做 | 10/23 |
| | P1 | **可移植性与工具链解耦**（Claude 直接完成，用户 10/23 批准）：编译命令由 pt_agent 自己生成（7 种模式与旧命令行逐字相同）、参数文件内置于 pt_agent/compiler_profiles、不再导入旧 SDK 认证模块；位置统一由 paths.py 提供并可用环境变量覆盖；依赖本机产物的测试在缺文件时跳过（缺产物时 160 跳过、155 通过、0 报错）；Dacapo 本地补丁和使用说明入库。Agent 决策行为不变：315 项测试通过，mlp/ResNet20 端到端编译自举数 20/383 与之前相同 | ✅ 完成 | 10/23 |
| | P2 | **导出后端**（用户 10/23 要求）：PT-DSL → 结构化 Hecate DSL 包（与 DSL_benchmark 的 Qwen 包同格式：model.py 每个运算一行函数调用，近似参数显式写在 config，interfaces/manifest/README，全部相对路径，不导入 pt_agent）→ 编译器。run.py 的 compile 改为编译这个包；验收：与 Emit 发出的运算序列完全相同、与 Mock 输出完全相同、7 个程序编译后的自举数和运算计数与记录值相同。spec：`specs/P2-hecate-package.md`。结果：7 个程序 Emit 文本和常量完全相同、Mock 20 图完全相同、自举数 226/160/72/24/383/288/461 与记录一致、11 类运算计数全部相同；ResNet20 的 model.py 59 行（旧直线代码 18007 行）；327 项测试通过 | ✅ 完成 | 10/23 |
| | R8f | **张量级快速验证**（用户 10/23 要求优化精调效率）：循环内验证由 16384 槽逐条模拟改为张量层参考解释器（同一多项式系数与表达式），并去掉 validate 内重复的代价模型调用；加三层缓存（结果、结构化前缀激活、小函数）且与无缓存逐位相同；用 R8e5/R8e7 全部轨迹重放精调，决策逐条一致才切换默认；冻结确认和报告集测试仍用 Mock。spec：`specs/R8f-tensor-validate.md` | ✅ 完成：精调提速 13–17 倍；精调用张量引擎，结束后对最终程序做一次 Mock 复核；面向 LLM 的 evaluate、冻结确认和报告仍用 Mock。46/46 数值一致，224 条精调历史、18 轮反馈文本与冻结版本逐字一致；358/358 测试 | 10/23 |
| | R8f2 | **flat 尾槽缺陷修复**：R8f 数值对照发现 flat 布局激活后空槽位变成 p(0)≠0，被下一层 linear 的复制旋转卷入，使 Chebyshev-MLP 的 logits 带常数偏移（逐类等于 p(0)·W 行和，真实加密程序同样存在）。在 lower 的 act 后用一次明文加法把尾槽清零（不耗 level；p(0)=0 时不发任何运算，冻结程序逐字不变），然后完整重跑 R8f 验收并重算受影响的 R9a/R9a2 MLP 行。spec：`specs/R8f2-flat-tail-fix.md` | ✅ 完成（含 exact 模式的 sigmoid/exp 尾槽）。MLP 历史重算：E1 中 MLP 的可行配置 98→137；cheb 127 阶（5.29M µs、20 次自举、精度下降 0）比冻结的 minimax（6.29M µs、24 次自举）快 16%；cheb 63 阶（4.24M µs）在 200 张图上的下降恰为 0.5pp，处于边界，不算稳健。E1 延迟跨度 4.47× 不变；E4 的 cheb 余量 1.05–1.5 变为可行；R9a2 的 robust 68/68 不变 | 10/23 |
| | R8g | **MLP 重跑**（用户 10/23 决定）：用修复后的代码按冻结协议只重跑 MLP 的 Agent（≤8 次 LLM 调用），作为冻结 v1.1 的 MLP 部分；其余 3 个模型不受尾槽缺陷影响，沿用 R8e7。spec：`specs/R8g-mlp-rerun.md` | ✅ 完成（3 次 LLM 调用，累计 48/300）：结果与 R8e7 相同，仍是 minimax、24 次自举、6.29M µs，报告集 top-1 56.05%。第 2 轮 LLM 提出了可行的 cheb 255（8.64M），但它不是新的最好结果，按规则不触发精调；而从 255 降到 127 本可得到 5.29M。这暴露了"只对新的 LLM 最好结果做精调"这条触发规则的局限 | 10/23 |
| | R8h | **消融：对每个可行候选都精调**（用户 10/23 决定，10/24 改为消融）：精调触发由"只对新的 LLM 最好结果"改为"每个可行的 LLM 候选"（相同程序去重）；按冻结协议重跑 4 个模型。spec：`specs/R8h-tune-all-v11.md` | ✅ 完成，作为消融（用户 10/24 决定冻结仍用 v1，即方案 A）：mlp 24→16、SiLU 160→148，cnn_plain 72→87，ResNet20 去重后选出 325（v1 为 226）。结论：每个候选都精调会得到更激进的程序，在确认集上更容易发散，泛化变差 | 10/24 |
| | R8h2 | **确认池去重 + 重新确认**：confirm 前按规范化程序文本去重，确认次数按不同程序计数；用 R8h 的候选池重新确认 4 个模型（不调用 LLM）；ResNet20 预先登记"仍失败则重跑一次"。spec：`specs/R8h2-confirm-dedup.md` | ✅ 完成：366/366；去重后重新确认 4 个模型，ResNet20 第 1 轮 LLM 原样程序（325 次自举）通过，attempt-02 未触发；LLM 0 次，累计 63/300。冻结**不**采用 v1.1，tag `pt-agent-freeze-v1` 保持为冻结版本；去重修复保留在代码中 | 10/24 |
| | R8h3 | **恢复 v1 的默认精调触发规则**：`run_agent` 增加 `tune_policy`，默认 `"new_best"`（v1 行为），`"all"` 为 R8h 消融；后续 R9b 等默认使用 v1 行为。spec：`specs/R8h3-tune-policy.md` | ⏳ 待做 | 10/24 |
| | R10 | ~~未见模型所需的 kernel~~ → 并入 §4.B 的 U3/U4（用户 10/23 调整），未见模型评测在 U3/U4 各自完成后做 | ⤳ 并入 U3/U4 | — |
| | R11 | **M1 第 5 条：真实 CKKS**（40 张图 + 4 组探针），需要可自举的运行时（GPU 原生 Boot 或 gy 的 CPU 回调）；同时标定该平台的延迟表、自举输入上界和精度。**用户 10/23：暂不考虑 GPU**，论文暂以编译器给出的自举数和预测延迟为主指标 | ⏸ 暂缓 | — |
| H 论文 | — | 数据定稿、作图 → 摘要 → 全文 | ⏳ | 摘要 11/11，全文 11/18 |

**ResNet20 验收（M1）在哪一步完成：**
- 第 1–4 条（Agent 冻结程序原样使用、结构正确、测试集精度、编译成功）：**R7c 已通过**（10/21，r7-v2 提示词，旧口径 10000 张）。**R8e4 用冻结前的最终 agent 复核**（约 10/23），测试口径改为报告测试集 8000 张（第 0–7999 张），门槛为 torch 在这 9000 张上的 top-1 − 1pp；冻结版本的 M1 证据以 R8e7 确认后的结果为准（Agent 内部的精度可行条件为留出集 top-1 下降 ≤0.5pp）。
- 第 5 条（真实 CKKS）：**R11**，取决于何时拿到可自举的运行时。最晚 11/05 完成，才能写进论文；拿不到时，M1 记为"1–4 通过、5 待执行"，论文的 CKKS 部分改用本机不需要自举的小模型。

<details><summary>各轮详细完成标准（原轮次表）</summary>

| 轮 | 目标日期 | 覆盖工作包 | 写 spec 前 Claude 要读的代码 | 本轮完成标准（摘要） |
|---|---|---|---|---|
| R1 | 10/21 | T0 + T2 | 无 | 骨架和 registry；PT-DSL 的 IR、parser、printer、typecheck 加完整 ops_spec；样例往返通过，非法样例报出指定错误码 |
| R2 | 10/22 | T3（离线）：Mock / Count / Emit 三个后端，生成独立的 Hecate 前端程序，Nix 运行入口 | `first_output_dsl_math.py`、`poseidon_candidate_sandbox.py`、`hecate_python_env.enter_nix` | 三个后端在 P1–P5 上结果和计数一致；生成的程序经测试用的假前端运行后与 Mock 逐位相等 |
| R2b | R2 通过后 | T3（真实）+ T12-spike：在沙箱里 trace 生成的程序，用 eva / dacapo / pars 编译，本机 CPU CKKS | 当前 SDK 选择文件、`poseidon_runtime_compile.command`、`poseidon_validation.execute`、绑定周期常量的流程 | 最小程序编译通过，CKKS 与 Mock 的差在容差内；确认 `--pars` 保留了 Boot |
| R3 | 10/24 | T5（前端 A + ResNet20）：torch.fx 导入成 PT-Graph，numpy 参考前向，校准统计，CIFAR 读取；torch 2.0.1 下用 fx（没有 torch.export） | B 线 `model_reference.py`、冻结选取文件、CIFAR 记录、r1811/r1819 精度记录 | 34 个单算子用例与 torch 一致；ResNet20 共 74 个节点，与 torch 的误差 ≤1e-9；10000 张 top-1 与 91.73% 一致 |
| R4 | 10/27 | T4-a：mux 布局（单密文）+ conv2d（步长 1/2，BN 折叠到掩码）+ 下采样捷径（1×1 恒等选择卷积）+ 残差加 | `poly/MPCB.py`（MultParPack、MultParConvBN、SumSlots、InferShapes） | T1–T10 在 Mock 上与参考卷积一致；T1 计数 rot15/mul_cp13/add15；本机 CKKS 跑 T1、T7；ResNet conv1（n=16384）编译成功。算法已于 10/20 用原型验证 |
| R5 | 10/29 | T4-b：Chebyshev 激活（BSGS，深度为 log2(阶数+1)+1，另加 1 层仿射）、全局平均池化、全连接（diag/bsgs）、批量 Mock、ResNet20 按图组装；精确模式结构检查；校准集上的阶数×余量网格；整网 Emit 后尝试编译 | `poly/Func.py`；R3 的 calibration.json | 测试中的计数与原型一致；整网精确模式误差 ≤1e-9；网格 6 组结果；编译尝试或生成服务器作业。算法已于 10/20 用原型验证 |
| R5b | 10/29 | 修复常量去重造成的内存问题（16 字节摘要作为去重键；Count 不保存常量数据），重跑 r5_resnet20 | — | 整网 Count（63 阶）：14386 个 op、4992 个常量、182 个旋转密钥、深度 193、峰值 ≤1 GiB（10/20 已实测） |
| R5c | 10/29 | 网格中发散的配置照实记录（diverged、max_u），跑完 r5_resnet20 | R5b 的诊断 | 6 组网格结果；Emit 和编译尝试 |
| R5d | 10/30 | pt_agent 自己生成的程序改用可信运行器（不进沙箱；内存 6 GiB、单文件 8 GiB、1 小时），尝试整网 trace 和 dacapo+qwen48 编译 | R5c 的整网程序 | 整网编译的耗时、内存和自举数，或服务器作业 |
| R6 | 10/31 | T7 + T8：PT-DSL lowering、规则规划器（图 + 校准 → PT-DSL）、`pt_agent.run` 端到端入口（plan/check/mock/count/emit/compile/test） | R5 的 `resnet20_fhe.build`、R1 的 IR | lowering 与 R5 组装在 Count 上统计完全相等；精确模式误差 ≤1e-9；带 --final 跑测试集 top-1，即 M1 第 3 条 |
| R6c | 10/31 | 用验证集选 deg 和 margin（训练集中未参与校准的 2000 张；精确网络需要 1.33 倍余量）；测试集逐张统计发散的图并照实报告 | R6 的两次运行 | M1 第 3 条：测试集 top-1（发散的图算错） |
| R6b | 10/23 | T6 + T12：代价模型（Count + 延迟表 + RefreshBackend 惰性 level 模拟）、由 Agent 惰性放置自举（--pars）、编译后清理中间文件 | Qwen 的 latencyTable；R6c 编译出的 plan | 10/21 原型：(63,1.4) 惰性 382 次 = pars plan 382 次，depth-dp 288 次；127→63 阶，depth-dp 自举减少 25% |
| R6b2 | 10/22 | 修复内层读取 registry 的接线问题（改由外层传入校准路径），跑完 r6b_refresh | — | 4 行汇总表 |
| R7 | 10/27 | T9 → **M1**：LLM 规划器（DeepSeek，关闭思考）+ 工具循环（parse/typecheck → 精确模式结构检查并定位出错的激活 → 代价模型 → 500 张验证图上的近似精度与 \|u\|），最多 6 轮，冻结最好的可行版本；规则规划器作为基线 | `deepseek_http_worker.py`、`agent_credentials.load_api_key`、`generation_protocol.Config` | 离线测试用假的 LLM；真实运行 1 次连通性检查 + 1 次 agent（≤6 次调用）；冻结程序原样通过 M1 第 2–4 条 |
| R7b/R7c | 10/22 | 修正测试预期；反馈中给出输入形状、宏调用处的行号和源码文本；提示词 r7-v2 补充 pads 的语义和"只改报错的地方"；思考模式开与关各跑一次 agent | R7 的 agent_trace | 失败原因是反馈缺少信息、提示词漏了 pads 的语义，不是思考模式本身 |
| R8a | R7c 之后 | 通用化（上）：前端静态解析 size/shape，新增 global_avgpool_flatten 分解规则；模型库共 4 个模型（resnet20、resnet20_silu〔Dacapo 权重〕、cnn_plain 和 mlp〔用本机 CIFAR 训练，用户已批准〕）；规则规划器支持 flat 输入；4 个模型各跑一遍规则基线 | Dacapo 的 ResNet.py 和 resnet20.silu.model | 4 个模型都能导入并与 torch 一致；规则基线全部编译通过；resnet20 的程序与 R6c 逐字相同 |
| R8a2 | 10/22 | **通用 ATen 前端**（make_fx 真实跟踪 + 37 种 ATen 算子的映射表 + 按 data_ptr 恢复权重名），取代 fx 上的逐点补丁；zoo 改用 aten 前端；跑完 R8a | 10/21 实测：S1–S3、通道重排、ResNet20-SiLU 跟踪误差均为 0.0 | 34 个单算子用例和 4 个模型的 reference 与 torch 一致；resnet20 的 canonical 与 R6c 相同 |
| R8b | R8a 之后 | 通用化（下）：能力注册表自动生成提示词的算子部分；图摘要中标注 lowering_pattern；示例库（排除目标模型）；通用提示词 r8-v1；4 个开发模型各跑一次 agent；冻结约定 | R7c 的结论（思考模式开或关、反馈格式） | 提示词中没有模型专用的写法；4 个开发模型的 agent 运行全部可行 |
| R8c | 10/27 | 代价模型与编译器一致：DagBackend + 移植 depth-dp（GreedyBootstrapPlacement.cpp）；profile 资源上限；自举输入幅度 watch 与可选噪声；hotspots 反馈；r8-v2；4 个模型的真实运行 | GreedyBootstrapPlacement.cpp、R6b 的 runtime plan | 两个 R6b 程序的预测自举数等于 383/288，边界数 76，延迟误差 ≤2%；4 个模型的预测与实际对照 |
| R9 | — | 基线、motivation 和消融：朴素 DSL 编译对照（T1）、直接让 LLM 写 Hecate（B1）、Dacapo 专家 DSL（B4）、规则规划器（B3）；消融：去掉代价反馈 / 去掉验证反馈 / 反馈只给错误码 / 思考模式开关 | R2b 工具链、R6c/R7c 的结果 | 论文 motivation 和评测的数据表 |
| R10 | — | 补齐未见模型所需的 kernel（depthwise、通道重排、普通 avgpool、concat、matmul/softmax/layernorm 等）；打 freeze tag 后一次性评测未见模型；统计导入覆盖率（RQ5） | T11/T13 | 未见模型的可行率、精度和成本 |
| R8e | 10/26–27 | 精调工具：validate() 抽取、tuner.py（定义域对准 |u|≤0.80 + 阶数阶梯的贪心坐标下降 + 前缀二分回退）、agent 循环接入、r8-v3；4 个开发模型真实运行 | R8c 的代价模型与 validate | 原型：R7c B 定义域修复后可行（max_u 0.93→0.82）；精调后预测自举 288→248（−14%）、延迟 −13%，一遍约 29 分钟；加 0.992 的确认门槛防止贴边；真实运行要报告"LLM 原程序 vs 精调后"的对照 |
| R11 | 10/30–11/05 | M1 第 5 条：真实 CKKS 执行 + 目标平台标定 | 后端提供的运行入口 | 40 张图 + 4 组探针误差 ≤1e-5+1e-4\|ref\|，类别一致 |

</details>



### 4.B 通用化阶段（用户 2026-10-23 调整优先级）

用户要求在 R8f2 之后优先做三件事：①支持多种常用明文模型的描述框架；②尽可能多地支持算子，至少覆盖 NN 和 LLM，初步具备通用性；③支持多密文布局。R9b（三组对照）推迟到 U3 之后，R11（GPU）暂缓。

**现状**：PT-DSL 语法（`dsl/ops_spec.py`）已经定义了 matmul、softmax、layernorm、avgpool、concat、permute、split、poly 等运算；ATen 前端（`intake/`）已经能映射 bmm、softmax、layer_norm、gelu、max_pool2d、cat 等。真正的缺口在**可信 lowering**：`HANDLED_OPS` 只有 10 种运算，而且全部基于单密文（中间张量必须放进 16384 个槽位）。所以通用化的主体工作是"布局系统 + kernel 库 + 代价模型 + 张量参考解释器"四件套，每个新运算都要同时提供这四样。

依赖关系：U1 与 U2 可以先行；U3 和 U4 依赖 U2，因为 VGG/AlexNet 的中间张量和 LLM 的 FFN 都超出单个密文；U5 是 U4 的前提，softmax 和 norm 都需要它。

| 轮次 | 内容 | 验收 | 目标日期 |
|---|---|---|---|
| U1 模型描述框架 | `ModelSpec` 注册表：每个模型声明来源，三类都必须支持（用户 10/23 确认）：PyTorch nn.Module（含 torchvision/timm）、HuggingFace Transformers（按配置构建，处理 tokenizer、注意力掩码和固定序列长度）、ONNX（覆盖 TensorFlow/Keras、JAX、scikit-learn 的导出）、权重来源、输入规格（形状、归一化、dtype）、数据适配器（CIFAR、ImageNet 子集、token 序列）、评价指标（分类 top-1；LLM 用下一 token 一致率和 logits 误差）、容差。`validate` 和 `confirm` 的"准确率下降"改为调用指标接口，现有 4 个模型的结果保持逐位不变。前端增加 `torch.export` 路径（与 make_fx 并存），用于 HuggingFace 风格的模块 | 4 个现有模型在新框架下的全部数字逐位不变；HuggingFace 和 ONNX 各至少 2 个模型能导入成 PT-Graph（各含 1 个 CNN 或 MLP、1 个 Transformer；只验证导入、与原框架前向一致和覆盖率统计，不要求能 lowering）；同一个模型分别从 PyTorch 和 ONNX 导入，得到的 PT-Graph 等价 | 10/24–27 |
| U2 多密文布局 | 布局对象改为"张量 → 密文列表"：`MuxLayout` 增加跨密文的通道分块；`FlatLayout` 增加分块（向量切成多个 n 槽块）；新增 `TokenLayout`（每个 token 一行，参考 Qwen 包的 PackedVector）。conv、linear、mean、add、act 的 kernel 支持多输入、多输出密文；代价模型、DAG、depth-dp 复现、张量解释器和 P2 包导出同步支持；单密文是多密文的特例，现有程序发出的运算序列逐字不变 | 现有 7 个程序逐字不变；一个强制拆成 2 个和 4 个密文的 ResNet20 与 Mock、张量解释器一致；编译通过，自举数与代价模型一致 | 10/27–31 |
| U3 CNN 算子 | depthwise/分组卷积、一般 kernel 大小和步长、一般 avgpool/adaptive avgpool、maxpool（多项式 max，复用 DaCapo HE_Max 的组合 sign 思路）、concat、一般 slice/pad、独立的 batchnorm。用 DaCapo 自带权重（已批准）评测 AlexNet、VGG16、MobileNet、SqueezeNet、ResNet20-relu，与 DaCapo、Orbit 是同一批基准 | 5 个模型能导入、lowering、编译；与张量解释器一致；Agent 在冻结协议下生成，报告自举数和精度 | 11/01–05 |
| U4 Transformer/LLM 算子 | 密文×明文的 token 矩阵乘（复用 Qwen 包验证过的打包线性层）、密文×密文 matmul（QKᵀ、AV）、softmax（exp 近似、求和、倒数）、LayerNorm/RMSNorm（均值、方差、倒数平方根）、RoPE、SwiGLU 门控乘、causal mask、残差。目标是 1–2 个 Qwen2.5 decoder block（团队包的合成权重）加一个本地训练的小 Transformer | 与张量解释器一致；编译通过；Agent 能为每个 softmax 和 norm 选择近似参数 | 11/06–12（论文中可能只作为示范） |
| U5 非线性自动拟合 | 在可信库中自动生成近似，不需要新 DSL：`method` 增加 remez（任意函数单段 minimax）、composite（组合 sign，用于 relu/max，阶数拆分自动搜索）、newton（倒数、倒数平方根，用于 softmax 和 norm，初值多项式加迭代次数）。参数仍写在 PT-DSL 的 act/softmax/layernorm 上，系数在 lowering 时生成并写入 P2 包的 config | 每种方法在给定定义域上的最大误差达到要求；与张量解释器一致；可以作为 Agent 的新旋钮 | 与 U4 一起，11/03–08 |

| S0 编译器流水线验证（半天，不改编译器） | 确认两件事：①hecate-opt 能否用单独注册的 pass 组成"跳过 scale 管理"的流水线，并单独调用 RuntimePlan 导出；②`check-smu` 能否完整检查外部写出的已管理 earth MLIR。做法：取现有一份编译记录中编译器自己产生的 `lowered.earth.mlir`（已管理），送进自定义流水线，检查导出的 RuntimePlan 与原流水线导出的是否一致；再人为删掉一个 rescale 或 modswitch，确认 `check-smu` 能报错 | 两项都成立才继续 S1；任何一项需要改编译器就停下，由用户决定 | ✅ 10/23 通过：①自定义流水线 `cse,canonicalize,convert-earth-to-ckks,convert-upscale-to-mulcp,cse` → `canonicalize,emit-runtime-plan{...}` 对 6 个输入（含 mlp 24 次、ResNet20 226 次自举、23575 步）导出的 RuntimePlan 与原流水线逐步相同，常量逐个相同；②删 rescale、改 downFactor、删 bootstrap、改 level 共 20 个破坏变体，全部被 MLIR 类型校验（verifier）在解析阶段拒绝，没有一个被静默导出；`check-smu` 在 Release 构建中只有 assert（NDEBUG 下不生效），不能当检查器用；③自带的 eva/pars 流水线对已管理的输入不幂等（会改动管理或报错），S1 必须用自定义流水线 |
| S1 DSL 管理模式（复现编译器） | ①P2 包增加 `config/management.json` 和 `earth_backend.py`，`trace_model.py`、`compile_model.py` 增加 `--mode managed`，`model.py`、ops 和 kernel 不变；②pt_agent 新增 scale/level 管理器组件，**第一版逐条复现编译器的管理**（waterline rescaling、early modswitch、depth-dp 自举放置），由我们写出已管理的 earth MLIR，编译走 S0 验证过的自定义流水线；MLIR verifier 负责局部类型一致性，pt_agent 另需自建全局约束检查（每个 level 的 scale 不超过模数容量、自举输入 level 不低于 boot_in_level、scale 不低于 waterline），因为 `check-smu` 在 Release 构建中不生效；③PT-DSL header 增加管理模式（与 refresh="compiler"/"agent" 并列），代价模型和 Mock 支持该模式的 level/scale 语义 | 对现有 7 个程序，managed 模式导出的 RuntimePlan 与 compiler 模式的自举数和各类运算计数相同（理想情况下逐条相同）；`check-smu` 通过；compiler 模式完全不受影响 | 11/05–10 |
| S2 联合优化（研究） | 在 S1 的基础上，让近似决策（每层的方法、阶数）与 level 分配、自举位置一起选，作为 Agent 和精调的新旋钮；与"compiler 模式 + 同一近似"对比自举数和预测延迟 | 在至少 ResNet20 上找到比编译器管理更少自举或更低预测延迟、且精度满足冻结协议的程序；否则如实报告为负结果 | DAC 之后；如果 S1 提前完成，可以先做 ResNet20 的初步结果 |

**顺序与论文范围（10/23 用户确认把 DSL 管理模式放在 U1–U3 之后）**：S0 → R8h/R8h2（消融，冻结仍为 v1）→ R8h3 → U1 → U2 → U3 → S1 → R9b → U5/U4 → S2（S0 由用户 10/23 提前到 R8f2 之后；R8h 由用户 10/23 排在 S0 之后）。11/18 前能完整做完的是 U1–U3（CNN 通用性，与 DaCapo、Orbit 同一批基准），这是 DAC 版本通用性的主证据。S1 能按时完成的话，论文写成"同一份 PT-DSL 既能交给编译器管理，也能由 DSL 层管理，两者导出相同的 RuntimePlan"。R9b 是论文核心对照，必须在全文截止前完成，安排在 S1 之后。U4/U5 和 S2 大概率排不进 DAC 版本，作为未来工作；如果 S0 不通过，S1 和 S2 取消，时间还给 R9b 和 U5。

### 4.C 是否把 rescale、自举等编译器优化信息放进 DSL（Claude 2026-10-23 评估）

结论（10/23 更新）：**用户决定做 DSL 管理模式，排在 U1–U3 之后（见 §4.B 的 S0–S2）**。技术上可能不改编译器就能做到，先用 S0 验证；价值取决于 S2 能否与近似决策联合优化。以下为当时的分析，仍然有效；只做两类编译器缺少、又能通过现有接口传给编译器的"数据感知信息"；非线性自动拟合要做（即 U5），不需要新 DSL。

1. **能不能做**：
   - 编译器自带的 5 条流水线（eva、elasm、hecate、dacapo、pars）都会自己做 scale 管理，dacapo 的第一步还会删除用户写的 bootstrap；只有 pars 保留用户的 bootstrap，rescale 仍由编译器放置。Python 前端也没有 rescale、modswitch 语法。所以**只要走"Python 前端 + 自带流水线"这条路，rescale 就一定由编译器放置**。
   - 但编译器的 IR（earth 方言）本身有 `rescale`、`modswitch`、`upscale`、`bootstrap` 运算，类型里带 scale 和 level；`optimizer.cpp` 用 `registerEarthPasses`、`registerCKKSPasses`、`registerConversionPasses` 把各个 pass 单独注册成了命令行可调用的形式，其中包括检查 scale 管理正确性的 `check-smu`。
   - 因此存在一条**不改编译器代码**的路：DSL lowering 直接输出已经完成 scale 管理的 earth MLIR（自己放 rescale、modswitch、bootstrap），再用 hecate-opt 的单独 pass 拼一条跳过 scale 管理的流水线：`check-smu` → earth→ckks 转换 → upscale→mulcp → RuntimePlan 输出。
   - 待确认的是 RuntimePlan 输出能否作为单独的 pass 调用，以及 `check-smu` 能否完整验证我们生成的管理。这需要一个约半天的 spike，不改编译器源码。
2. **值不值得做**：
   - 更正：R6b 的 461 对 383，比较的是两种**放置算法**，不是代价模型与编译器的差异。代价模型对两种模式的预测都准确：refresh="compiler" 时按 depth-dp 预测 383，实际编译 383；refresh="agent" 时程序里的自举由 RefreshBackend 按**惰性策略**（level 不够时才自举）插入，预测 461，pars 编译 461。另一组同样一一对应：288 对 382。惰性策略本来就比 depth-dp 差。
   - 所以 DSL 层放置的上限取决于算法：照搬 depth-dp（代价模型里已经有）只能打平 383；要更好，需要 Orbit 式的联合优化（Orbit 比 DaCapo 好 7–19%），或者**与近似决策联合优化**，例如阶数、level 分配和自举位置一起选。后者编译器做不了，是唯一可能体现 DSL 层价值的方向，但属于新的研究问题，工作量大。
   - DAC 版本不做；保留 refresh="agent" 作为消融。如果 DAC 之后要做，先做 1 中的 spike，再设计"带 level/scale 注解的 PT-DSL"。
3. **值得做的"数据感知信息"**（编译器本来拿不到，DSL 层有校准数据）：
   - (a) **数值范围归一化**：用校准得到的范围，把缩放因子并入相邻权重和多项式的仿射变换，使自举输入落在后端要求的范围内。不额外消耗 level，接入真实 GPU 时必需；
   - (b) **明文常量的 scale 提示**：`hc.Plain(data, scale=)` 是前端现有的接口，可以按层设置精度。是否有收益，需要先做一个小实验再决定。
   - 这两项都放在 U2、U5 之后，按需要再开轮次。
5. **"绕过前端"的含义与组件化影响**（10/23 补充，供 DAC 之后决策）：
   - 现在的流程：PT-DSL → 可信 lowering → P2 包（Python，调用 `hecate.expr`）→ 运行 Python，由前端库 libHecateFrontend 记录运算并生成**未做 scale 管理**的 earth MLIR → hecate-opt 自带流水线（删除用户 bootstrap、放置自举、rescale、modswitch、转换到 CKKS 层、输出 RuntimePlan）。
   - 绕过前端的流程：PT-DSL（带 level/scale 注解或由新组件自动决定）→ 可信 lowering → **新组件"scale/level 管理器"**（在 pt_agent 内放置 rescale、modswitch、upscale、bootstrap，给每个值标注 scale 和 level）→ **新后端"earth 发射器"**（直接写出 earth MLIR 文本，不运行 Python 前端）→ hecate-opt 自定义流水线（`check-smu` 验证 → earth→ckks 转换 → upscale→mulcp → 输出 RuntimePlan，不再做 scale 管理）。
   - 组件化：可以作为**可选模式**加入，不破坏现有组件。PT-DSL 仍然是各组件之间的契约，header 增加一种管理模式（与现有的 refresh="compiler"/"agent" 并列）；新增的是一个管理器组件和一个后端，代价模型和 Mock 需要相应地支持新模式的 level/scale 语义。默认模式不变。
   - 但有三处边界会变：①**正确性责任**从编译器移到 pt_agent，编译器只用 `check-smu` 验证；②**交付物格式**：Python 前端写不出 rescale，P2 的结构化 Python 包表达不了这种程序，交付物只能是 earth MLIR，或者另定义一种带管理信息的包格式；③**论文定位**从"与编译器正交"变为"部分替代编译器的 scale 管理"，需要证明联合优化比编译器更好。
4. **非线性自动拟合**：PT-DSL 的 act 已经能描述方法、阶数和定义域，系数由可信库在 lowering 时生成（现在的 cheb 就是这样）。扩展方法（remez、composite、newton）和 softmax、norm 的近似参数，就是 U5，**不需要新 DSL**。真正需要新 DSL 的只有"显式 level/scale"一类信息，而它依赖编译器扩展，列为 DAC 之后的未来工作。

### 4.0 里程碑 M1：ResNet20 通过（短期验收，目标 11/03）
输入：B 线的 ResNet20 PyTorch 模型和权重（`DSL/DSL_agent/workloads/resnet20/model_reference.py`、`public_assets.py`，只读），**经前端 A 从 `nn.Module` 导入**。"通过"要求以下全部满足：
1. Agent（T9）生成的 PT-DSL 冻结后原样使用，过程中无人工修改；
2. exact 模式 Mock 与 PyTorch 参考差 ≤1e-9（结构正确）；
3. 近似模型在报告测试集（测试集第 0–7999 张）上的 top-1 ≥ torch 原模型在同一批图上的 top-1 − 1pp，发散的图按错误计。数据使用约束：定义域、阶数等一切选择只能使用校准集（训练图）、留出集（第 9000–9999 张）和确认集（第 8000–8999 张），报告集不参与任何选择。**本条不要求定义域覆盖校准集上观测到的极值**：越界造成的影响已经按结果计入准确率和发散数，单独报告"校准范围覆盖"仅作诊断。（2026-10-22 澄清；原文为"10000 张测试图 top-1 ≥ 91.73%−1pp，1000 张校准图只用于定义域选择"，后半句的本意是"不得用测试图选定义域"。）
4. hecate-opt 编译成功（refresh=compiler 用 `--dacapo --boot-placement=depth-dp`；refresh=agent 用 `--pars`），并记录时间、RSS、自举数；
5. 真实 CKKS：在可用的带自举后端（GPU 原生 Boot，或 gy 的 bootstrap48 CPU 回调）上跑 40 张固定图 + 4 组探针，误差 ≤ 1e-5+1e-4|ref|（相对冻结的 Mock 参考），预测类别与明文一致。

第 5 条依赖服务器和后端。到 11/03 还拿不到的话，M1 记为"1–4 通过、5 待执行"，同时把服务器作业包准备好。
M1 关键路径的卡：T0 → T2 → T3 → T4（只做 ResNet20 需要的：conv2d-mpcb、stride-2 下采样、shortcut 的 option A 零填充和 option B 1×1conv+BN 两种都要支持、BN 折叠、relu 近似、残差 add、全局 avgpool、linear-bsgs）→ T5（前端 A）→ T6 → T7 → T8 → T9。其余卡排在 M1 之后。

优先级：P0 必须进论文；P1 尽量完成；P2 时间允许再做。

**T0 骨架（P0，0.5 天）**
建 `pt_agent/__init__.py`、`registry.py`（`append(row: dict)`、`load()`）、`config.py`（读 `PT_RESULTS`、hecate-opt 路径默认 `/home/lhohy/poseidon-work/platforms/aarch64-linux/build-dacapo/hecate-18.1.2-nix/bin/hecate-opt`、OperatorSpec 路径）、`docs/pt_agent/STATUS.md`。
验收：`test_t0_registry.py` 覆盖写入两行再读回。

**T1 Motivation 编译实验脚手架（P0，1.5 天）**
`pt_agent/experiments/compile_bench.py`：
- 输入：一个 Hecate DSL 文件（.py）和 pipeline 配置列表：`dacapo+optimized`、`dacapo+depth-dp`、`dacapo+greedy`、`pars`（仅用于含 bootstrap 的源）。
- 流程：前端 trace 出 `.mlir` → `/usr/bin/time -v hecate-opt <flags>`，加 `--mlir-timing`；墙钟上限可配（默认 3600 s）。
- 记录：前端耗时/RSS、编译耗时/峰值 RSS、返回码/是否超时、Earth op 数（按 op 类型 grep 统计 mul/add/rotate/constant/bootstrap）、RuntimePlan 中的 bootstrap 数和 step 数。RuntimePlan 指标复用 `scripts/baseline/application_planning/compiled_resources.analyze_program`。flag 组合参照 `DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/evidence/compile-full-timing.json` 的 command 以及 `baseline/poseidon_runtime_compile.command`。
- 输入程序有两组：（a）专家 DSL：`third_party/ckks-runtime/third_party/dacapo/examples/benchmarks/{MLP,Lenet?,ResNet,AlexNet,VGG16,MobileNet,SqueezeNet}.py`；（b）朴素 DSL：同模型经 `baseline/fx_to_hecate.py` 或 `unified_graph_lowering.candidate_source` 生成（哪个能跑就用哪个，两个都不行就报告）。
- 本机只跑 MLP 和 LeNet；其余模型生成 `server_jobs/compile_bench/run.sh` 加输入清单。
验收：MLP 在本机产出一行 registry；脚本在 dry-run 模式下能列出全部服务器任务。

**T2 PT-DSL IR、解析器、打印器、类型检查（P0，2 天）**
文件：`ir.py`（dataclass：`TensorType(shape, kind, layout)`、`Op(kind, inputs, attrs, out, lineno, region)`、`Program(header, params, ops, macros, outputs)`）、`parser.py`（`parse(text) -> Program`，AST 白名单按 §2.1；宏在解析时内联展开；所有错误抛 `PTDSLError(code, lineno, msg)`）、`printer.py`（`print_program(p) -> str`，往返稳定）、`typecheck.py`（`check(p, graph) -> list[Diag]`：形状、布局、权重名/形状）。
算子签名表统一放在 `ops_spec.py`，每个算子写明参数名、类型、取值集合。
验收：MLP 和 ResNet20 两份手写 PT-DSL 样例（放 `pt_agent/examples/`）能通过 parse 和 print 往返；10 个非法样例各自报出指定的错误码。

**T3 三后端 Expr（P0，1.5 天）**
`backends.py`：
- `MockExpr`：numpy，slot 长度向量，`+ - * neg rotate bootstrap`。rotate 方向必须和 `baseline/first_output_dsl_math.py` 的 `_Value` 一致。
- `CountExpr`：只计数 add/sub/cmul/pmul/rot（记录 rotation 步长集合）/bootstrap；按值跟踪 `depth` 和 `level`，规则是乘法 level−1、bootstrap 置为 `boot_out_level`、level<0 报错。
- `HcBackend`：薄封装 `hecate.Expr/Plain/bootstrap`。
统一接口：`backend.const(vec)`、`backend.input(name)`，以及运算符重载。
验收：（1）三后端跑同一个随机小程序，Mock 结果与 numpy 参考一致，Count 数与预期一致；（2）`test_t3_rotate_ckks.py`：一个 rotate+mul 的小程序在本机走 eva 编译加 CPU CKKS，结果与 MockExpr 一致（容差 1e-5+1e-4|ref|）。CKKS 执行入口复用 `baseline/poseidon_validation.execute`；如果签名对不上，就报告实际签名然后停下。

**T4 Kernel 库（P0，4 天）**
目录 `kernels/`，每个 kernel 是一个函数 `emit(backend, inputs, attrs, layout) -> outputs`，再加一个 `reference(np_inputs, attrs)`：
- `elementwise.py`：add/sub/mul/scale/bias，带布局校验。
- `linear.py`：`diag`（Halevi-Shoup 对角线法）、`bsgs`（n1 默认取 √n 向上到 2 的幂）、`rowsum`（逐行乘再旋转求和）。
- `conv2d.py`：`tap`（每个卷积核位置旋转一次，乘掩码权重再累加）和 `mpcb`（移植 `dacapo/python/poly/poly/MPCB.py` 的多路复用打包卷积，改成通过 backend 发 op，不依赖 torch module）。BN 折叠在 IR 层完成：`bn=前缀` 把 γ/β/μ/σ 并入卷积权重和偏置。
- `pool.py`：avgpool（旋转求和加缩放），全局平均池化。
- `activation.py`：`cheb`（Chebyshev 加 Paterson–Stockmeyer）和 `minimax_comp`（复合 minimax 符号函数，系数取自 `poly/data/*.txt`，与 `poly/Func.py` 的 HE_ReLU 一致）。定义域先归一化到 [-1,1]。relu = x·(1+sign(x))/2。
- `layout.py`：flat/mux/rep 的 slot 索引映射，`relayout` 用掩码加旋转实现。
验收：每个 kernel 在 MockExpr 下与 `reference` 差 ≤1e-9（exact 系数路径），并用随机形状做 property test；`mpcb` 卷积在 ResNet20 第一层形状上与 torch conv2d 一致。

**T5 通用模型导入、分解与分析（P0，3 天；前端 A 和 ResNet20 先做，其余在 M1 后）**
- `intake/graph.py`：PT-Graph 数据结构 `Node(id, op, inputs, outputs, attrs, src)`、`Tensor(name, shape, dtype, kind=input|weight|act)`、`Graph(nodes, tensors, weights: dict[str, np.ndarray], inputs, outputs)`，加 `to_json/from_json`；核心算子表 `intake/core_ops.py` 按 §2.2 列出，每个算子写明属性 schema。
- `intake/torch_frontend.py`（前端 A）：`import_module(module, example_inputs) -> Graph`。先用 `torch.export.export`，失败就回退到 `torch.fx.symbolic_trace`。逐个 `call_module/call_function/call_method` 映射到核心算子；映射表放在 `intake/torch_mapping.py`，覆盖 `nn.*` 模块、`torch.*`/`F.*` 函数、Tensor 方法；`get_attr` 提取为公开权重。遇到不支持的节点抛 `UnsupportedOp(op, target, stack_trace)`，并汇总全部不支持项，不在第一个处停下。PyTorch 用现有 intake 使用的 Nix 环境（入口见 `scripts/setup_agent.py` / `baseline/agent_setup.py`）；找不到就报告然后停下。
- `intake/legacy_frontends.py`（前端 B/C）：调 `baseline/model_import_cli` 和 `baseline/pytorch_model_capture.capture`，把它们的输出图转换成 PT-Graph。只做格式转换。
- `intake/decompose.py`：§2.2 的确定性分解规则，每条规则一个函数，并带 numpy 等价测试。
- `analyzer.py`：`reference_forward(graph, x)`（numpy 实现，以 PyTorch 前向作为对照），`calibrate(graph, inputs[N]) -> {tensor: (min, max, p99.9)}`。
- `intake/coverage.py`：`report(graph) -> {supported, decomposed, needs_composition, unsupported}`。
数据：ResNet20 复用 B 线 workloads 的模型定义、权重和 CIFAR-10 校准/测试集，只读。
验收：（1）ResNet20 经前端 A 导入后，`reference_forward` 与 PyTorch 前向逐元素差 ≤1e-9，top-1 与 91.73% 一致（±0.1pp）；（2）30 个单算子/小模块的 PyTorch 用例（每个核心算子至少 1 个）导入并前向一致；（3）M1 之后：§2.2 的 torchvision/timm 模型清单跑 coverage 报告，写入 registry（模型需本地已有或经批准下载）。

**T6 代价模型（P0，1.5 天）**
`cost.py`：`estimate(program, graph) -> Cost`，在 CountExpr 上跑 lowering，汇总：
- 各类 op 计数；
- 不同 rotation key 数；
- Earth op 总数（作为编译规模代理）；
- 关键路径深度、每个 region 的深度；
- 预测 bootstrap 数：refresh=agent 时等于显式 refresh 数；refresh=compiler 时按 depth/可用 level 模拟；
- 预测延迟：Σ count × 每 level 延迟，读 OperatorSpec v2 的 `latency_us_by_input_level`，路径从 `baseline/compiler_configuration.py` 当前配置里取；
- 明密文数量和内存估计（系数容量公式见 Qwen BOOTSTRAP_LEVEL_AND_MEMORY_REPORT §5）。
验收：MLP、LeNet 上预测的 op 计数与 T1 实测 Earth op 数相差 ≤5%（CSE 前口径）；registry 写入预测值与实测值对照。

**T7 规则规划器与搜索基线（P0，1.5 天）**
`planners/rule.py`：
- 固定默认决策：conv 用 mpcb，linear 用 bsgs；激活取 minimax_comp(15,15,27)，定义域 = 校准范围 ×1.2；
- 每个激活前若 level 不够就放 refresh；
- 输出 PT-DSL 文本，经 printer 打印。

`planners/search.py`：在同样的旋钮空间里做贪心逐节点坐标下降。目标是最小化预测延迟，约束是 MockExpr 校准精度下降 ≤1%；评估次数上限与 Agent 相同。
验收：两种规划器在 MLP、LeNet、ResNet20 上都能产出可 parse 且 typecheck 通过的程序；ResNet20 的 MockExpr 近似精度 ≥ 原模型 −1%。

**T8 Lowering 与外部评测（P0，2 天）**
`lower.py`：`trace(program, graph, out_dir)`，用 HcBackend 跑 kernel，调 `hc.save` 得到 `.mlir`，并写 `compile_config.json`（pipeline 和 flags）。
`evaluate.py`：`run(out_dir, stages=[compile, plan_metrics, ckks])`：
- 编译复用 T1 的计时封装；
- ckks 阶段复用 T3 的执行入口；
- 输出三层误差：近似误差 / exact-Mock 对参考 / CKKS 对 Mock，容差维持 1e-12 和 1e-5+1e-4|ref|；
- 汇总写入 registry。
CLI：`python3 -m pt_agent.run --model <name> --planner rule|search|agent --refresh compiler|agent --stages ...`
验收：MLP 用 rule 规划器在本机完整跑通 compile 和 CKKS 并通过容差。

**T9 LLM Planner Agent（P0，3 天）**
`planners/agent.py` 和 `prompts/`：
- LLM 调用复用 `baseline/deepseek_provider.HTTPSTransport` 和现有凭证加载器（不打印凭证）。模型用 `deepseek-v4-pro`，`max_tokens` 8192。
- 系统 prompt 固定为四部分：§2.1 语法、ops_spec 表、两份完整样例（MLP 和小 CNN，不用 ResNet）、规则。
- 用户消息包含：图摘要表（节点、算子、形状、校准范围）；每个节点各候选旋钮的预计算代价表（由 T6 对单节点估计得到）；资源约束（slots、l_max）。
- 循环：LLM 输出 PT-DSL（放在 ```ptdsl 代码块里）→ parse/typecheck → cost → Mock exact（8 个校准输入）→ Mock approx（256 张校准图的精度）→ 把结构化反馈 JSON（错误、代价汇总、各 region 深度、开销最大的 5 个 op、精度）回传给 LLM。
- 停止条件：（检查全通过，且连续 2 轮预测延迟改进 <5%）或达到 6 轮。每轮把 prompt、响应、反馈、token 数、耗时写进 `trace.jsonl`。冻结时写 `freeze.json`：最终程序、第一次全通过的轮次、总轮次。
- 外部 CKKS 和测试集结果永不进入循环。
验收：MLP、LeNet 各跑 3 次种子，都能在 6 轮内冻结出通过检查的程序；ResNet20 至少 1 次成功。

**T10 基线接入（P0，1 天）**
`baselines.py`：
- B1 直接 LLM：调现有 `scripts/agent_delivery.py` 路线，单次生成 Hecate，原契约不变；每个模型 N=5 次。
- B2 朴素确定性 lowering：fx_to_hecate / unified lowering。
- B3 rule（T7）。
- B4 专家手写：Dacapo examples。
- B5 search（T7）。
全部输出到同一套 evaluate。
验收：MLP 的 5 种方法各产出一行 registry。

**T11 模型集与开发/封存划分（P0，1 天）**
`models/zoo.py`：
- 所有模型都以 PyTorch `nn.Module` 定义，经前端 A 导入。
- 开发集：MLP、LeNet-5、ResNet20。
- 封存集：AlexNet、VGG-small、MobileNet、SqueezeNet（结构参考 `poly/models/*.py`，改写成标准 `nn.Module`）、ResNet32/56、MobileNetV2 和 ShuffleNetV2（CIFAR 尺寸；codex 记录里提到已有候选资产，先用已有的）、一个 Transformer encoder block（LayerNorm、softmax、GELU）。
- 没有训练权重的模型用固定种子生成合成权重，BN 统计量也从同一种子派生（需用户批准新权重，见 §7）。
- 封存规则：T9 的 prompt 和代码打 tag `pt-agent-freeze-v1` 以后，才允许跑封存集；registry 里标注 split。
验收：zoo 中每个模型的参考前向都能跑；封存集在 freeze tag 之前调用时直接报错。

**T12 Agent 刷新模式（P1，1.5 天，第 1 周先做 spike）**
spike：写一个含 `hc.bootstrap` 的小程序，跑 `hecate-opt --pars --ckks-config ... --runtime-plan-*`，确认 RuntimePlan 里保留了 Boot、scale 合法。正式实现：lower 时把 `refresh` 转成 `hc.bootstrap`，编译用 `--pars`；用 T1 对比 `--dacapo` 和 `--pars` 的编译时间与 bootstrap 数。spike 失败就报告 pars 的实际行为，然后停下。
验收：ResNet20 两种模式都能编译（在服务器上），registry 有对比行。

**T13 长尾算子组合（P0，M1 后，2 天，支撑通用性）**
coverage 中 `needs_composition` 的节点交给 Agent：Agent 在 PT-DSL 里用宏把它们组合成已有算子（例如 hardswish = x·poly(...)、channel_shuffle = relayout、maxpool = 近似 max 树）。每个组合宏在 MockExpr 上用随机输入（取值在校准范围内）与 PyTorch 参考比较，exact 部分 ≤1e-9，近似部分报告误差。typecheck 把这类节点标为 `composed`。验收：封存集里的 Transformer block、MobileNetV2、ShuffleNetV2 的全部节点都是 supported、decomposed 或 composed 之一。

**T14 实验总控与作图（P0，1.5 天）**
`experiments/run_all.py`：按"方法 × 模型 × 种子"矩阵生成本机任务和服务器任务，失败也写进 registry。`experiments/figures.py` 从 registry 生成论文图表（§6.5 清单），输出 PDF 和 CSV。

---

## 5. 时间线（摘要 11/11，全文 11/18）

| 日期 | 工程 | 论文 |
|---|---|---|
| 10/21 | R8a2 + R8a3（✅ 已完成） | Intro、Motivation 初稿 |
| 10/22–23 | R8b（实现 + 离线测试） | Design：PT-DSL、ATen 前端、kernel |
| 10/24 | R8d 组合 minimax ReLU | |
| 10/25 | R8c（实现 + 离线测试） | Design：代价模型（与编译器一致） |
| 10/26–27 | R8e 精调工具 + 4 个模型的真实运行 → 用户决定冻结（10/27 不变） | Design：Agent 循环（LLM 结构决策 + 搜索精调） |
| 10/23–24 | R8f/R8f2 张量验证与 flat 尾槽修复 | |
| 10/24 | S0 编译器流水线验证；R8h/R8h2 精调触发规则消融；R8h3 恢复 v1 默认 | |
| 10/24–27 | U1 模型描述框架（PyTorch、HuggingFace、ONNX） | Related Work；Evaluation 的实验设置 |
| 10/28–31 | U2 多密文布局 | Design：布局与 kernel 库 |
| 11/01–05 | U3 CNN 算子 + 5 个 DaCapo 基准 | Evaluation 的已有结果 |
| 11/05–10 | S1 DSL 管理模式（复现编译器） | 摘要 v1；Design 增加"两种管理模式" |
| 11/10–13 | R9b 三组对照与消融 | 摘要 11/11 提交 |
| 11/13 以后 | U5/U4、S2（排不进 DAC 的写入未来工作） | 11/14–16 写全文 |
| 11/17–18 | — | 内部审读，**11/18 提交** |

风险（10/23 修订）：GPU 暂缓，论文以编译器给出的自举数、预测延迟和仿真精度为主指标。U1–U3、S0–S1 和 R9b 都要占用 Nix，只能串行，排期很紧：R9b 被推到 11/10–13，摘要只能用已有数据；任何一轮超期，就先砍 S1（保留 S0 的结论），保证 U3 和 R9b 进论文。

---

## 6. 论文规划（DAC，正文约 6 页）

### 6.1 题目（候选）
- *PT-Agent: Performance-Aware FHE DSL Generation with LLM Agents for Scalable Encrypted Inference Compilation*
- 中文：面向 FHE 编译器的 Agent 驱动性能感知 DSL 生成

### 6.2 核心论点
现有 FHE DSL 只描述计算：无法表达 rescale/level/刷新意图，写起来也冗长。所有优化压给 MLIR 层，在"无优化信息"的大图上搜索，导致编译慢、占内存，甚至编译失败。把布局、算法、近似和刷新决策前移到 DSL 生成阶段，由带代价模型的 Agent 自动完成，编译器只做合法化和轻量的 scale 管理，就能实现大模型的高效端到端编译。

**Agent 内部的分工（10/21 定）**：LLM 擅长离散、结构性的决策（近似方法、布局与算法、宏结构、刷新位置、长尾算子组合），但不擅长在几十个连续旋钮上做细致的数值折中；确定性搜索恰好相反，它能在给定结构上把数值旋钮压到最低，但无法提出新结构。所以 Agent = LLM 提结构 + 精调工具压数值，两者都只依赖与编译器一致的代价模型和验证集，循环中不编译、不加密。R9 用"只用 LLM / 只用搜索 / LLM + 搜索"三组对照证明二者互补。

### 6.3 贡献
1. **刻画**：用数据说明 Hecate/Dacapo 前端在完备性（缺 rescale/modswitch/level，bootstrap 被删除）、易用性（ResNet20 约 1.7k 行）、编译可扩展性（超时、BypassDetection 爆炸、启发式之间自举数差 6–9×）三方面的问题。
2. **PT-DSL**：一个 FHE 感知的高层张量 DSL，把布局、算法、近似、刷新和 level 预算作为一等注解；经可信 kernel 库 lower 到 Hecate，编译器不改。
3. **工具化 Agent**：输入是通用的 PyTorch 模型，经规范化的 PT-Graph 和确定性分解后，由 LLM 做结构决策、由精调工具做数值旋钮的贪心搜索，二者共用与编译器一致的代价模型（预测自举数与真实编译完全一致）和验证集闭环，生成 PT-DSL；长尾算子由 Agent 组合并经数值验证。
4. **评测**：开发集和封存集共约 10 个模型。对比直接 LLM、朴素 lowering、规则、搜索和专家手写，报告编译时间、内存、自举数、延迟、精度和生成成功率。

### 6.4 Motivation：问题、现有证据与扩展实验（10/22 修订）

没有本 Agent 时的问题：P1 每个模型都要专家手写 FHE 程序；P2 决定性能的 DSL 层决策（近似方法、阶数、定义域、布局、刷新）只能人工做，编译器管不到；P3 把优化全部压给编译器，大模型编译慢甚至失败；P4 近似定义域靠人估，越界会发散；P5 直接让 LLM 写 FHE 代码，规模一大就失败；P6 真实编译和加密运行太贵，放不进调优循环。

现有证据大多只来自 ResNet20 一个模型，需要沿"模型、配置、编译策略、重复次数"四个方向扩展：

| 实验 | 证明 | 内容 | 成本与地点 |
|---|---|---|---|
| E1 设计空间扫描 | P2 | 每个模型随机采样约 200 种 DSL 配置，用代价模型算延迟、在 200 张留出图上测精度，画"延迟–精度"散点；每个模型再抽 10 个配置真实编译以确认预测 | 不调 LLM；本机约 5–6 小时 |
| E2 编译负担 | P3 | Dacapo 的 7 个专家 DSL（MLP/LeNet/ResNet/AlexNet/VGG16/MobileNet/SqueezeNet）及朴素 DSL 版本 × 3 种放置策略，记录编译时间、峰值内存、自举数、是否超时；加上已有的 Qwen 数据 | 不调 LLM；大模型需要服务器；合成权重需批准 |
| E3 手写成本 | P1 | 7 个模型 × 4 种写法的代码行数，以及"必须人工做的决定"的个数 | 零成本 |
| E4 定义域安全 | P4 | 所有有权重的模型：训练图与留出图的激活范围比分布；按训练图定义域、不同余量下的测试精度和发散率曲线 | 不调 LLM；本机数小时 |
| E5 LLM 直接生成 | P5 | B1（直接写）和 B1+（写完再测试修复，FHECrafter 式），4–5 个模型 × 5 次，统计各级通过率并分类失败原因 | 约 80 次 LLM 调用 |
| E6 反馈成本与预测准确性 | P6 | 成本表（编译、加密运行、模拟、预测）；约 40 个程序的"预测 vs 实测"散点，并与惰性模型对比 | 来自 E1、E2 |
| E7 文献旁证 | 全部 | DaCapo、FHE-Agent、FHECrafter、FHE-CodeEval、AutoFHE 等论文中报告的相关数据，写作时逐条核对原文 | — |

R9 拆成两轮：**R9a**（E1、E3、E4、E6，不调 LLM，冻结后立即进行）和 **R9b**（三组对照 + E5 + 消融）。E2 的大模型部分打包成服务器作业。

### 6.5 评测设计（RQ 与图表）
- RQ1 生成可靠性：每个模型 5 次种子，统计 parse / Mock / 编译 / CKKS 通过率，开发集和封存集分开报告；同时报迭代轮数、token 数、耗时。
- RQ2 编译效率：编译时间、峰值 RSS、Earth op 数、bootstrap 数，覆盖各方法 × 各模型，并对比 refresh=agent+pars 与 compiler+dacapo。
- RQ3 执行性能与精度：预测延迟（OperatorSpec）和实测 CKKS 延迟（能跑的子集）；测试集精度（Mock 近似模型，10000 张）；CKKS 误差。
- RQ4 **LLM 与搜索的分工（核心对照）**：同一套旋钮、同一评估预算（验证集评估次数）下比较三组：
  - **只用 LLM**：精调工具关闭，其余相同；
  - **只用搜索**：不调用 LLM，从规则规划器的程序出发，精调工具 + 方法选择的穷举/贪心（B5）；
  - **LLM + 搜索**：完整 agent。
  报告每组的预测延迟、自举数、验证/测试精度、评估次数、LLM 调用次数和墙钟时间；开发模型和未见模型分开报告。预期：只用搜索受限于规则给出的结构（例如不会换近似方法或重写宏结构），只用 LLM 数值旋钮压不到底，LLM + 搜索两者兼得。
- RQ4b 其他消融：去掉代价反馈 / 去掉热点反馈 / 去掉验证反馈（只给错误码）/ 思考模式开关 / 去掉 kernel 库（等于 B1）。
- RQ5 通用性与易用性：torchvision/timm 常用模型的导入比例和节点覆盖率（supported / decomposed / composed / unsupported），以及 DSL 行数。
- 相关工作定位：
  - Orion（ASPLOS'25）：PyTorch→FHE，自动打包与自举放置，ResNet-20 比此前快 2.38×，但规则固定在编译器内部；
  - FHE-Coder（ICLR'26）：LLM 写 FHE 代码，侧重安全参数；
  - FHE-Agent（arXiv 2511.18653）：LLM 搜 CKKS 配置，基于 Orion；
  - Hecate/ELASM/DaCapo：编译器侧的 scale 与自举管理；
  - Fhelipe、HEIR：其他编译器。
  本文的区别：在 DSL 生成阶段做性能规划，减轻**现有** MLIR 编译器的负担，且编译器不改。
  - FHECrafter（Yu、Xu、Li，ICS Workshops'26，DOI 10.1145/3774895.3812197，短文）：多 agent（规划、原语选择、一致性检查、测试驱动迭代）把张量计算翻译成 OpenFHE 代码，并调 CKKS 参数，侧重正确性。同组还有 FHE-CodeEval（EMNLP'26 基准）；
  - FHE-Coder（ICLR'26）/ TFHE-Coder：自然语言 → 安全的 FHE 代码；
  - AutoFHE（USENIX Security'24）：逐层混合阶数的多项式激活搜索，减少自举。
  定位（10/21 对比结论）：LLM 不写密文层代码，也不只调编译器参数，而是在 DSL 层做性能规划；循环中不编译、不加密，依靠与编译器一致的代价模型；输入是通用 PyTorch 模型，目标编译器不改。
  R9 必须加入的对照：（1）三组对照"只用 LLM / 只用搜索 / LLM + 搜索"（RQ4），其中"只用搜索"即同一套旋钮、同样评估预算的无 LLM 搜索基线（B5）；（2）"LLM 写 Hecate 代码 + 测试驱动修复"（FHECrafter 式的 B1+）。
  与 AutoFHE、FHE-Agent 的区别要在论文中写清：AutoFHE 只有搜索、只调阶数；FHE-Agent 由 LLM 直接调数值配置并在循环中跑加密推理；本文由 LLM 定结构、搜索定数值，循环中只用与编译器一致的预测。

### 6.6 章节与篇幅
1 Intro（0.75 页）· 2 Background & Motivation（1 页）· 3 Design（2.25 页；DSL 作为 Agent 的输出语言并入此章：总览 / 问题形式化 / 模型导入 / DSL 与 lowering / 代价模型 / 规划循环 / 长尾算子）· 4 Evaluation（1.75 页）· 5 Related Work · 6 Conclusion（合计 0.25 页）。论文 LaTeX 不放在项目里，在 Overleaf 维护（本地备份 `/home/lhohy/pt_agent_paper/`）。

---

## 7. 需要你决定或协调的事
1. **服务器**：W2 初之前需要拿到一台大内存机器跑编译（ResNet20 及以上），另外需要 GPU，或借用 gy 的 bootstrap48 CPU 回调来跑带自举的 CKKS。拿不到的话，论文中大模型只报告编译和预测代价。
2. **磁盘**：`poseidon-work` 已用 62 GB，上限 64 GiB，本机空闲 25 GB。新结果需要单独的 `PT_RESULTS` 目录和额度；旧结果是否归档压缩由你决定（计划里不删任何东西）。
3. **合成权重**：封存模型需要合成权重，按 AGENTS 规定要你批准。
4. **规则更新**：把 §3.2 的内容写进 `/home/lhohy/Code/Poseidon/AGENTS.md` 开头。旧线和 B 线的规则保持不变。
5. **每轮转达**：codex 每轮汇报后，在 Claude 会话里说"Rn 做完了"或贴出阻塞内容，Claude 审查后给出下一轮 spec。
6. **refresh=agent 模式**：要显式写 bootstrap，这与 codex 旧设定"Agent 不写 bootstrap"相反。它不改编译器（用现成的 `--pars`），我建议作为论文的核心对比项。

## 8. 验证方式
- 每轮 spec 自带 unittest 和完成标准。Claude 审查时会重新跑一遍本轮测试，对照 spec 检查 diff，结果记入 STATUS.md 的轮次表。
- 端到端冒烟（本机）：`python3 -m pt_agent.run --model mlp --planner agent --refresh compiler --stages compile,plan_metrics,ckks`，要求编译通过、CKKS 满足 1e-5+1e-4|ref|，registry 新增 1 行。
- **M1 ResNet20**：`python3 -m pt_agent.run --model resnet20 --frontend torch --planner agent --refresh compiler|agent --stages compile,plan_metrics,ckks`，逐条满足 §4.0 的 1–5。
- 通用输入：30 个单算子 PyTorch 用例通过导入加前向一致性测试；M1 之后跑模型清单的 coverage 报告。
- 封存集：freeze tag 之后一次性跑完，所有失败照实记录。
