# U1：模型输入框架（原样的模型 + 原样的输入样本，其余全部自动推断）

用户要求（2026-10-23/24；10/24 两次修正设计，以本版为准）：

**原则**：用户只提供他平时运行明文模型时本来就会用到的东西。不写任何描述文件，也不指定形状、归一化之类的参数。

```
pt-agent --model ./qwen2.5-0.5b             --inputs ./prompts.txt
pt-agent --model ./vit-cifar10              --inputs ./images/
pt-agent --model ./vgg16.onnx               --inputs ./batch.npz
pt-agent --model my_module.py:build_model   --inputs ./x.npy
```

本轮约束：
- 不调用 LLM。新增的代码、实验和测试不读取报告集；既有回归测试按 AGENTS.md 的例外执行。
- 不修改编译器和运行时。
- 冻结 Agent 的决策逻辑不变：**现有 4 个模型改走新入口后，所有数字逐位不变**。

## 0. 状态与修订记录

- §1（独立的 intake 环境）**已完成**：intake-2.0.1-cpu，29 个 wheel，129.3 MiB；protobuf 4.25.x，huggingface-hub 0.24.x；主环境前后 pip freeze 完全相同。
- 前两版 spec 中的手写 `ModelSpec`、`--data`、`--normalize`、`--shape`、`--seq-len`、`--task` **全部作废**，以本版为准。
- 上一版执行时的停止项（HF ResNet 的 `aten.max_pool2d_with_indices` 被 ATen 前端拒绝），由 §5.3 处理。
- 现有代码中需要替换的硬编码：
  - `models/zoo.py`：`adapt` 按 `name=='mlp'` 展平；`get` 按名字选择构造方式和归一化常量（实际上 4 个模型都用 ImageNet 的均值/方差）。
  - `planners/agent_tools.py`：`build_context` 写死了 `heldout_set` 和 `MEAN/STD`；`validate` 用标签算 argmax top-1。
  - `planners/confirm.py` 写死了 `confirm_set`。
  - `run.py` 写死了 `calibration_set`、`report_test_set` 和 argmax。

## 1. 环境（已完成）

intake 环境只用于读取 HF 和 ONNX 并导入成 PT-Graph，导入结果通过 `Graph.save/load` 加上 §2 的 `model_info.json` 交给主环境。主环境不导入 transformers 和 onnx。

## 2. 内部统一的模型信息 `ModelInfo`（自动生成，用户看不到）

`pt_agent/models/info.py`（新增），完全由 §3、§4 根据模型和输入样本生成，保存为运行目录下的 `model_info.json`，只用于记录和复现：

| 字段 | 内容 |
|---|---|
| `source` | torch / hf / onnx，以及原始路径或 Hub 名 |
| `graph` | PT-Graph（结构与权重） |
| `input_handling` | 客户端处理方式：HF 为"调用仓库的 processor/tokenizer"（记录类名和调用参数，不复制归一化数值）；有 tokenizer 时记录"embedding 查表在客户端"；ONNX/PyTorch 为"原样张量" |
| `inputs` | 每个加密输入的名字、kind（image/tokens/features/mask）、静态形状（batch=1）、dtype |
| `sequence` | 只用于变长输入：选定的长度、补齐与截断规则、分档方案（§4.2） |
| `task`、`metric`、`threshold` | 由输出头决定（§3.4），有无标签时的阈值口径（§4.4） |
| `split` | 样本划分方式、种子、每个划分的样本下标（§4.3） |
| `assumptions` | 自动选择的地方，例如选定长度、截断比例 |

之后的校准、验证、精调、确认和报告，都只读 `ModelInfo` 和样本，不再按模型名或数据集名分支。

## 3. 读取模型与推断（`pt_agent/intake/probe.py`、`hf_frontend.py`、`onnx_frontend.py`，新增）

`--model` 的识别顺序：
1. 以 `.onnx` 结尾 → ONNX；
2. 目录中有 `config.json`，或者是 Hub 名 → HF；
3. 形如 `path.py:callable` 或 `module:callable` → PyTorch。

### 3.1 HuggingFace（intake 环境）

- 读 `config.json`：`architectures`、`model_type`、层数、`hidden_size`、`vocab_size`、`num_labels` 或 `id2label`。
- 权重：`from_pretrained(path, local_files_only=True)`，读取 safetensors 或 bin，不联网。
- 预处理：**直接调用仓库自带的** `AutoImageProcessor` 或 `AutoTokenizer`，与 HF pipeline 的用法一致，不另外读取或复制归一化参数。processor 的输出（`pixel_values` 或 `input_ids`、`attention_mask`）就是模型输入。
- 有 tokenizer 时：tokenize 和 `aten.embedding` 查表都在客户端，加密输入是 embedding 之后的张量。
- 与输入无关的张量（位置编码、RoPE 的 sin/cos、causal mask 的上三角部分）在导入时折叠为常量。与样本长度有关的 attention_mask 按 §4.2 处理。

### 3.2 ONNX（intake 环境）

- 结构来自 `graph.node`，权重来自 initializers，支持外部数据文件。
- 输入名字和 dtype 来自 `graph.input` 中不是 initializer 的项；**具体形状以用户样本为准**（batch 固定为 1），再用 onnx 的 shape inference 补全中间形状。
- 用户样本就是送进 onnxruntime 的张量，预处理已在用户那边完成，系统不做任何归一化。

### 3.3 PyTorch（主环境，沿用现有的 ATen 前端）

- 调用 `callable()` 得到 `nn.Module`。用户样本就是送进 `module(x)` 的张量，形状以样本为准（batch=1）。
- 不再有 `adapt`：MLP 的用户本来就送入 `(N, 3072)` 的张量。
- 现有 4 个模型改为 `pt_agent/models/builders.py` 中的 4 个 callable，只负责构造模块、加载权重，不带归一化等元数据。`zoo.get` 保留为兼容接口，内部改走新入口。

### 3.4 任务类型与指标（由输出头自动决定）

| 判据 | task | 指标 |
|---|---|---|
| HF 架构名以 `ForImageClassification` 或 `ForSequenceClassification` 结尾；或 ONNX/PyTorch 输出形状为 `[1, C]` 且 C ≥ 2 | classification | top-1（§4.4） |
| HF 架构名以 `ForCausalLM` 结尾；或输出形状为 `[1, L, V]`，且 V 等于 tokenizer 或 embedding 的词表大小 | causal_lm | 下一个 token 一致率加 logits 误差 |
| 其他 | regression | 输出误差 `max|Δ| / max|ref|` |

判定依据写入 `model_info.json`。

## 4. 输入样本（`pt_agent/data/samples.py`，新增）

### 4.1 读取

`--inputs` 支持以下几种：
- 目录：图片（png/jpg）、文本（txt/jsonl，每行一个样本）、张量（npy/npz）。图片的标签来自子目录名（ImageFolder 约定），没有子目录就视为无标签。
- 单个文件：`.txt`/`.jsonl`（每行一个样本）；`.npy`（第一维是样本）；`.npz`（键名与模型的输入名对应；可选的 `labels` 键作为标签）。
- jsonl 可以带 `label` 字段。

按模型类型交给 HF processor/tokenizer，或者原样作为张量。

### 4.2 静态形状（由系统决定，用户不需要知道）

- **batch 一律为 1**，FHE 程序逐样本运行。
- **形状固定的输入**（图像、固定尺寸的张量）：直接用样本的形状。如果样本之间形状不一致，报错并列出不同的形状。
- **变长输入（文本序列）**，Claude 的方案如下，参数写在 `pt_agent/config.py` 中，作为系统行为，不是用户参数：
  1. 用 tokenizer 统计校准样本和留出样本（不看确认集和报告集）的长度分布。
  2. **单一长度（默认）**：L 取 8 的倍数中，不小于长度 p95 的最小值，并且不超过 `max_position_embeddings`。
  3. 不足 L 的右侧补齐（pad token；没有 pad token 时用 eos），同时生成 attention_mask。causal LM 也在右侧补齐，causal mask 保证补齐位不影响真实位置。
  4. 超过 L 的截断：分类模型保留开头（含 CLS），causal LM 保留末尾 L 个 token（预测依赖最近的上下文）。截断的样本数与比例写入报告。
  5. **分档（自动触发）**：在单一长度下，如果截断比例超过 5%，或者 p95/p50 > 4（大量补齐，造成浪费），就改为最多 3 档。档位边界取 8 的倍数，按长度分布的分位数选择（例如 p50、p90、p99 对应的 8 的倍数），每档单独生成一个程序，样本按长度进入最短的能容纳它的那一档。各档共享 Agent 的近似决策：以最长一档的程序为准，其余档复用同一组方法和阶数，但各自校准定义域，各自验证。档位方案写入 `model_info.json` 和报告。本轮只实现并测试长度选择与分档规则，各档的程序生成在 U4 中接入。
  6. **attention_mask 作为额外的加密输入**（kind=mask），不作为明文元数据，避免把样本的真实长度暴露给服务器。
  7. **causal LM 输出全部 L 个位置的 logits**，客户端按真实长度取最后一个真实位置，服务器不做依赖数据的索引。

### 4.3 划分（全部来自用户样本）

- **规则**：按排序后的样本顺序，用固定种子 20261024 打乱。先取 10% 作为校准，其余按 留出验证 : 确认 : 报告 = 1 : 1 : 8 划分，与现有 CIFAR 测试集的 1000:1000:8000 同比例。
- 每个划分至少 20 个样本，因此总数至少 223 个。不足时报错："至少需要 223 个输入样本，当前 N 个"。
- 划分清单写入 `model_info.json` 的 `split` 字段。
- **不能用随机输入代替用户样本做校准和验证**（E4：未见的真实数据可能比校准范围大 36%）。§7 中"与原框架前向一致"的导入核对可以使用随机输入，因为它不是校准，也不进入任何决策。
- **现有 4 个模型的逐位等价**：冻结结果是在 CIFAR 的既有划分（校准取自训练集的选图，留出 test[9000:10000]，确认 test[8000:9000]，报告 test[0:8000]）上得到的。为了逐位可比，测试工具提供一个**仅供回归测试用**的内部样本源 `legacy_cifar`，严格复现这套既有划分和数组：用 ImageNet 均值/方差归一化，mlp 送入展平后的 `(N, 3072)`，与模型平时接收的张量相同。它不是面向用户的选项，CLI 不暴露。另外，用同样的 CIFAR 张量走通用划分规则跑一遍（§7.5），证明通用路径可用，这一项不要求与冻结数字相同。

### 4.4 指标与阈值

| 场景 | 判据 |
|---|---|
| 分类，有标签 | 沿用现有的 acc_drop_pp ≤ 0.5（发散按错误计入） |
| 分类，无标签 | 与明文模型 top-1 的一致率 ≥ 99.5%（不一致率不超过 0.5%，与 0.5pp 的精度下降上限同口径，因为精度下降不会超过不一致率），发散按不一致计入 |
| causal LM（无论有无标签） | 在真实的最后一个位置上，下一个 token 与明文模型的一致率 ≥ 99%，同时 logits 相对误差 ≤ 1e-2；有标签（即文本中真实的下一个 token）时，另外报告真实准确率，但不作为判据 |
| regression | 输出相对误差 ≤ 1e-2 |

`pt_agent/metrics.py`（新增）实现这几种指标，接口为 `compare(exact_outputs, approx_outputs, labels_or_None) -> dict`。`classification_top1` 在有标签时，输出字段和浮点计算顺序都与现有 validate 完全相同。

### 4.5 接入（本 spec 明确授权修改以下冻结模块）

- `planners/agent_tools.py`：`build_context` 改为接收 `ModelInfo` 和样本划分。留出样本、exact logits 和标签（可能为 None）都来自它们。`validate` 中的 agree、top1、acc_drop 或一致率改由 `metric.compare` 计算，feasible 判据中的精度项改用 `ModelInfo.threshold`。其余逻辑不变：分批、发散重试、rows、worst_acts、max_u、自举输入约束。
- `planners/confirm.py`：确认样本来自划分。
- `run.py`：校准和报告样本来自划分，argmax 改用指标。旧的 `--model <name>` 参数保留，在内部映射到 `builders` 加上 `legacy_cifar`。
- 面向 LLM 的反馈：对有标签的分类模型，反馈的字段和数值必须与现在完全相同，用 `test_r8f2_feedback.py` 验证；无标签时，字段名改为一致率，并相应修改提示词中的描述（列出原文和新文）。

## 5. 导入层

### 5.1 PT-Graph 参考执行器（`intake/graph_exec.py`，新增）

按 `CORE_OPS` 的语义，用 torch float64 逐节点执行。只用于导入核对和覆盖率统计，不参与决策，两个环境都能运行。

### 5.2 覆盖率（扩展 `intake/coverage.py`）

每个模型给出：原始节点数、映射到 `CORE_OPS` 的比例、可以 lowering 的比例、未映射的算子及其次数。

### 5.3 语义与 lowering 能力分离

- `CORE_OPS` 表示语义明确，`KERNEL_OPS`/`HANDLED_OPS` 表示当前能够 lowering。导入不能因为某个算子没有 kernel 就失败（例如 `max_pool2d_with_indices`），lowering 是否支持由 check 阶段报告，并带上 op id 和行号。
- 为验收模型扩充 `CORE_OPS`，每个都给出语义说明，并在 §5.1 中实现。至少包括：
  - `maxpool`；
  - 一元函数并入 `act` 的 fn：rsqrt、sqrt、reciprocal、erf、tanh、pow（常数指数）；
  - `embedding` 只作为客户端边界的记录，不进入加密图；
  - 条件为常量的 `where`/`masked_fill` 折叠为常量；
  - attention_mask 参与的加法或乘法保留为对 mask 输入的运算；
  - `expand` 和 `unsqueeze` 并入 reshape 或 broadcast。
- 真正无法映射的算子（语义不明，或依赖数据的控制流）：导入失败，并列出全部未映射算子。

## 6. 入口（`pt_agent/cli.py`，以及 `scripts/pt-agent` 包装脚本，新增）

```
pt-agent --model <原样模型> --inputs <原样输入样本> [--out <目录>] [--stages ...]
```

- 不提供任何形状、归一化或任务参数。`--out` 和 `--stages` 只控制运行，不描述模型。
- 阶段：`probe`、`import`、`info` 在需要时通过 nixrun 进入 intake 环境；之后的 calibrate、plan、check、mock、count、export、compile、test 在主环境中执行，复用 run.py 的现有阶段。
- `info` 阶段打印推断出的全部信息和 `assumptions`。

## 7. 验收与测试（新增 `tests/test_u1_*.py`）

1. **现有 4 个模型逐位不变**：`builders` 加上 `legacy_cifar` 走新入口，与旧的 `run.py --model <name>` 相比：
   - PT-Graph 的 graph.json 文本和 weights 逐项相等；
   - 校准统计相等；
   - `build_context` 的留出样本、exact logits 和标签 `np.array_equal`；
   - 4 个冻结程序的 validate（mock 和 tensor）、confirm_select 和 estimate 逐项相等；
   - `test_r8f2_feedback.py` 照常通过。
2. **HF 导入**（§8 中的 HF 模型）：
   - 导入成功；
   - 参考执行器与 HF 原 forward（float64）在用户样本上满足 `max|Δ| ≤ 1e-6·(1+|ref|)`，文本模型在 embedding 之后比较；
   - 推断出的 task、类别数或词表大小、输入形状与 config 一致；
   - 预处理调用的是仓库自带的 processor/tokenizer。
3. **ONNX 导入**（§8 中的 ONNX 模型）：
   - 导入成功；
   - 参考执行器与 onnxruntime（float32）满足 `max|Δ| ≤ 1e-4·(1+|ref|)`；
   - 推断出的 task 和形状正确。
4. **跨来源等价**：resnet20 和 mlp 分别从 PyTorch 和 ONNX 导入（`torch.onnx.export`，opset 17，在本机导出）：
   - 在 8 个样本上，两者的参考执行器输出 `max|Δ| ≤ 1e-5·(1+|ref|)`；
   - 规范化之后，op 种类的多重集合相同。规范化规则（BN 折叠、flatten 与 reshape 视为同一种、去掉 identity）写在代码里，并逐条列入汇报。
5. **通用划分**：用 CIFAR 张量的 npz 文件（不带 split 信息）走通用规则：划分可以复现，样本不足时报错，边界情况为 222 个和 223 个样本。
6. **无标签路径**：至少 1 个模型（建议用 §8 的 tiny Qwen2 和 resnet20 无标签版各一个），在没有标签时使用一致率指标，判据按 §4.4。
7. **变长输入**：用文本样本测试长度选择、补齐与截断、分档触发条件，以及 attention_mask 作为输入的导入。
8. **隔离**：主 venv 的 pip freeze 不变；主环境的测试不导入 transformers 和 onnx。
9. 既有 373 项测试全部通过。只有直接依赖被替换硬编码的旧测试，允许按新接口机械调整，断言的数值不能改，并在汇报中逐项列出。

## 8. 测试模型与样本（需要用户批准下载，分两步）

**第一步：只查元数据，然后停下。** 对下列候选，用 Hub API、GitHub 页面或 raw 链接的 HEAD 请求，确认是否存在，列出每个要下载的文件：文件名、大小、sha256（有就给）、URL，以及总量。写入 STATUS 后**停下来等待用户批准**，不下载权重。候选不存在或过大时，给出同类替代并标明。

| 用途 | 候选 | 说明 |
|---|---|---|
| HF CNN | `microsoft/resnet-18` | 已训练的 ImageNet 分类模型，带 image processor，约 47 MB |
| HF Transformer 分类 | 一个已微调、带 tokenizer 的 tiny BERT 文本分类模型（例如 `M-FAC/bert-tiny-finetuned-sst2`）；不存在时用 `hf-internal-testing/tiny-random-BertForSequenceClassification` | 约 20 MB |
| HF LLM（验收） | `hf-internal-testing/tiny-random-Qwen2ForCausalLM`；不存在时用同类 tiny Qwen2 或 LLaMA | 带 tokenizer，含 RMSNorm、RoPE、SwiGLU、GQA，几 MB |
| HF LLM（可选冒烟） | `Qwen/Qwen2.5-0.5B` | 约 1 GB。只做 probe、info 和单个 block 的导入，检查本机内存是否足够；可以单独批准或拒绝 |
| ONNX CNN（非 PyTorch 来源） | ONNX Model Zoo 的 `mnist-12.onnx` | 约 26 KB |
| ONNX CNN | ONNX Model Zoo 的 `squeezenet1.1-7.onnx` | 约 5 MB |
| ONNX Transformer | 在本机把上面的 HF BERT 导出为 ONNX | 不下载 |

输入样本不下载：
- 图片：把 CIFAR-10 图片存成 png 目录（带类别子目录和不带两种），交给 resnet-18 自带的 image processor；
- 文本：用仓库中已有的 Markdown 文档按行切分，作为 prompts 和分类文本。这是真实文本，不是随机输入；
- ONNX 的样本张量（npz）：用 CIFAR 图片在测试工具中按各模型期望的形状做一次用户侧预处理后生成，处理方式写入测试代码。它们只用于导入核对，不进入校准。

下载位置：`<poseidon-work>/platforms/<platform>/cache/pt-agent-models/`，不进入 git。

**第二步（用户批准后）**：下载、记录实际大小和 sha256，然后执行 §2–§7。

## 9. 汇报

- §8 第一步的下载清单，在此处停下；
- 批准后：§7 各项结果、每个模型推断出的 `model_info` 摘要、覆盖率表、未映射或不能 lowering 的算子汇总（作为 U2、U3、U4 的输入）、旧测试的调整清单；
- STATUS.md 中更新 U1 一行。

## 10. 不要做

- 不修改主 venv，不升级 torch 和 numpy；未获批准不下载任何模型或数据。
- 不修改 lowering、kernels、cost、placement、tuner。§4.5 列出的模块只按该节修改；prompts 只改 §4.5 所说的无标签字段描述。
- 不给用户增加任何描述模型的参数；推断不出时报错，说明缺什么，不猜。

## 11. 补充：§5.3 对旧测试的授权调整（Claude 于 2026-10-24 复核 MaxPool 断言停止项）

§5.3 改变了 ATen 前端的语义：导入只看语义，能否 lowering 由 check 阶段报告。所以凡是断言"因为没有 lowering 而在导入时被拒绝"的旧测试，都需要把同一个意图挪到 check 阶段去断言。这属于 §5.3 授权的语义迁移，不是机械调整。

1. **`tests/test_r8a2_aten.py::test_collect_maxpool_and_transposed_convolution`**（ATen 前端，主路径）改为三条断言：
   - `nn.Sequential(nn.MaxPool2d(2), nn.ConvTranspose2d(1, 1, 2))` 用 `frontend='aten'` 导入，仍然抛出 `IntakeError`，因为转置卷积的语义没有映射；`reasons` 中包含 `'transposed convolution'`，**不再**包含 maxpool 的原因。
   - 只含 `nn.MaxPool2d(2)` 的模型可以成功导入，PT-Graph 中有 `maxpool` 节点；参考执行器（§5.1）的输出与 PyTorch 前向一致（float64，`max|Δ| ≤ 1e-12`）。
   - 对这个图生成程序或进入 check 阶段时，报告 maxpool 不能 lowering，并带有 op id 和行号。断言报告中的原因文本包含 "maxpool"。

   原测试的意图（maxpool 目前不能进入 FHE 程序，转置卷积不被支持）全部保留，只是 maxpool 的拒绝点从导入挪到了 check。

2. **`tests/test_r3_intake_errors.py::test_collect_all_unsupported`**（R3 的 fx 前端）**不修改**。fx 前端是旧路径，新入口统一使用 ATen 前端，本轮不改 fx 前端的行为。如果实现中发现必须改动 fx 前端才能满足 §5.3，停下来汇报。

3. 同类情况的一般规则：其他旧测试如果断言"ATen 前端因为没有 lowering 而拒绝导入某个算子"，并且该算子已按 §5.3 加入 `CORE_OPS`，都按第 1 条的方式迁移：导入成功、参考执行一致、check 阶段拒绝。逐项列入汇报。不属于这种情况的旧测试失败，仍然停下来汇报。

完成后继续 §2–§7 的剩余部分：接入完整管线、§7 全部验收和既有 373 项回归。

## 12. 补充：tiny Qwen2 导入与原 forward 不一致的诊断和修复（Claude 于 2026-10-24）

**现象**：ResNet18 和 BERT 的参考执行与 HF 原 forward 完全一致（最大差 0）；tiny Qwen2（182 个节点）最大绝对差 45.86，归一化差 19.1。差值这么大，是语义错误，不是精度问题。

**线索**：Qwen 图中有几类 BERT 和 ResNet 都没有的算子：
- `broadcast` ×4：GQA 的 repeat_kv，即 expand 加 reshape，2 个 KV 头扩展为 4 个；
- `slice` ×8 和 `concat` ×4：RoPE 的 rotate_half；
- `compare` ×1 和 `where` ×1：由 attention_mask 输入生成的注意力掩码；
- `cast` ×14：RMSNorm 中先转成 float32 再转回。

BERT 一致，说明 linear、softmax、LayerNorm、matmul 这些公共部分是对的。错误应当在上面几类算子的语义，或者它们的前端映射中。

### 12.1 诊断（不改任何断言和容差）

1. **逐算子单元对照**：对 broadcast/expand、reshape、slice、concat、neg 或乘 −1、compare、where、cast、带 GQA 形状的批量 matmul、带 −inf 或 dtype 最小值掩码的 softmax，各构造一个最小的 PyTorch 函数，用 ATen 前端导入，再用参考执行器运行，与 PyTorch 在 float64 下比较。要覆盖 Qwen 中实际出现的形状和属性，例如 expand 的目标形状、slice 的维度与步长、cast 的目标 dtype。
2. **逐层对照**：给 HF 模型的各个子模块注册 forward hook，记录 embedding 之后的输入、每一层的 input_layernorm、q/k/v 投影、RoPE 之后的 q/k、注意力输出、o_proj、post_attention_layernorm、MLP 输出、最后的 norm 和 logits。参考执行器按图节点记录中间值，用 make_fx 节点的 `nn_module_stack` 或节点名对应到子模块，找出**第一个**不一致的位置。
3. **检查输入和比较口径**：
   - 两边使用完全相同的 `inputs_embeds`、attention_mask 和 position 信息；
   - HF 的 forward 必须使用 eager 注意力（`attn_implementation="eager"`），避免 SDPA 与参考执行器之间的实现差异混入比较；
   - 两边使用相同的 dtype：整个模型转成 float64，cast 节点按 HF 实际的 dtype 语义执行。
   
   不一致只出现在补齐位置时，也算不一致，补齐位置的输出同样要与 HF 相同；但需要在报告中单独说明。

### 12.2 修复

- 定位到的问题属于本轮自己写的前端映射、`CORE_OPS` 语义或参考执行器时，按 AGENTS.md"自写错误例外"修复。
- 修复必须是**通用的语义修正**，不能针对 Qwen 或特定节点名做特殊处理。
- 每一处修复都补一个 12.1 第 1 条中的单元测试。
- 修复后 ResNet18 和 BERT 仍然保持最大差 0。

如果第一个不一致的位置在 HF 自身，例如 eager 与 sdpa 不同，或者 float64 下 HF 的行为异常，停下来汇报，不调整容差。

### 12.3 继续

tiny Qwen 满足 §7.2（`max|Δ| ≤ 1e-6·(1+|ref|)`）之后，继续完整 CLI 接线、§7 全部验收和既有 373 项回归。汇报中写明根因、修复方式，以及逐层对照中第一个不一致的位置。
