# Python 与高层算子输入

高层图、受限 Python 与可信本地 PyTorch 均先生成统一模型 JSON，再交给 Agent。受限 Python 使用静态解析；可信本地 PyTorch 在隔离进程中执行用户明确授权的代码。

## 高层算子图

格式为 `poseidon-operator-graph-v1`，字段与[统一图](supported-inputs-models-configurations.md)相同。除基础算子外，还支持：

| 算子 | 分解方式 |
|---|---|
| dot、rank-2 matmul | 基础乘法与归约 |
| exp | 显式 Taylor 次数和有效区间 |
| reciprocal | 固定 Goldschmidt 迭代 |
| rsqrt | 固定 Newton 迭代 |
| rms_norm | 平方均值、倒平方根近似和增益 |
| softmax | 指数近似、归约和倒数近似 |
| rope | 向量 split-half RoPE |

近似方法、区间和参数必须在模型中声明。原图与分解后图都要满足节点和容量限制。分解结果、原始图和节点来源会一起保存。

```bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-decomposition-v1/rmsnorm.json
```

默认只检查计划。实际执行参考检查并生成新目录：

```bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-decomposition-v1/rmsnorm.json \
  --write --output "$HOME/poseidon-work/imported-rmsnorm"
```

完成后将输出目录中的 `model.json` 传给 `scripts/agent.py`。

## JSON 空间计算图

选择 `poseidon-operator-graph-v4`；字段沿用高层图的 `id / inputs / constants / nodes / outputs`。空间计算可接矩阵、算术、分支和命名输出。

| 算子 | 输入及参数 |
|---|---|
| resize（1D） | rank 3，[N,C,L]；nearest、nearest-exact、linear |
| resize（2D） | rank 4，[N,C,H,W]；nearest、nearest-exact、bilinear |
| conv_transpose1d | rank 2/3；公开权重 [Cin,Cout/groups,K]，可选公开 bias |
| conv_transpose2d | rank 3/4；公开权重 [Cin,Cout/groups,KH,KW]，可选公开 bias |

resize 的 `attrs` 包含 `size / scale_factor / mode / align_corners / recompute_scale_factor / antialias`；指定 size 或 scale_factor 之一，另一项为 null。linear/bilinear 的 align_corners 使用布尔值；nearest 类使用 null。当前语义对应固定 Torch 2.0.1 的 CPU Float64 连续布局，antialias=false。

转置卷积的 `attrs` 包含 `stride / padding / dilation / groups / output_padding / output_size`。各空间参数采用固定长度列表；output_size 可为 null 或静态形状。按现有资源配置检查图、张量、常量与展开规模。小型计算采用精确公开 Linear 映射；超过矩阵预算时使用公开索引与已有 Conv 的分解。

~~~bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-spatial-v4/resize_1d_linear_0.json \
  --resource-profile general-tensors-v1 --write \
  --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/imported-spatial-example"
~~~

应用可提交 `{"format":"poseidon-model-source-request-v1","kind":"graph","model":graph}`，经 `prepare_source()` 生成统一请求。导入后仍使用原生成、验证和程序包接口。

[三个示例](../../scripts/baseline/cases/operator-spatial-v4/catalog.json)包括 1D linear resize、2D bilinear resize 和分组转置卷积。45 个规则案例已完成真实 Poseidon CPU 编译、数值验证和程序包重载，其中这三包通过新输入、重复执行与缓存复用；192 组次／3,496 值，最大绝对误差 4.861826131685731e-7。[验收记录](agent-spatial-offline-results-r1403.md)。新增空间能力的真实 Agent 生成另行评测。可信 PyTorch 通过 v27 接入空间调用，旧 JSON v1–v3 和 PyTorch 来源版本保持原行为。

## 受限 Python

清单格式为 `poseidon-python-model-v1`，包含 id、files（文件名和 SHA-256）、entry（模块.函数）、inputs、公开 constants 和命名 outputs。

最多 8 个平铺的 `.py` 文件，总源码 64 KiB，每文件最多 4096 个 AST 节点。

支持：

- 纯函数、清单内本地 helper、名称赋值。
- 公开有界 range 循环、列表/元组和公开容器索引。
- 算术、已映射的 torch / F.linear 调用。
- 附带显式近似参数的 exp、rsqrt、reciprocal、softmax。
- 静态命名空间 `poseidon.rms_norm` 和 `poseidon.rope`。

不支持 `nn.Module` 类定义、默认/可变函数参数、递归、动态张量索引、数据相关分支、任意属性或动态调用、文件/网络访问、torch.load、eval/exec、动态导入。近似函数的额外参数是本前端约定，不是普通 PyTorch 的调用签名。

```bash
python3 -B scripts/prepare_model.py \
  --python-manifest scripts/baseline/cases/operator-decomposition-v1/mlp-python.json \
  --write --output "$HOME/poseidon-work/imported-mlp"

python3 -B scripts/agent.py --backend local candidate -- \
  --case "$HOME/poseidon-work/imported-mlp/model.json" \
  --self-test --compiler-configuration poseidon-cpu-eva-w45-v1
```

平台与工作目录环境变量沿用 [README](../../scripts/README.md)。输出目录必须尚不存在。

## 当前示例

| 示例 | 状态 |
|---|---|
| 4→2→2 平方激活 MLP | 受限 Python、可信 PyTorch 捕获、独立参考、Poseidon CPU 密态与程序包重载通过 |
| 小型 RMSNorm | 默认规则基线与真实Agent均完成Poseidon CPU数值验收和重载；Agent修复一次后通过 |
| 单头小型 Attention decode | 默认规则基线完成Poseidon CPU数值验收和重载；真实Agent四次生成均编译失败 |

Attention 示例含投影、RoPE、密文点积、近似 Softmax 和显式 K/V 输入输出；不是完整 Transformer 或任意长度 Attention。

参考检查分别记录展开计算误差和相对原函数的近似误差。通过展开计算的数值检查，不代表近似已达到应用的精度要求。

导出 DSL 包时可通过 `--model-import` 带入原始模型来源，见[程序包接口](agent-component-v1.md)。Python入口的导入验收与JSON整模型的真实Agent生成分别记录。

## PyTorch 捕获与资源配置

可信本地模型可通过显式文件清单捕获为统一图：

```sh
python3 -B scripts/prepare_model.py --help
```

实际子命令以当前启动器帮助为准。导入实现入口为 `scripts/baseline/model_import_cli.py`，参数包括 `--trusted-pytorch-manifest`、`--trust-local-code`、`--model-package` 和 `--resource-profile`。

### PyTorch 来源清单

`scripts/prepare_pytorch_source.py` 将本地文件和静态签名整理为清单。
它读取配置、语法和哈希；模型捕获在后续明确选择的隔离导入阶段执行。

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:1,4 --output logits \
  --resource-profile general-tensors-v1 --manifest ./source.json

python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest ./source.json --trust-local-code \
  --resource-profile general-tensors-v1 \
  --write --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/imported-model"
~~~

`--entry` 是 `模块名.工厂名`。输入按 `--input 名称:维度,维度` 依次声明，
输出按 `--output 名称` 声明；配置工具与导入端使用相同资源配置。
`--source-format latest` 当前选择 v28，生成清单记录具体版本。省略该选项使用 `poseidon-trusted-pytorch-v18`；旧版本可显式选择。

| 参数 | 用法 |
|---|---|
| `--helper` | 同目录的平铺 Python helper；源码合计最多 8 文件、64 KiB |
| `--factory-args`／`--factory-kwargs` | 工厂位置参数与关键字参数，使用 JSON 数据 |
| `--weight` | `state_dict名称=文件.npy`，自动记录 dtype、形状、大小和 SHA-256 |
| `--dtype-policy` | 默认 float64；float32 选择 `convert-float64-v1` |
| `--approximations` | 显式近似配置 JSON |
| `--manifest` | 创建同模型目录的新 JSON；省略时输出配置预览 |

NPY 权重采用 C 顺序，支持 float32、float64、int32、int64 和 bool。
通用配置允许 1–4 个输入、1–4 个输出、rank 1–32；元素、图规模和算子范围沿用当前资源及能力检查。
通过来源导入后，生成的 `model.json` 或应用任务进入同一 DSL 生成与验收流程。

### 可信本地 PyTorch

以清单中的 `nn.Module` 为入口，使用当前锁定的 Torch 版本捕获 FX 图；模块、输入和权重按 CPU float64 eval 语义处理。清单最多 8 个平铺 Python 文件，总源码 64 KiB。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-v1/manifest.json \
  --trust-local-code --write --output "$results/imported-pytorch-mlp"

python3 -B scripts/agent.py --backend local candidate -- \
  --case "$results/imported-pytorch-mlp/model.json" --self-test \
  --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

导入报告包含原模块、高层图和分解图的固定输入比较。程序包来源清单可通过 `dsl_bundle.py export --model-import` 保存；重新加载使用已编译的产物。命令与验收记录见 [Python 导入迁移](poseidon-python-import-migration-r288.md)。

资源配置包括 `legacy-v1`、`medium-v1`、`resnet20-v1`；非legacy分解输出采用 `poseidon-model-graph-v2` 并绑定资源配置。medium已有1024元素物理入口，包含预算内的公开张量、跨块Linear、非对齐concat、内轴stack及指定轴sum/mean；resnet20-v1已有4096元素JSON图的物理IO验收，可信PyTorch4096元素已有代表性数值与重载验收，整网继续推进。当前容量验收使用JSON图与规则基线，516元素PyTorch归约融合模型已完成捕获到密态和包重载；其他较大组合仍有容量／执行阻塞，见[容量报告](poseidon-capacity-tensor-acceptance-r441.md)。

显式多项式近似配置包含 method、basis、domain、coefficients。多项式以原输入为自变量；Chebyshev 系数也不隐含区间映射。原函数误差与近似计算的执行误差分别统计。

本轮实现和验证范围见 [第三阶段进度](stage3-extension-progress.md)。

## 张量与激活输入

算子图可先离线导入，再交给现有 Agent：

```sh
POSEIDON_PLATFORM=aarch64-linux python3 scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-extension-v3/adaptive_avg_pool_2-source.json \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/import-adaptive-pool --write
```

使用新建输出目录。导入产物及参数格式见该目录中的 model.json 和绑定记录。上例的源模型包含高阶 adaptive_avg_pool，导入后由基础 slice、mean、concat 表达。

新增算子的参数和验证范围见 [张量与激活扩展](stage3-extension-progress.md#张量与激活扩展r226)。公开索引接口当前对应 index_select：一维索引沿指定轴选取，完整 torch.gather 的逐元素索引语义仍待实现。


## 跨密文空间算子

medium-v1已验收跨块Conv1d/2d（含分组/深度卷积）、AvgPool1d/2d及代表组合。当前公开张量版本为v7，包含池化流与并行行归约；v5起的文件编码支持重复向量无损压缩。各版本按对应原请求与绑定重建。[范围与结果](poseidon-capacity-spatial-acceptance-r449.md)。当前结果为规则基线，真实Agent验收另计。

## FX 张量调用与中型来源包

sum／mean 支持静态单轴、轴元组、全部轴、keepdim 和 float64；cat／concat／concatenate、stack 支持公开张量列表和静态轴。最终标量输出使用 reshape(1)。dtype转换、out修改和动态轴保留为输入诊断。

medium-v1 来源包沿用同一资源配置进行分解与独立参考复核。三个冻结 PyTorch 案例的范围和已知失败见 [r473](poseidon-stage3-pytorch-acceptance-r473.md)。

~~~sh
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-medium-v1/reduction_fusion.json \
  --trust-local-code --resource-profile medium-v1 --write \
  --output "$results/imported-reduction"
python3 -B scripts/agent.py --backend local candidate -- \
  --case "$results/imported-reduction/model.json" --self-test \
  --unified-chunk-period 256 --unified-guidance explicit-v10 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

该示例使用规则候选；Python／容量入口的真实Agent评测单独执行。

## 矩阵调用与两密文张量广播

可信FX支持torch.dot、torch.mm、torch.bmm、torch.matmul／@及对应Tensor方法，接受静态位置或关键字参数。dot采用匹配的一维向量；mm采用二维；bmm采用批量数相同的三维；matmul采用既有rank-2/3广播。当前dot/matmul分解针对两个密文输入，公开矩阵权重采用Linear。标量返回可用reshape(1)。

9个矩阵来源包、9个广播加减乘案例已通过Poseidon CPU数值验收与程序包重载。广播覆盖二维、四维和period-4跨块；源码与工作量预算保持。示例清单为scripts/baseline/cases/trusted-pytorch-matrix-v1/acceptance.json。导入、规则基线和真实Agent生成分别计分。[最新验收](poseidon-pytorch-broadcast-acceptance-r508.md)。


## 大输入布局

resnet20-v1使用capacity V3，最多16物理输入／输出块，逻辑输入／输出各4096元素。JSON逐项、归约、尾块和四输入四输出已完成Poseidon CPU数值与重载验收；PyTorch4096元素捕获到执行已完成逐项、归约、四输入四输出验收；新增功能真实Agent生成另行验收。[容量范围](poseidon-extended-capacity-acceptance-r518.md)。

## 大输入 PyTorch 与标量组合

resnet20-v1的可信FX入口已验收4096元素逐项计算、3072元素CIFAR形状归约及不同rank的四输入四输出。torch.stack可组合sum/mean产生的零维张量；直接标量输出使用reshape(1)。完整示例、五个来源包及命令见[r535](poseidon-pytorch-extended-acceptance-r535.md)。

ResNet20固定权重的标准ReLU参考已有40张分层图片和4组诊断探针的独立对照；原尺寸stem的Conv→BatchNorm→通道均值诊断已完成密态与重载验收，完整残差块和整网继续推进。

## 通道布局

公开张量v8增加分块BatchNorm和对齐的二次幂后缀sum/mean。CLI使用`--unified-packing capacity-public-tensors-v8`，组件使用`capacity_packing_version`，默认v7保持。旧V2/V3周期为4–256；resnet20-v1显式V4周期为512/1024/2048/4096。[参数与验收](poseidon-channel-layout-acceptance-r547.md)。

大周期与精简编译配置见[r574](poseidon-wide-trace-acceptance-r574.md)。新版布局的代表性结果采用规则候选，真实Agent生成另行验收。


## 参数化 PyTorch 模型

新清单使用 `poseidon-trusted-pytorch-v2`，在原字段上增加工厂参数和浮点策略：

```json
{
  "format": "poseidon-trusted-pytorch-v2",
  "files": {"models.py": "<文件 SHA-256>"},
  "entry": "models.make_mlp",
  "factory": {"args": [3], "kwargs": {"seed": 71}},
  "dtype_policy": "convert-float64-v1",
  "inputs": [{"name": "x", "shape": [1, 4]}],
  "outputs": ["scores"],
  "approximations": {}
}
```

工厂返回 eval 推理模型，参数使用 JSON 数据。工厂和 forward 在无网络的隔离进程内运行。现有 v1 清单继续按原接口处理。

浮点策略：

| 策略 | 处理方式 |
|---|---|
| `strict-float64-v1` | 浮点权重及常量采用 float64，其他浮点类型返回诊断 |
| `convert-float64-v1` | 将 float32 权重及常量明确转换到 float64 计算域，保存转换前类型 |

模型参考和密态比较采用 float64 计算语义。报告单独记录原类型执行是否已比较；当前导入检查验证转换后的模块。公开整数的数值必须能精确表示，整数索引用于公开 index_select。布尔算术返回类型诊断。

返回值可以是 Tensor、静态 tuple/list，或字符串键字典，最多四个张量叶子。清单的 outputs 按返回结构的遍历顺序给这些张量命名。来源包保留容器类型、字典键和张量名称；本地执行同时提供平铺的 `outputs` 和还原结构的 `structured_outputs`。

常见静态形状写法已映射：`x.shape`、`x.size()`、`x.size(axis)`、`view/reshape`、`squeeze/unsqueeze`、C-order `contiguous`。加减乘、Linear 模块及 functional.linear 支持相应关键字写法。形状必须由静态签名确定，各算子继续执行自己的轴和容量检查。

导入示例：

```bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-general-v2/mlp.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-general-mlp"
```

宿主 worker 可以提交导入目录：

```python
task_id = app.submit_import(
    imported_directory,
    options={"chunk_period": 4},
)
result = app.run(task_id, provider)
```

导入来源作为受校验的程序包元数据保存，生成模型仍只接收公开计算图、权重、布局和规则。来源变化会改变缓存身份。已通过的任务沿用 `app.execute(task_id, samples)`，重载程序包即可处理符合签名的新输入。

通用配置 `general-tensors-v1` 允许 rank 1–32，元素、节点、权重和密文块预算沿用最高已有配置。元素容量和计算深度并未随 rank 增加。

## 声明式图权重包

`--model-package` 读取计算图清单和 NumPy 权重。每个权重记录文件名、SHA-256、字节数、形状和 `<f8` dtype；加载器检查数组头、有限值、资源上限及内容哈希。导入生成 original.json、model.json、分解绑定和双参考报告。权重包完整字段见 `model_weight_package.py`。

```bash
python3 -B scripts/prepare_model.py \
  --model-package weights/manifest.json \
  --output imported-model --write
python3 -B scripts/agent.py --backend local candidate -- \
  --case imported-model/model.json --prepare \
  --public-weights descriptors-v1 --unified-guidance explicit-v10
```

medium-v1、resnet20-v1 和 general-tensors-v1 可显式选择权重描述符。原权重和可信绑定保留在执行端，生成上下文只包含形状、元素数量、内容哈希及布局对应关系。不同内容产生不同请求身份。候选继续使用预绑定常量名。

声明式图使用该权重包；PyTorch 工厂通过下述 v3 清单加载独立 state_dict 权重。可信请求上限 4 MiB，生成上下文和候选上限各 128 KiB，其他逻辑及物理资源限制仍参与判断。

本轮真实 Poseidon CPU 验收覆盖六个参数化 PyTorch 模型、八个随机权重模型，以及 rank 1–32 的代表组合。采用规则程序；来源包、输出结构和程序包重载均检查。r622 时 100×100 Linear 的万参数例只完成请求准备；后续已完成密态与程序包验收，见 [r665](agent-resource-acceptance-r665.md)。[范围与未验证事项](agent-generalization-acceptance-r622.md)。

### 公开数学常量映射

`descriptors-math-roles-v2` 在生成上下文中保留公开标量与多项式系数的数学含义，并列出每个系数对应的预绑定 Plain 名称。`pN` 编号没有系数顺序；多项式按声明的 power 或 Chebyshev 基、升序系数解释。普通张量仍使用原分块与布局映射，大权重保留描述符。

CLI 选择 `--public-weights descriptors-math-roles-v2`；组件选择 `public_weight_policy: "descriptors-math-roles-v2"`。数学角色最多 256 个值、48 KiB，总生成上下文保持 128 KiB。缺少非零系数绑定或超出预算时在准备阶段拒绝。

旧 `descriptors-v1` 和旧请求哈希保持。新策略形成独立请求与验证身份，使用当前版本重新捕获来源；历史程序与报告继续按旧绑定保留。[离线结果](agent-public-math-roles-results-r1307.md) · [原真实 Agent 批次](agent-gelu-live-results-r1306.md)。


## PyTorch 外部权重

v3 清单在参数化 PyTorch v2 的基础上增加 `weights`。模型工厂构造网络后，可信捕获器按照原始 `state_dict` 名称加载完整状态，再进行 FX 捕获。

| 字段 | 内容 |
|---|---|
| `format` | `poseidon-trusted-pytorch-v3` |
| `weights` | state_dict 名称到 NPY 声明的映射 |
| 单份声明 | `file`、`sha256`、`bytes`、`shape`、`dtype` |
| 文件布局 | 与清单同目录的平铺 NPY 文件 |
| NPY 类型 | `<f4`、`<f8`、`<i4`、`<i8`、`|b1` |
| 模块匹配 | 完整名称集合、精确 shape 和 dtype |
| dtype policy | `strict-float64-v1` 或 `convert-float64-v1` |

整数和布尔用于公共状态；加密输入及计算 reference 为 float64。整型值范围为 `[-2^53, 2^53]`，浮点值有限且绝对值最多 1024。标量缓冲区采用 `shape: []`，例如 BatchNorm 的 `num_batches_tracked`。权重数量、元素和 rank 按选定资源配置检查。

在项目的 Python 工作环境中，可从已经构造好的 CPU 模块导出声明：

~~~python
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, "scripts/baseline")
from pytorch_weight_package import export_state

# model 是与 model.py 中 make_model 架构相同的 CPU nn.Module。
folder = Path("pytorch-input")
weights = export_state(model, folder, resource_profile="general-tensors-v1")
source = Path("my_model.py").read_bytes()
(folder / "model.py").write_bytes(source)
manifest = {
    "format": "poseidon-trusted-pytorch-v3",
    "files": {"model.py": hashlib.sha256(source).hexdigest()},
    "entry": "model.make_model",
    "factory": {"args": [], "kwargs": {}},
    "inputs": [{"name": "x", "shape": [1, 4]}],
    "outputs": ["out"],
    "approximations": {},
    "dtype_policy": "convert-float64-v1",
    "weights": weights,
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
~~~

导出目录使用新建或空目录。NPY 数据先通过检查，再写入文件；已存在的数据保持原样。源文件使用 UTF-8，文件清单按字节哈希校验。

用 `prepare_model.py --trusted-pytorch-manifest pytorch-input/manifest.json --trust-local-code --resource-profile general-tensors-v1 --write --output "$results/imported-model"` 导入。运行命令见 [README](../../scripts/README.md#pytorch-外部权重)。

捕获报告记录源 dtype、显式 float64 策略、加载状态哈希及图常量对应关系。程序包保存数学图、公开权重绑定和输出结构，执行时直接使用已经编译的产物。原生 float32 应用精度仍需单独评测。

[六类模型验收](agent-pytorch-state-weights-acceptance-r636.md) 使用规则候选；[此前 14 项真实 Agent](agent-generalization-live-acceptance-r626.md) 覆盖参数化 v2 与声明式权重包。v3 的真实 Agent 评测单列。


## 函数式归一化与总体方差

可信本地PyTorch输入支持 `F.layer_norm`、`F.group_norm`、`torch.var` 和 `Tensor.var`。

- LayerNorm使用静态的归一化后缀shape。
- GroupNorm输入为rank 3–4，分组数为通道数的静态约数。
- gain／bias使用公开张量；省略时采用单位gain和零bias。
- 方差使用 `correction=0` 或 `unbiased=False`，轴和keepdim静态确定。
- 归一化指定rsqrt近似，epsilon为 `0 < eps <= 1`。

示例输入：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted_pytorch_functional_v1/layer_0.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-layernorm"
~~~

导入后的 `model.json` 使用现有生成和程序包接口。9个小模型已通过规则候选的Poseidon CPU数值验收，新增真实Agent测试另列；见[验收记录](agent-pytorch-functional-acceptance-r678.md)。示例的近似计算精度与原模型应用精度分别评估。


## 常见模块与函数写法

可信 PyTorch 捕获器将下列等价入口统一为已有算子图：

| 输入调用 | 规范化图与参数 |
|---|---|
| `F.conv1d/2d` | Conv1d/2d；公开权重和可选 bias；静态 stride、整数 padding、dilation、groups |
| `F.avg_pool1d/2d` | AvgPool1d/2d；静态 kernel、stride、padding；`ceil_mode=False` |
| `F.batch_norm` | BatchNorm；rank 2–4；公开统计量；`training=False` |
| `nn.LayerNorm(..., elementwise_affine=False)` | LayerNorm；公开单位 gain、零 bias；静态后缀 shape |
| `nn.GroupNorm(..., affine=False)` | GroupNorm；rank 3–4；通道数整除 groups |
| `Tensor.T` | 按反向轴顺序的 permute |
| 公开二维矩阵参与 `mm/matmul/@` | Linear 及必要的转置；另一操作数为 rank 2–3 密文张量 |

LayerNorm／GroupNorm 沿用显式 rsqrt 近似配置。公开矩阵权重来自可信常量绑定；公开 rank-3 批量矩阵与向量形式另有接入诊断。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/pytorch_spelling_v1/conv2_2.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-functional-conv"
~~~

24 个模型覆盖上述八组写法的三个上下文；均完成原模块、两套独立参考及规则候选 Poseidon CPU 数值验收。完整范围、近似误差与尚未验证的写法见 [r695](agent-pytorch-spelling-acceptance-r695.md)。这批来源包未新增真实 Agent 生成。

## 程序包输入区间

本地执行按来源绑定的原图检查新输入及相关中间值的声明区间，采用既定近似语义。成功结果包含 `input_domain_check`，越界诊断包含节点和公开区间；区间检查、CKKS 数值比较与原函数应用精度分别记录。执行入口见[应用接口](application-agent-component-v1.md)。

Agent 自有容量规则基线已接入 `split` 的多个输出，覆盖连续视图和跨块路由。三个代表上下文已完成真实 Poseidon CPU 数值验收；Attention 的当前容量基线仍有编译 scale 约束失败。[验收记录](agent-execution-domain-acceptance-r706.md)。

## 静态形状与标量输出

使用 `poseidon-trusted-pytorch-v4` 清单。字段沿用 v3，包括 `factory`、`dtype_policy` 和完整的 `weights`；无参数、无缓冲区的模块使用 `weights: {}`。

v4 根据声明的输入形状和已登记的形状规则处理：

- `x.shape`、`x.size()`、`x.size(dim)`。
- `x.ndim`、`x.dim()`、`x.numel()`、`len(x)`。
- 由这些公开整数确定的循环、分支及形状计算。

循环在 FX 捕获时展开，展开规模按资源配置检查。中间张量的形状可来自加减乘、矩阵运算、Linear／Conv、平均池化和形状变换等已登记操作。加密数据相关控制流沿用原拒绝规则。

`sum`、`mean`、`dot` 等可以直接返回零维 Tensor，也可以与向量放在 tuple/list 或字典中：

~~~python
def forward(self, x):
    result = x
    for _ in range(x.shape[-1]):
        result = result + x * 0.125
    return {"score": result.sum(), "features": result}
~~~

清单 `outputs` 按叶子顺序命名，例如 `["score", "features"]`。零维输出在计算图中表示为单元素张量；本地执行的 `structured_outputs` 恢复为数值，`outputs` 保持原有命名数组接口。输出结构保存在程序包中。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-static-v4/scalar_dot.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-static-model"
~~~

六个[示例来源包](../../scripts/baseline/cases/trusted-pytorch-static-v4/)覆盖公开循环、直接标量和混合输出。v1–v3 的入口及哈希规则保留；v4 显式启用新捕获策略。[离线验收](agent-pytorch-static-acceptance-r710.md)。

## 逐节点与模块近似

`poseidon-trusted-pytorch-v5` 沿用 v4 的静态形状、输出结构和 NPY 状态权重，增加版本化的 `approximations`。选择顺序为 **节点 → 模块 → 算子默认值**。

~~~json
{
  "format": "poseidon-pytorch-approximations-v1",
  "defaults": {
    "relu": {
      "method": "polynomial", "basis": "power",
      "domain": [-2.0, 2.0], "coefficients": [0.0, 0.25, 0.0625]
    }
  },
  "modules": [{
    "path": "act", "function": "relu",
    "profile": {
      "method": "polynomial", "basis": "power",
      "domain": [-2.0, 2.0], "coefficients": [0.0, 0.5, 0.125]
    }
  }],
  "nodes": [{
    "name": "act_1", "function": "relu",
    "profile": {
      "method": "polynomial", "basis": "power",
      "domain": [-2.0, 2.0], "coefficients": [0.0, 0.375, 0.0625]
    }
  }]
}
~~~

上例对应共享 `act` 模块的两次调用：第一次采用模块配置，第二次采用节点配置；其他 ReLU 调用采用默认值。系数用于配置接口示例，应用精度另行评测。

| 字段 | 匹配方式 |
|---|---|
| `nodes[].name` | 固定 Torch 版本中的 FX 名称，如 `act_1`；原图中对应 `fx_act_1` |
| `modules[].path` | 完整模块路径，如 `encoder.0`；函数式运算按所属模块匹配 |
| `function` | 近似组件，如 `relu`、`exp`、`reciprocal`、`rsqrt` |
| `profile` | 已登记的方法、有效区间、系数或迭代参数 |

复合运算分别选择组件：LayerNorm 使用 `rsqrt`；Softmax 使用 `exp` 和 `reciprocal`；LogSoftmax 使用 `exp` 和 `log`。多个所属模块选择器同时命中同一组件属于同级冲突，即使存在节点覆盖也返回诊断。每个选择器须至少匹配一次；被高优先级覆盖的有效默认项仍计为匹配。

配置最多 128 个节点／模块选择器、128 KiB。区间和参数经过现有近似规则检查；原图、分解图、选择记录及其哈希共同绑定。所有近似组件使用显式配置。

v5 新增以下调用映射，并沿用各算子的形状和参数约束：

| 计算 | 调用 |
|---|---|
| 激活 | `F.relu/leaky_relu/prelu/gelu/silu`，`torch.relu/sigmoid/tanh`，`Tensor.sigmoid/tanh` |
| 基本函数 | `torch.exp/reciprocal/rsqrt/sqrt` 及相应 Tensor 方法 |
| Softmax | `nn.Softmax`，`torch/F/Tensor.softmax` |
| LogSoftmax | `torch/F/Tensor.log_softmax` |

激活采用返回新值的调用；GELU 的源函数选择 `approximate="none"`。Softmax 的维度静态确定，数值域采用 float64。v1–v4 的默认行为和请求接口保持原样。

导入示例：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-approximation-v5/nodes.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-scoped-activation"
~~~

宿主继续使用 `app.submit_import(imported_directory, options=...)`，由既有生成、修复及验证流程处理。来源包的 `capture-reference.json` 保存 `approximation_selection`；程序包和新输入执行沿用该绑定。

原函数误差和 CKKS 执行误差分别报告。当前代表例已完成规则／人工表达的 Poseidon CPU 数值验收；v5 新功能的真实 Agent 评测待单独授权。[本批结果与剩余限制](agent-pytorch-approximation-acceptance-r721.md)。

## 静态张量索引

可信 PyTorch 清单选择 `poseidon-trusted-pytorch-v6`。字段沿用 v5：工厂参数、dtype 策略、NPY 状态权重、输入签名、命名输出及 `approximations`。

| 写法 | 范围 |
|---|---|
| `x[:, 1::2]`、`x[..., 1]` | 正步长切片、单个省略号 |
| `x[-1, 0]`、`x[None, :, :]` | 整数／负整数、插入单元素轴，支持标量输出 |
| `torch.select`、`Tensor.select` | 静态轴和整数位置 |
| `torch.narrow`、`Tensor.narrow` | 静态整数 start/length，长度为正 |
| `x[:, [3, 0, 3]]` | 单个公开整数向量轴，允许重复及负位置 |
| `torch.index_select`、`Tensor.index_select` | int32/int64 公开索引；位置、关键字调用均可 |

例如固定 `[2,4]` 输入：

~~~python
def forward(self, x):
    return {"features": x[:, 1::2], "score": x[-1, 0]}
~~~

`outputs` 按叶子顺序填写 `["features", "score"]`。索引后可继续使用公开 shape 查询、计算和静态输出结构。模块中的索引 Tensor 作为状态缓冲区，采用已有 NPY 状态导出和哈希绑定。

选择须非空，步长为正，每次最多一个整数向量轴，同次选择中向量和整数不混用。布尔及加密数据相关选择返回输入诊断。

导入一个来源包：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-index-v6/slicing_0.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-index-model"
~~~

应用继续调用 `app.submit_import(imported_directory, options=...)`，v6 容量图自动选择 v9 公共路由；显式旧选项保持。CLI 从导入图准备请求：

~~~bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case "$results/imported-index-model/model.json" --prepare \
  --unified-chunk-period 4 --unified-packing capacity-public-tensors-v9 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

v1–v5 和 JSON 默认配置保持。15 个索引来源包、重复输出回归及当前源码的三份程序包已完成规则候选的 Poseidon CPU 验收；新功能真实 Agent 评测另计。[本批结果](agent-pytorch-index-acceptance-r761.md)。

## PyTorch 矩阵运算

`poseidon-trusted-pytorch-v7` 清单沿用 v6 的工厂参数、NPY 权重、近似选择和输出结构，增加静态矩阵乘法规范化。

| 写法 | 输入 |
|---|---|
| `x @ y`、`torch.matmul(x, y)`、`x.matmul(y)` | 向量、矩阵、多维批次及兼容的批次广播 |
| `dot` | 两个等长向量 |
| `mm` | 两个二维矩阵 |
| `bmm` | 批次数相同的两个三维张量 |
| 公开权重参与矩阵乘法 | 固定向量、二维矩阵或批量矩阵 |

操作数至少有一个来自加密输入。维度与类型按静态签名校验；向量升维、批次广播和输出还原均记录在 `matrix_capture` 中。标量返回值保留标量结构。

声明式 `poseidon-operator-graph-v1` 的直接 `matmul` 当前接入 rank 2/3 双密文输入。向量、高阶批次和公开权重由上述 v7 捕获规范化接入；两条入口的范围尚待统一。[24 例跨入口对照](agent-matrix-input-parity-audit-r900.md)。

二维公开权重和公开向量复用 Linear；批量公开权重复用广播、乘法和归约。多个密文操作数统一到已有矩阵分解。规范化发生在 Agent 自有输入层，编译器和执行库保持原接口。

规模按所选资源配置检查。`general-tensors-v1` 允许 rank 1–32、输入和输出各 4096 元素、单个中间张量 65536 元素。矩阵乘法物化的乘积张量参与中间规模检查；中间结果与最终命名输出分别计量。

导入一个示例：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-matrix-v7/cipher_batch_1/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-matrix-model"
~~~

宿主调用 `app.submit_import(imported_directory, options=...)`；v7 容量图自动选择 v9 公共路由。命令行准备请求：

~~~bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case "$results/imported-matrix-model/model.json" --prepare \
  --public-weights descriptors-v1 --unified-chunk-period 8 \
  --unified-packing capacity-public-tensors-v9 \
  --unified-guidance explicit-v10 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

[30 个示例来源包](../../scripts/baseline/cases/trusted-pytorch-matrix-v7/)按十类用法各安排三个上下文，包含标量、批次广播、公开权重及多输出。已完成规则候选的 Poseidon CPU 编译、数值和程序包复用验收；v7 的真实 Agent 生成另行评测。[验收报告](agent-pytorch-matrix-acceptance-r815.md)。

示例可在新的目录重建：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/create_pytorch_matrix_fixtures.py \
  --output "$results/matrix-fixtures"
~~~

生成器按字节复现清单、Python 源码和 NPY 权重。旧 v1–v6 的格式、矩阵范围及请求规则继续保留。

## PyTorch InstanceNorm

`poseidon-trusted-pytorch-v7` 接入模块和函数写法，输出同一高层图与程序包接口。

| 入口 | 静态输入 |
|---|---|
| `nn.InstanceNorm1d` | `[C,L]` 或 `[N,C,L]` |
| `nn.InstanceNorm2d` | `[C,H,W]` 或 `[N,C,H,W]` |
| `F.instance_norm` | `[N,C,L]` 或 `[N,C,H,W]` |

模块的 `num_features` 与通道数匹配，epsilon 范围为 `0 < eps <= 1`。

- 当前输入统计量：模块 `track_running_stats=False` 或函数 `use_input_stats=True`，清单指定 rsqrt 近似。函数的 running_mean／running_var 取 None。
- 冻结统计量：eval 模式的模块 `track_running_stats=True`，或函数 `use_input_stats=False`；公开均值与方差复用精确 BatchNorm 推理分解。
- affine 为公开 gain 和 bias；省略时取单位增益和零偏置。
- 非 batch 模块输入在捕获时增加 batch 轴，输出恢复原形状。
- v7 来源清单绑定完整 state_dict 的 NPY 文件；近似选择沿用节点、模块路径、默认值的优先级。

示例导入：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-instance-v1/module1_input_1/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-instancenorm"
~~~

冻结统计量示例使用 `unbatched2_frozen_1/manifest.json`。导入目录由 `app.submit_import(directory)` 接入应用；v7 使用现有 v9 公共路由。

[来源包](../../scripts/baseline/cases/trusted-pytorch-instance-v1/)提供十组写法各三个上下文，包含 Tensor、tuple／dict、均值和残差输出。[来源生成器](../../scripts/baseline/benchmarks/tools/generate_instance_sources.py)从固定模板重建清单与 82 个 NPY 状态数组。

30 个规则候选完成真实编译、Poseidon CPU 数值及三份程序包复用验收。当前统计量示例的近似误差与 CKKS 误差分别记录，原模型应用精度单独评估。[验收报告](agent-pytorch-instance-acceptance-r847.md)。

## 组合模型

[24 个来源上下文](../../scripts/baseline/cases/trusted-pytorch-composition-v1/)将卷积、归一化、矩阵乘、索引、分支及输出结构放在同一静态图中。每组采用三种形状，输出分别为 Tensor、tuple 和字符串键字典。

Conv→GroupNorm→AvgPool→Linear、公开索引后的批量矩阵乘、公开矩阵融合、Linear 残差与 LayerNorm 均有规则程序的 Poseidon CPU 数值结果。较深平方分支、InstanceNorm 后接平方与池化、Attention 核心的部分组合停在编译容量检查。

torch.split 的三个原 Python 来源已在新源码下完成捕获和 Poseidon CPU 数值验收；原失败与人工等价 JSON 结果按原绑定保留。[当前结果](agent-pytorch-split-acceptance-r878.md)。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-composition-v1/cnn_norm_head_1/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-composition"
~~~

来源生成器采用完整 NPY 状态，输出到新目录：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/generate_composition_sources.py \
  --output "$results/composition-fixtures"
~~~

[本批报告](agent-pytorch-composition-acceptance-r864.md)包含已执行范围、近似误差、原失败与下一项映射工作。原模型应用精度、真实 Agent 新生成分别验收。


## PyTorch 张量拆分

可信 PyTorch v6/v7 支持 torch.split 与 Tensor.split。拆分大小为公开正整数，或一组覆盖目标轴的正整数 sections；dim 为静态有效轴，可使用负轴。

~~~python
def forward(self, x):
    left, right = torch.split(x, [2, 2], dim=-1)
    return {"left": left.square(), "right": right + left}
~~~

整数 size 支持不足一块的尾部。拆分结果可直接返回 tuple，也可参与公开循环、索引、拼接、stack 或 Linear。内部端口为2–4个时映射到已有 split，其余映射到 slice；展开节点和最终输出分别受所选资源配置约束。公开 NPY 权重同样支持静态拆分。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-split-v1/function_sections_0/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/split-example"
~~~

[18 个来源上下文](../../scripts/baseline/cases/trusted-pytorch-split-v1/)覆盖函数、方法、关键字、tuple、公开循环和公开权重。18 个来源、原三个 split→stack 来源及三个兼容回归均通过真实 Poseidon CPU 数值验收；三个程序包完成重载、新输入重复执行和缓存复用。真实 Agent 新生成另计。[验收记录](agent-pytorch-split-acceptance-r878.md)。


## 容量规则程序

Poseidon CPU 的规则 lowering 使用绑定的公开常量和布局。完整周期的乘法恒等向量以及重复二进制掩码可以消除；短向量和补零尾块仍保留其掩码作用。公共因子排程按依赖关系选择，不按模型名称分派。

这一排程用于确定性规则程序。Agent 提交的候选保留生成源码，通过现有静态、编译、产物和数值检查。深度估计用于诊断，具体组合仍须编译与密态验收。[当前验收](agent-capacity-scheduling-static-acceptance-r885.md)。

## JSON 矩阵计算图

`poseidon-operator-graph-v2` 将 JSON 与 PyTorch v7 的静态矩阵形状处理统一。支持 `matmul` 和 `dot`，输入可为向量、矩阵或批次广播张量；固定公开向量、矩阵和批量矩阵可位于左侧或右侧。规模按所选资源配置、每个算子及物理布局检查。

沿用 `inputs / constants / nodes / outputs` 声明，`matmul` 与 `dot` 的 `attrs` 为 `{}`。向量点积输出使用 `[1]` 逻辑张量，后续 stack、广播和归约按该形状处理。原图、静态变换、公开权重变换和节点来源随程序包绑定；近似函数继续采用显式区间和参数。

~~~bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-matrix-v2/public_left_0.json \
  --resource-profile general-tensors-v1 \
  --write --output "$results/imported-json-matrix"
~~~

导入后使用 `app.submit_import(imported_directory, options=...)`。容量图默认采用 `capacity-public-tensors-v9`；也可显式指定布局。旧 `poseidon-operator-graph-v1` 的图、矩阵范围和请求规则继续保留。

[36 个模型](../../scripts/baseline/cases/operator-matrix-v2/catalog.json)已完成导入、独立参考、候选静态检查和任务提交；当前新格式的真实编译、密态及 Agent 生成待验收。[接入结果](agent-json-matrix-static-acceptance-r913.md)。

## 高层算子版本回归

JSON v1/v2 的 34 个登记高层算子已完成 110 个不同结构、1,760 组输入的参考与分解比较，每个至少三个上下文。覆盖归一化、池化、RoPE、eval Dropout 及显式近似函数。模型集保持；这些结果单独计作明文参考，新格式的后端和 Agent 验收继续单列。[范围与复现](agent-operator-version-reference-acceptance-r923.md)。


## 公开量与张量算术

JSON 图选择 `poseidon-operator-graph-v3`，PyTorch 来源清单选择 `poseidon-trusted-pytorch-v8`。原字段沿用上一版本。

| 计算 | 公开量位置 | PyTorch 写法 |
|---|---|---|
| 加法 | 左侧或右侧 | `a + b`、`torch.add(a, b)`、`a.add(b)` |
| 减法 | 左侧或右侧 | `a - b`、`torch.sub(a, b)`、`a.sub(b)` |
| 乘法 | 左侧或右侧 | `a * b`、`torch.mul(a, b)`、`a.mul(b)` |

至少一个操作数来自加密输入或中间张量。公开量可以是有限标量、固定 Tensor 或可广播的固定 Tensor。`add/sub` 的 `alpha` 为 1；输入形状、节点、权重和物理资源仍按所选配置检查。

来源清单选择 `poseidon-trusted-pytorch-v9` 后，Tensor 加减乘结果支持静态 `shape`、`size`、`ndim`、`dim`、`numel` 和 `len` 查询，例如 `y = x.add(public); return y.reshape(y.numel())`。V8 保留原行为。[用法与范围](agent-pytorch-arithmetic-shape-acceptance-r974.md)。

Agent 输入层把公开量左侧的加法、乘法转为等价操作数顺序，把 `public - x` 转为 `-x + public`。原始来源、公开状态、变换记录和分解图分别保存；DSL 构造与编译管线沿用现有接口。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"

python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-arithmetic-v3/arithmetic_subtract_tensor_left.json \
  --resource-profile general-tensors-v1 --write \
  --output "$results/imported-arithmetic-json"

python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-arithmetic-v8/v8_subtract_tensor_left_operator/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-arithmetic-pytorch"
~~~

宿主使用 `app.submit_import(directory, options=...)`。新版本的容量图默认选择 v9 公共路由；原版本的接入与请求规则保持。

21 个 JSON 上下文和 25 个 PyTorch 入口案例已完成明文参考与应用提交检查。PyTorch 中包含 18 个数学上下文、6 个写法变体和 1 个 dtype 变体；与 JSON 重叠的数学模型只计一次。Poseidon 编译、密态计算和真实 Agent 生成分别验收。[本批结果](agent-arithmetic-intake-acceptance-r949.md)。

## PyTorch v10 算子结果与形状查询

可信来源清单使用 `"format": "poseidon-trusted-pytorch-v10"`，其余工厂、权重、输入和输出字段沿用现有格式。v10 继承 v9 的算术和静态索引接口，并接入以下结果的 shape、size、ndim、dim、numel 和 len 查询：

- `nn.AdaptiveAvgPool1d/2d` 与 `F.adaptive_avg_pool1d/2d`。
- `nn.InstanceNorm1d/2d` 与 `F.instance_norm`。
- `nn.ConstantPad1d/2d`、`nn.ZeroPad2d`、`nn.ReflectionPad1d/2d`、`nn.ReplicationPad1d/2d`。

函数式自适应平均池化接受 `[N,C,L]` 或 `[N,C,H,W]`，输出空间尺寸为公开正整数，并且逐轴不超过输入尺寸。InstanceNorm 的冻结统计量沿用公开仿射推理；使用输入统计量时沿用显式 rsqrt 近似配置。

例如：

~~~python
from torch.nn import functional as F

def forward(self, x):
    y = F.adaptive_avg_pool1d(x, 2)
    return y.reshape(y.numel())
~~~

来源清单示例：`scripts/baseline/cases/trusted-pytorch-operator-shape-v10/adaptive_pool_0_functional_query/manifest.json`。捕获命令和 `app.submit_import(directory)` 保持现有用法；旧 v1–v9 清单继续按各自规则处理。

这组扩展完成39份来源、12个数学模型组的导入和明文参考检查，Poseidon密态验收待运行。[接入记录](agent-pytorch-operator-shape-v10-acceptance-r994.md)。

## PyTorch v11 常用调用

来源清单选择 `poseidon-trusted-pytorch-v11`，保留工厂参数、NPY 状态权重、dtype、静态签名和输出结构。

| 调用 | 范围 |
|---|---|
| `torch.add/sub/subtract`、Tensor 方法 | 静态有限数值 `alpha`，支持密文或公开标量、固定张量作为另一操作数 |
| Dropout 系列模块与函数 | eval 模式、`training=False`、`inplace=False`，空间 Dropout 按 rank 检查 |
| `F.scaled_dot_product_attention` | 静态 Q/K/V，包括 Linear 投影、reshape/transpose 后的张量；`dropout_p=0` |
| Attention mask | 公开状态中的布尔 mask、有限加性 mask，或静态因果模式；布尔每行至少保留一个位置 |
| 输出 | Tensor、静态 tuple/list、字符串键字典，沿用输出结构声明 |

Q/K/V 的收缩维度、批次广播及 score/output 大小按资源配置检查。因果模式与显式 mask 分别使用。布尔和因果 mask 映射为 0/1 公开张量，参与 exp、归约与 reciprocal；Softmax 等非多项式计算使用声明的近似方法。

示例目录包含 33 个来源上下文：[清单](../../scripts/baseline/cases/trusted-pytorch-common-v11/index.json)。其中 18 个沿用原模型源码与权重，另有 12 个投影 Attention 上下文及 3 个公开量 alpha 上下文。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-common-v11/projected_attention_boolean_0/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-projected-attention"
~~~

该示例在清单中配置 exp Taylor 和 reciprocal Goldschmidt，方法、区间与参数可从 `approximations` 查看。原函数应用精度与近似图的 CKKS 数值验收分别记录。

应用直接通过 `submit_source` 提交该清单，沿用现有任务、预算、取消、Provider、自动验证与程序包接口。生成模型收到规范化图、公开权重或描述符和生成规则；来源与参考记录保存在可信侧。

~~~python
task = app.submit_source(
    {"format": "poseidon-model-source-request-v1",
     "kind": "trusted_pytorch",
     "manifest": "/path/to/manifest.json",
     "trust_local_code": True},
    resource_profile="general-tensors-v1",
    validation_level="numerical",
)
state = app.status(task)
~~~

公开 CLI 的 33 个上下文完成隔离捕获和 16 组/例的独立明文检查；8 个代表例完成生产应用接入。编译、密态和真实 Agent 新生成仍单列待验收。v1–v10 保持原选择行为和旧接口。[接入记录](agent-pytorch-common-v11-acceptance-r1045.md) · [下一批范围](agent-common-pytorch-offline-proposal-r1042.md)。


## PyTorch v12 有序张量操作

清单选择 `poseidon-trusted-pytorch-v12`，其余入口继续使用已有版本。该入口处理已登记的静态视图、切片写入、`add_`、`mul_`、`copy_` 和公开权重的转置／拆分，保留写入顺序与快照。复制按静态、无重叠区域检查，模型状态保持公开只读。

清单沿用 entry、files、factory、inputs、outputs、weights 和 approximations；float64选择 `strict-float64-v1`。Linear的完整示例见 `scripts/baseline/benchmarks/tools/staged_ordered_intake_v1/fixtures/linear/manifest.json`，对应文件包含固定NPY权重及哈希。

~~~python
task = app.submit_source(
    {
        "format": "poseidon-model-source-request-v1",
        "kind": "trusted_pytorch",
        "manifest": "/absolute/path/manifest.json",
        "trust_local_code": True,
    },
    resource_profile="general-tensors-v1",
    validation_level="numerical",
    options={
        "chunk_period": 4,
        "generation_guidance": "explicit-v11",
        "capacity_packing_version": "capacity-public-tensors-v9",
        "public_weight_policy": "descriptors-v1",
    },
)
result = app.run(task, trusted_provider)
outputs = app.execute(task, samples)
~~~

`general-tensors-v1` 的逻辑rank为1–32，输入和输出各最多4096元素；各算子与物理布局保持各自限制。组合有序操作与常用调用时显式选择 v13；融合矩阵调用选择 v14。旧版本保留原契约。

当前三例完成真实来源捕获、规则候选编译、Poseidon CPU数值比较和包复用；其他组合和新接口真实Agent评测单独记录。[当前结果](agent-generalization-progress-r1231.md)

## PyTorch v13 有序操作与常用调用

来源清单选择 `poseidon-trusted-pytorch-v13`。沿用 v12 的静态视图、切片写入、
快照与公开权重包，并接入静态 `alpha`、eval Dropout 和
`F.scaled_dot_product_attention`。

Attention 使用静态 Q/K/V、公开 mask 和声明的近似配置。
布尔 mask 每行至少保留一个位置；因果模式与显式 mask 分别选择。
模型、原函数参考和近似图分别保存。

清单字段沿用 `entry`、`files`、`factory`、`inputs`、`outputs`、`weights`
和 `approximations`。应用仍通过 `submit_source` 导入，再用 `run` 生成与验证、
`execute` 执行已验证程序包。

联合入口已完成 alpha、eval Dropout、布尔掩码 Attention 的代表性 Poseidon CPU 验收，
包含包重载、新输入和缓存复用。这些结果使用规则／人工候选；
新版真实 Agent 生成单独验收。

## PyTorch v14 融合矩阵调用

来源清单选择 `poseidon-trusted-pytorch-v14`，包含 v13 的输入能力。

| 调用 | 计算 | 形状 |
|---|---|---|
| `torch.addmm` / Tensor 方法 | `beta*bias + alpha*(left @ right)` | 两个 rank 2 矩阵 |
| `torch.baddbmm` / Tensor 方法 | 各批次矩阵乘再加偏置 | 两个 rank 3 张量，批次数相同 |
| `torch.addbmm` / Tensor 方法 | 批次矩阵乘沿 batch 轴求和再加偏置 | 两个 rank 3 张量，输出 rank 2 |

收缩维度必须匹配，偏置可广播到结果形状。
`alpha`、`beta` 为有限公开标量，绝对值最多 1024。
公开权重和加密操作数按类型与布局检查；`out` 仅接受省略或 `None`。

示例模型：

~~~python
class Model(torch.nn.Module):
    def forward(self, bias, left, right):
        return torch.addmm(bias, left, right, alpha=0.5, beta=0.25)
~~~

对应静态输入签名可选：

~~~json
[
  {"name": "bias", "shape": [2, 3], "dtype": "float64"},
  {"name": "left", "shape": [2, 4], "dtype": "float64"},
  {"name": "right", "shape": [4, 3], "dtype": "float64"}
]
~~~

三种调用各完成全加密、公开右矩阵、公开偏置的代表性验收，共九例，
均通过真实编译、密态数值比较、包重载后的新输入和缓存复用。
这些是有限上下文的规则基线结果。
[联合入口验收](agent-common-fused-cutover-acceptance-r1303.md)记录具体案例与绑定。



## PyTorch v15 GELU 来源变体

清单选择 `poseidon-trusted-pytorch-v15`，字段沿用 v14。支持：

| 写法 | 原函数变体 |
|---|---|
| `nn.GELU(approximate="none")`、`F.gelu(x, approximate="none")` | erf |
| `nn.GELU(approximate="tanh")`、`F.gelu(x, approximate="tanh")` | tanh |
| `F.gelu(input=x, approximate=...)` | 按静态参数选择 |

`approximations` 中声明 GELU 的多项式、有效区间与系数。
导入分别保存原函数参考、近似计算和分解结果；数值验收比较密态输出与声明的近似计算。

示例清单位于 `scripts/baseline/cases/trusted-pytorch-gelu-v15`：
module、functional、input keyword 三种写法，各含 none／tanh 变体。
六例完成隔离捕获和双参考检查；三例 tanh 规则程序完成 Poseidon CPU 编译、
密态、程序包重载和新输入检查。真实 Agent 新生成按单独批次验收。
[结果与源码绑定](agent-resource-gelu-progress-r1305.md)。

## 自动执行资源检查

数值验收与程序包本地执行在加密运行前核验绑定的 RuntimePlan 和 plaintext bundle，
按固定 Runtime 的全部定义保留行为计算系数存储下界。超过原沙箱上限时返回
`resource_gate / compiled_retained_payload_exceeds_budget`；错误上下文仅含公开位置和整数预算。

通过下界检查后继续真实执行与数值验收。峰值、评估密钥和 evaluator 工作区分别记录为未知项。
`resource-precheck.json` 保存参数、源码、产物和策略绑定；执行前后都核验该文件。
显式编译验收与数值验收保持各自交付等级。


## PyTorch v16 MultiheadAttention

来源清单选择 `poseidon-trusted-pytorch-v16`，沿用 v15 的工厂、公开权重、
有序操作和近似配置。输入是静态的 query、key、value，支持 rank 2 单样本
和 rank 3 批次输入。

| 配置 | 范围 |
|---|---|
| 维度 | embed_dim 可被 num_heads 整除；静态序列长度与 batch |
| 调用 | self/cross Attention、batch_first、独立 kdim/vdim |
| 公开权重 | 合并或独立 Q/K/V 投影、输出投影、bias_kv、add_zero_attn |
| mask | bool 或有限浮点公开 mask；显式因果 mask |
| 返回 | need_weights、average_attn_weights；Tensor、tuple/list、字典及静态 None |
| 组合 | 现有有序视图和修改、Linear、残差、融合矩阵与显式近似 GELU |

MHA 的布尔 mask 采用 PyTorch 语义：True 表示屏蔽。每行至少保留一个位置。
None 是返回结构的一部分，其余 Tensor 叶子按清单 outputs 顺序命名，
总共最多四个 Tensor 输出。近似配置分别指定 exp 和 reciprocal 的方法、
区间及参数；每个配置还需通过所选资源、布局和后端验收。

示例位于 `scripts/baseline/cases/trusted-pytorch-mha-v16`。
应用调用形式保持：

~~~python
task_id = app.submit_source({
    "format": "poseidon-model-source-request-v1",
    "kind": "trusted_pytorch",
    "manifest": "/absolute/path/to/manifest.json",
    "trust_local_code": True,
}, resource_profile="general-tensors-v1", options={
    "generation_guidance": "explicit-v11",
    "public_weight_policy": "descriptors-math-roles-v2",
    "chunk_period": 4,
})
~~~

上述选项可复现三个固定小例子的请求绑定。来源在隔离进程内捕获，
再交给现有生成与验证流程。

三个示例的规则程序通过了真实编译、密态比较、重载后的新输入和重复执行。
这些低阶配置的原函数近似误差仍单独保留，应用精度与真实 Agent 生成另行验收。
[本次接入结果](agent-mha-component-cutover-results-r1312.md)。

规则开发接口可显式选取已提取的布局策略：

~~~python
source, record = candidate_source(request, strategy="mixed-packed-cells-v1")
~~~

该接口适用于确定性基线。策略在 C-order packed Linear 和按需 replicated cell
之间衔接，Provider 生成的候选走原有检查与验收路径。


## PyTorch v17 静态 chunk 与 unbind

清单选择 `poseidon-trusted-pytorch-v17`，沿用 v16 的工厂参数、公开权重、
近似配置、有序视图和输出结构。函数和 Tensor 方法均可使用。

| 调用 | 语义 | 范围 |
|---|---|---|
| `torch.chunk(x, chunks, dim)` / `x.chunk(chunks, dim)` | 沿指定轴分成大小接近的视图；末块可以较短 | 静态正整数 chunks，rank 1–32 |
| `torch.unbind(x, dim)` / `x.unbind(dim)` | 按该轴逐项取出视图并移除该轴 | rank 2–32 |
| 组合 | Linear、公开权重、算术、transpose、命名或 tuple 输出、共享视图写入 | 沿用所选资源配置 |

支持负轴与公开关键字参数。chunk 的实际返回数量按轴长度计算；
例如长度 13、chunks=6 返回 5 个张量，长度依次为 3、3、3、3、1。
来源和展开图都受节点预算约束，最终 Tensor 输出按原签名检查。
rank 1 的 unbind 会产生标量视图，当前按能力检查拒绝。

~~~python
class Model(torch.nn.Module):
    def forward(self, x):
        left, right, tail = x.chunk(3, dim=-1)  # x: [2, 5]
        return {"merge": left + 2 * right, "tail": tail}
~~~

来源清单采用原字段，并显式选择新版本：

~~~json
{
  "format": "poseidon-trusted-pytorch-v17",
  "entry": "model.make",
  "files": {"model.py": "<SHA-256>"},
  "factory": {"args": [], "kwargs": {}},
  "inputs": [{"name": "x", "shape": [2, 5]}],
  "outputs": ["merge", "tail"],
  "dtype_policy": "strict-float64-v1",
  "weights": {},
  "approximations": {
    "format": "poseidon-pytorch-approximations-v1",
    "defaults": {}, "modules": [], "nodes": []
  }
}
~~~

沿用 `app.submit_source(...)` 和 `app.prepare_source(task_id)`，
`resource_profile="general-tensors-v1"`；生成指导可显式选择
`generation_guidance="explicit-v13"`。布局、预算、Provider 与程序包接口保持。

捕获将这两种写法规范化为已有静态索引，保存原图、规范化图、逐端口映射和实现哈希。
共享视图的写入顺序保持，写入前 clone 的快照单独保存。
六个来源模型各通过四组资格输入和四组新输入的真实 Poseidon CPU 数值检查，
含程序包重载；这是规则基线验收，真实 Agent 新生成尚待独立评测。
[接口与结果](agent-pytorch-partitions-acceptance-r1339.md)。


## PyTorch v18 静态 einsum

清单采用 `poseidon-trusted-pytorch-v18`，沿用 v17 的工厂、公开权重、
近似配置、有序视图、chunk/unbind 和输出字段。

| 写法 | 支持范围 |
|---|---|
| `torch.einsum("ij,jk->ik", x, y)` | 矩阵收缩 |
| `torch.einsum("bhqd,bhkd->bhqk", q, k)` | 批次、分头和标签广播 |
| `torch.einsum("...ij,...jk->...ik", x, y)` | 静态 ellipsis |
| 三／四张量方程 | 有界广播乘法与归约 |
| 单张量转置／归约 | 视图别名或现有 sum |

方程须是静态字符串，接受 1–4 个 rank 1–32 的 Tensor 操作数，
也接受旧式 Tensor 列表调用。隐式输出按 PyTorch 标签顺序处理。
单操作数内重复标签、标量 Tensor 操作数和整数标签调用有明确拒绝诊断。
结果、工作区及展开节点均按资源配置检查。

~~~python
class Model(torch.nn.Module):
    def forward(self, x, y):
        return {"product": torch.einsum("ij,jk->ik", x, y)}
~~~

来源清单在原字段基础上选择新版本：

~~~json
{
  "format": "poseidon-trusted-pytorch-v18",
  "entry": "model.make",
  "files": {"model.py": "<SHA-256>"},
  "factory": {"args": [], "kwargs": {}},
  "inputs": [{"name": "x", "shape": [2, 2]}, {"name": "y", "shape": [2, 2]}],
  "outputs": ["product"],
  "dtype_policy": "strict-float64-v1",
  "weights": {},
  "approximations": {
    "format": "poseidon-pytorch-approximations-v1",
    "defaults": {}, "modules": [], "nodes": []
  }
}
~~~

沿用 `app.submit_source(...)`／`app.prepare_source(task_id)`，
资源配置 `general-tensors-v1`，生成指导可选择 `explicit-v13`。
独立来源参考、规范化图和实现哈希进入原有程序包绑定。
[八模型验收](agent-pytorch-einsum-acceptance-r1340.md)包含真实编译、
密态数值和重载后的新输入；真实 Agent 新生成另行评测。

## PyTorch v19 对角线与标量

清单使用 `poseidon-trusted-pytorch-v19`，沿用 v18 的静态方程、
工厂参数、NPY 权重和输出结构。

| 写法 | 计算 |
|---|---|
| `torch.einsum("ii->i", x)` | 对角线视图 |
| `torch.einsum("ii->", x)` | trace，返回标量 |
| `torch.einsum("...ii->...", x)` | 每个批次的 trace |
| `torch.einsum("bii,ij->bj", x, w)` | 对角线与矩阵收缩 |
| `torch.einsum(",ij->ij", s, x)` | 标量 Tensor 与矩阵相乘 |

重复标签对应的维度相等。单输入、无归约的对角结果保留共享视图，
写入该视图会更新原张量，原张量的后续写入也会反映在视图中。
标量 Tensor 由归约、基本索引或 `reshape(())` 产生；编译图使用一元素载体，
来源包保留逻辑标量形状。输入签名仍采用 rank 1–32 的静态张量。

~~~python
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        diagonal = torch.einsum("ii->i", x)
        diagonal.add_(0.125)
        return {"diagonal": diagonal, "updated": x}

def make():
    return Model().eval()
~~~

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,2 --output diagonal --output updated \
  --source-format poseidon-trusted-pytorch-v19 \
  --manifest ./model-source.json
~~~

将清单交给既有 `prepare_model.py`、`submit_source` 或 CLI 捕获入口，
选择 `general-tensors-v1` 和索引布局 `capacity-public-tensors-v9`。
元数据配置步骤保持默认 v18；旧清单的版本规则保持。

[验收报告](agent-pytorch-einsum-views-acceptance-r1346.md)覆盖六个 v19 案例及
旧版 Linear、MLP 的编译、数值和新输入复用。新构造的真实 Agent 生成评测另行进行。


## PyTorch v20 标量 unbind

来源清单选择 `poseidon-trusted-pytorch-v20`，沿用工厂参数、NPY 权重、
静态 rank 1–32 输入和命名输出。`torch.unbind`／`Tensor.unbind`
可将一维输入拆成标量 Tensor；这些结果保持对源张量的共享视图。

| 写法 | 用途 |
|---|---|
| `a, b = x.unbind()` | 标量算术 |
| `p[-1]`、`p[::2]` | 静态端口或子集选择 |
| `torch.stack([p[1], p[1], p[-1]])` | 标量组合与重复端口 |
| `a.add_(b * 0.125)` | 修改视图并更新源张量 |
| `snapshot = a.clone()` | 独立快照 |
| `a, b = self.weight.unbind()` | 读取公开 NPY 权重 |

输出支持 Tensor 和静态 tuple/list/dict，按清单中的输出名称对应张量叶子。
源张量的后续写入会反映在视图中；`clone` 保留独立结果，`detach` 保留共享关系。
图和布局继续使用所选资源配置；输出叶子最多 4 个，stack 输入最多 8 个。
标量在编译图中使用一元素载体，交付结果恢复逻辑标量。

~~~python
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        a, b = x.unbind()
        a.add_(b * 0.125)
        return {"scalar": a, "updated": x}

def make():
    return Model().eval()
~~~

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2 --output scalar --output updated \
  --source-format poseidon-trusted-pytorch-v20 \
  --manifest ./model-source.json
~~~

将来源清单交给既有 `prepare_model.py` 或应用 `submit_source`，
选择 `general-tensors-v1`、`capacity-public-tensors-v9`。v17–v19 清单按原版本处理。

[本批验收](agent-pytorch-scalar-partitions-acceptance-r1347.md)包含六个 v20 模型、
v19 对角线和 v18 MLP，共八个模型的 Poseidon 数值与程序包复用结果。
真实 Agent 新生成和修复评测单独计数。

## PyTorch v21 幂运算与公开除法

来源清单选择 `poseidon-trusted-pytorch-v21`，沿用 v20 的静态输入、NPY 权重、
共享视图和输出结构。支持以下固定数学写法：

| 写法 | 要求 |
|---|---|
| `x ** n`、`torch.pow(x, n)`、`x.pow(n)`、`x.pow_(n)` | 指数为公开固定 2 或 4 |
| `x / d`、`torch.div/divide/true_divide(x, d)` | 除数为有限非零公开数 |
| `x.div/divide/true_divide(d)` 及原地形式 | 同上，视图写入按原执行顺序传播 |
| `x.div_(self.gain)` | gain 为直接声明并经 NPY 哈希绑定的 float64 标量状态 |

除数和倒数的绝对值各不超过 1024。密文除数、Tensor 向量或计算得到的 Tensor 除数、
动态／负／分数指数、rounding 和 out 写入返回不支持。
幂分解到 square，公开除法分解到固定倒数乘法；标量中间值与张量形状保持。

~~~python
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        snapshot = x.clone()
        x.pow_(4)
        x.div_(2)
        return {"snapshot": snapshot, "updated": x}

def make():
    return Model().eval()
~~~

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2 --output snapshot --output updated \
  --source-format poseidon-trusted-pytorch-v21 \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

公开标量状态使用 `--weight gain=gain.npy` 绑定，文件为 shape `[]`、dtype float64 的 NPY。
来源清单继续通过 `prepare_model.py` 或应用 `submit_source` 导入，默认清单仍为 v18。
[本批验收](agent-pytorch-exact-arithmetic-acceptance-r1348.md)包含八个新来源模型、
两个兼容模型、原两项失败的修复和程序包新输入执行；真实 Agent 生成单独评测。

## PyTorch v22 公开张量算术

5 个代表组合已通过真实 Agent 生成、Poseidon CPU 数值验证及程序包复用，见[验收报告](agent-public-tensor-live-acceptance-r1366.md)。

选择 `poseidon-trusted-pytorch-v22`，沿用有序视图、静态形状、输出结构及 NPY 权重接口。

| 计算 | 输入要求 |
|---|---|
| `x / self.std`、`torch.div(x, self.std)`、`x.div(self.std)` | std 为直接声明的公开 float64 标量／向量／张量，按 NPY 绑定 |
| `x.div_(self.std)` | 广播后的 shape 与 x 相同 |
| `x * self.mask`、`torch.mul(x, self.mask)`、`x.mul(self.mask)` | mask 为直接声明的公开 bool 状态，广播合法 |
| float32 公开权重 | 清单显式选择 `convert-float64-v1`，按 float64 计算语义验收 |

除数和倒数的绝对值各不超过 1024，值须有限且非零。布尔 mask 分解为固定 0/1 乘法。
原始权重、类型、形状和派生常量分别绑定。

~~~python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std", torch.ones(3, dtype=torch.float64))
        self.register_buffer("mask", torch.ones(3, dtype=torch.bool))

    def forward(self, x):
        return (x / self.std) * self.mask

def make():
    return Model().eval()
~~~

std.npy 保存公开 float64 向量，mask.npy 保存公开 bool 向量。准备来源清单：

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,3 --output value \
  --source-format poseidon-trusted-pytorch-v22 \
  --weight std=./std.npy --weight mask=./mask.npy \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

通过原 `prepare_model.py --trusted-pytorch-manifest` 或应用 `submit_source` 接入。
省略来源版本使用 v18；`latest` 当前为 v28。
[本批验收](agent-public-tensor-activation-acceptance-r1363.md)记录四例真实编译、Poseidon CPU 数值比较和程序包新输入／重复执行；候选来自规则基线。


## PyTorch v23 公开张量表达式

选择 `poseidon-trusted-pytorch-v23`，在原 NPY 权重和有序来源接口上增加公开状态的静态变换：

| 写法 | 范围 |
|---|---|
| reshape、view、flatten | 静态 shape；允许可确定的单个 -1 维度 |
| permute、transpose、t、T | 静态轴与既有 rank 约束 |
| 基础切片 | 整数、正步长 slice、Ellipsis 和 None，结果非空 |
| unsqueeze、squeeze、contiguous | 静态维度及已登记别名 |
| bool.float()、bool.double() | 公开布尔状态转 0/1 浮点 |
| 使用变换结果作除数或 mask | 沿用 v22 的非零、广播、幅值与倒数上限 |

例如，将公开向量和布尔 mask 变为按行广播的形状：

~~~python
def forward(self, x):
    std = self.std.view(1, -1)
    mask = self.mask.float().view(1, -1)
    return (x / std) * mask
~~~

来源清单选择新版本并绑定原数组：

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,3 --output value \
  --source-format poseidon-trusted-pytorch-v23 \
  --weight std=./std.npy --weight mask=./mask.npy \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

std 为公开 float64 向量，mask 为同长度 bool 向量；原始文件与每步派生状态都保留哈希。公开变换使用现有资源配置，输入数据相关常量、任意 dtype 转换或设备移动不在此策略中。

253 项控制检查及 18 份明文来源配置通过，其中 6 份完成隔离源码捕获。原批次 18 份完成真实编译和 Poseidon CPU 四组数值验收，17 项完整通过。命名 None 返回修复后，原 named 案例在新身份下完成首次验收、3 组新输入及 1 组重复执行，共 8 组／96 值，最大绝对误差 8.1459362763e-9。原失败记录保留；v23 真实 Agent 新生成另计。
[代码检查](agent-public-expression-component-checkpoint-r1368.md) · [离线结果](agent-public-expression-offline-results-r1370.md) · [程序包修复](agent-public-expression-output-repair-r1371.md) · [补验结果](agent-public-expression-output-results-r1373.md)。

## PyTorch v24 公开参数类型转换

来源清单选择 `poseidon-trusted-pytorch-v24`。公开参数和 buffer 支持：

| 写法 | 目标类型 |
|---|---|
| `value.float()` | float32 |
| `value.double()` | float64 |
| `value.to(torch.float32/float64)` | 显式浮点类型 |
| `value.to(dtype=torch.float32/float64)` | 显式浮点类型 |

原始类型可以是 float32、float64、int32、int64 或 bool。原始 NPY 和每步转换结果分别绑定；float32 的舍入结果随后精确扩为 float64，进入现有数学图。转换可以和静态 reshape、转置、切片、算术及命名输出组合。输入值和容量沿用选定资源配置。

例如，将公开增益先转为 float32，再用于 float64 计算：

~~~python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gain", torch.ones(3, dtype=torch.float64))

    def forward(self, x):
        return x * self.gain.float().double().reshape(1, 3)


def make():
    return Model()
~~~

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,3 --output value \
  --source-format poseidon-trusted-pytorch-v24 \
  --weight gain=./gain.npy \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

此策略处理公开状态，密文计算沿用当前 CKKS 数值类型。默认 v18、latest=v28。
161 项控制检查和 33 份隔离源码捕获通过；36 项配置完成 Poseidon CPU 编译和数值验收，三份程序包完成新输入及重复执行。真实 Agent 生成另计。

[接入检查](agent-public-dtype-component-checkpoint-r1374.md) · [离线验收](agent-public-dtype-offline-results-r1380.md)。

## PyTorch v25 公开常量算术

来源清单选择 `poseidon-trusted-pytorch-v25`。公开参数和 buffer 可在 `forward` 中使用加、减、乘、除、取负，以及已有的类型转换、形状变换和静态切片。支持运算符、对应的 Torch 函数和 Tensor 方法。

~~~python
def forward(self, x):
    return (x - self.mean) / (self.std + 0.01)
~~~

`mean`、`std` 由 NPY 权重包绑定。导入工具计算公开常量表达式，记录原始状态、派生值、依赖和哈希；涉及输入张量的运算保留在计算图中。

公开算术采用配置的 float64 参考语义。整数和布尔状态先使用 `double()` 或 `to(dtype=torch.float64)`；显式 float32 中间值也先转换到 float64。操作数、结果、标量和 add/sub 的 alpha 均为有限值，绝对值最多 1024；除数及其倒数符合相同上限。广播后的常量和原始／派生状态总量按所选资源配置检查。

清单配置示例：

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.build --input x:2,3 --output y \
  --source-format poseidon-trusted-pytorch-v25 \
  --weight mean=./mean.npy --weight std=./std.npy \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

默认 v18、latest=v28。15 个新上下文通过隔离捕获和独立明文参考；6 项应用接入检查通过。18 个规则候选已完成 Poseidon CPU 编译、数值验收及程序包重载，3 包通过新输入、重复执行和缓存复用。15 个代表上下文已完成真实 Agent 生成和数值验收；15 包均通过重载、新输入、重复执行及缓存复用。

[接入结果](agent-public-arithmetic-component-checkpoint-r1384.md) · [规则数值验收](agent-public-arithmetic-offline-results-r1386.md) · [真实 Agent 验收](agent-public-arithmetic-live-acceptance-r1388.md)。

### PyTorch v26 结构化输入

通过 `--source-format poseidon-trusted-pytorch-v26` 选择，沿用 v25 的算子和公开状态规则。新增输入写法：

- `forward(batch)`，batch 为字符串键字典。
- tuple/list 嵌套输入，叶子为声明的 Tensor。
- 位置参数、仅关键字参数及公开默认参数。

例如模型的 `forward(self, batch)` 使用 `batch["left"] + batch["right"]`。输入声明为 x:[2,3]、y:[3]，`call.json` 配置如下：

~~~json
{
  "format": "poseidon-pytorch-call-v1",
  "args": [
    {"kind": "dict", "items": [
      ["left", {"kind": "tensor", "name": "x"}],
      ["right", {"kind": "tensor", "name": "y"}]
    ]}
  ],
  "kwargs": {}
}
~~~

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,3 --input y:3 --output output \
  --source-format poseidon-trusted-pytorch-v26 --call ./call.json \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

Tensor 绑定使用 `kind: "tensor"`；公开标量使用 `kind: "literal"` 和 `value`。tuple/list 使用有序 `items`，字典使用有序键值对。仅关键字参数放在 `kwargs` 中，公开默认值从 forward 签名冻结。

每个 Tensor 叶子绑定一次，最多 4 个；树深度 8、节点 256、调用配置 8 KiB。各 Tensor 的形状和当前资源配置共同限制容量。运行时公开参数保持捕获时的值。

程序包执行可以沿用命名样本，也可以按原 forward 调用传入结构化样本：

~~~python
app.execute(task_id, [
    {"args": [{"left": x, "right": y}], "kwargs": {}}
], sample_format="forward_call")
~~~

执行器使用绑定的名字、形状和布局，并保留原输出结构。每次调用支持 1–16 组样本。

216 项初始单元／组件检查、12 个隔离来源上下文和 6 项应用接入通过；执行适配器修复后的 86 项相关回归通过。默认 v18、latest=v28。

[接入结果](agent-input-binding-component-checkpoint-r1389.md) · [离线验收方案](agent-input-binding-offline-proposal-r1390.md)

12 个结构化输入上下文和 3 个旧接口兼容已通过 Poseidon CPU 离线验收；3 份程序包完成结构化新输入、命名输入、重复执行与缓存复用。[验收结果](agent-input-binding-poseidon-results-r1391.md)。

12 个新上下文的真实 Agent 测试全部通过：11 个首次通过、1 个经自动修复通过；12 份程序包完成重载、新输入、结构化调用和缓存复用。[真实 Agent 验收](agent-structured-input-live-acceptance-r1393.md)。


## PyTorch v27 空间计算

来源清单选择 `poseidon-trusted-pytorch-v27`：

| PyTorch 调用 | 范围 |
|---|---|
| nn.Upsample、F.interpolate | 1D：nearest、nearest-exact、linear；2D：nearest、nearest-exact、bilinear |
| nn.UpsamplingNearest2d、nn.UpsamplingBilinear2d | 静态 size 或 scale_factor |
| nn.ConvTranspose1d/2d、F.conv_transpose1d/2d | 公开权重、可选 bias；静态 stride、padding、dilation、groups、output_padding |

1D resize 使用 [N,C,L]，2D 使用 [N,C,H,W]。size 和 scale_factor 选择其一；linear/bilinear 使用显式 align_corners。当前空间语义对应固定 Torch 2.0.1 的 CPU float64 连续布局，antialias=false。转置卷积支持 rank 2/3（1D）及 rank 3/4（2D）；模块可传静态 output_size。类型、形状、展开及物理密文资源按现有配置检查。

v27 沿用 v26 的结构化输入，包含显式 call.json；平坦调用也登记参数绑定。公开状态、dtype 转换、别名写入及输出结构按已有规则处理。

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:1,1,3 --output output \
  --source-format poseidon-trusted-pytorch-v27 --call ./call.json \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

平坦输入的 call.json：

~~~json
{
  "format": "poseidon-pytorch-call-v1",
  "args": [{"kind": "tensor", "name": "x"}],
  "kwargs": {}
}
~~~

清单通过同一 `prepare_model.py --trusted-pytorch-manifest` 或应用 `submit_source` 接入。三个完整来源示例见 [目录](../../scripts/baseline/cases/pytorch-spatial-v27/README.md)，包括 resize、分组转置卷积和字典输入的公开 mask 组合。

34 个明文上下文、128 项生产检查及 3 个应用提交通过。18 个代表案例（15 项 v27、2 项 v26、1 项旧 JSON）完成 Hecate 编译、Poseidon CPU 数值验证和程序包重载；3 包通过新输入、重复执行和缓存复用。真实 Agent 新生成待单独评测。[接入结果](agent-spatial-source-component-checkpoint-r1404.md) · [CPU 验收](agent-spatial-source-offline-results-r1406.md)。默认 v18，latest=v28。


## PyTorch v28 静态广播

选择 `--source-format latest` 或显式 `poseidon-trusted-pytorch-v28`。

| 调用 | 用法 |
|---|---|
| `x.expand_as(y)` | 将 x 扩展到 y 的静态形状 |
| `x.broadcast_to((2, 2))` | 将 x 扩展到公开的静态维度 |

广播遵循 PyTorch 的维度规则：从右侧对齐，原维度为 1 或与目标维度相同。`broadcast_to` 接收静态 tuple/list/torch.Size；已有维度可以用 -1 保持。标量中间值可扩展成张量。结果保持源 dtype 和 view alias，后续读写沿用别名检查。

例如输入 x 为 [1, 2]、y 为 [2, 2]：

~~~python
import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x.expand_as(y) + y

def make():
    return Model().double().eval()
~~~

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:1,2 --input y:2,2 --output result \
  --source-format latest --resource-profile general-tensors-v1 \
  --manifest ./source.json
~~~

普通位置参数按 `--input` 顺序绑定；字典、嵌套容器和关键字参数使用 `--call`。v28 沿用 v27 的空间算子、公开权重、dtype 处理与输出结构。

3 个数学请求分别验证广播加法、转置视图与别名写入后归约、标量广播接 Linear。6 个来源写法完成独立参考、真实生成或原样 DSL 重放、Poseidon CPU 数值比较及程序包复用。[阶段结果](agent-static-broadcast-phase-summary-r1433.md)。
