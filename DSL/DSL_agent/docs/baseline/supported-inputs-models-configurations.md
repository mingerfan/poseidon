> 后端迁移更新：本文描述输入契约。2026-09-28 起，新执行入口使用 Poseidon RuntimePlan；旧 SEAL 验收不计为 Poseidon 验收。当前后端范围和配置见 [迁移说明](poseidon-runtime-migration.md)。

## 当前 PyTorch 入口

`scripts/prepare_pytorch_source.py --source-format latest` 创建固定 v34 清单。
支持显式多文件包，以及登记范围内的公开状态、公开／加密混合拼接、结构化输入和命名输出。
默认入口仍为 v18；旧 JSON schema 和显式来源版本继续兼容。
[统一用法](operator-decomposition-and-python.md) · [最新拼接验收](agent-public-join-acceptance-r1480.md)。

显式 v35/v36 提供完整轴置换规范化，每个视图最多4,096个元素。v36增加完整来源记录的无损格式：存储4MiB、还原16MiB，应用任务和程序包采用相同限制。已验收2,052元素双输入、10,100公开参数Linear及32维置换三个上下文。[来源验收](agent-complete-source-acceptance-r1530.md)。


# 模型输入格式

应用组件统一来源入口为 `Application.submit_source()`，格式为 `poseidon-model-source-request-v1`。`graph` 接收下述 JSON 图，`restricted_python` 接收受限源码清单，`trusted_pytorch` 接收明确授权的本地 PyTorch 清单。三类来源沿用各自已登记的算子、形状、资源配置和导入规则。旧 CLI、schema 1–5、`submit` 及 `submit_import` 保留。[来源提交接口](application-agent-component-v1.md#模型来源提交)。


可信 PyTorch v29 使用显式源码文件清单，接收带 `__init__.py` 的 Python 包、相对导入和嵌套模块。使用 `--source-root` 指定根目录，文件与公开权重分别登记。计算规则沿用 v28。[源码包用法](operator-decomposition-and-python.md#pytorch-v29-源码包)。


新模型使用 `poseidon-model-graph-v1` JSON。它描述数学计算，不包含 DSL 源码、API 配置或测试输入数据。

## 示例：两层平方激活 MLP

保存为仓库根目录的 `model.json`：

```json
{
  "format": "poseidon-model-graph-v1",
  "id": "small_mlp",
  "inputs": [{"name": "x", "shape": [4]}],
  "constants": {
    "w1": [[0.1, 0.2, -0.1, 0.1], [0.2, -0.1, 0.1, 0.2]],
    "b1": [0.01, -0.01],
    "w2": [[0.2, -0.1]],
    "b2": [0.01]
  },
  "nodes": [
    {"id": "fc1", "op": "linear", "inputs": ["x", "w1", "b1"], "attrs": {}, "outputs": ["h"]},
    {"id": "act", "op": "square", "inputs": ["h"], "attrs": {}, "outputs": ["a"]},
    {"id": "fc2", "op": "linear", "inputs": ["a", "w2", "b2"], "attrs": {}, "outputs": ["y"]}
  ],
  "outputs": [{"name": "prediction", "value": "y"}]
}
```

`inputs` 声明名字和形状；`constants` 保存公开权重；`nodes` 按依赖顺序排列；`outputs` 选择需要返回的节点结果。节点通过名字连接，不需要指定模型类别。

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case model.json --prepare --compiler-configuration poseidon-cpu-eva-w45-v1
```

## 支持的算子

| 用途 | 算子名 | 说明 |
|---|---|---|
| 算术 | add、subtract、multiply、negate、square、power | power 指数仅 2 或 4 |
| 多项式 | polynomial | 幂基或 Chebyshev 基，系数公开 |
| 全连接 | linear | 最后一维加权求和，可含偏置 |
| 归一化 | batch_norm | 固定统计量的推理计算 |
| 形状 | flatten、reshape、transpose、permute | 静态形状；reshape 不改变元素数 |
| 组合 | concat、stack、slice、split | 轴、范围和分段大小静态确定 |
| 归约 | sum、mean | 显式指定 axes、keepdims |
| 旋转 | rotate | 按逻辑张量定义旋转，由布局转换为密文操作 |
| 卷积 | conv1d、conv2d | 支持受限 stride、padding、dilation、groups |
| 平均池化 | avg_pool1d、avg_pool2d | 窗口、步长、padding 和边界除数显式指定 |

完整属性检查在 [benchmark_graph.py](../../scripts/baseline/benchmark_graph.py)；更多 JSON 见[案例目录](../../scripts/baseline/cases/)。

可以连接成 MLP、卷积小网络、残差、多分支、多输入融合和多输出图。张量形状与广播需匹配；不是任意参数组合都能编译。

## 规模与数据

| 项目 | 上限 |
|---|---:|
| 输入 / 命名输出 | 各 4 个 |
| 输入 / 输出总元素 | 各 256 |
| 张量 rank | 1–4 |
| 图节点 | 64 |
| 公开常量条目 | 32 |
| 常量元素 | 由 validator 按常量形状检查，最多 4096 |
| 模型 JSON | 128 KiB |

常量必须是有限实数或规则数组，绝对值不超过 1024。模型是静态无环图，仅支持推理；不支持训练、随机 forward、动态 shape 或加密数据决定的分支。

输入数据不放进模型 JSON。验收器生成零、有符号、固定随机和边界四组输入。

默认使用 packed 布局。显式 `--unified-chunk-period P` 可选择 4/8/16/32/64/128/256 的分块周期；当前还受最多 4 个模型输入块和 4 个输出块限制。增加逻辑元素上限不会自动增加密文容量。

## Python 和高层算子

[受限 Python / 算子图入口](operator-decomposition-and-python.md) 先把源码或高层算子分解为上述 JSON，再进入生成流程。可信本地PyTorch `nn.Module` 可通过显式清单和隔离FX捕获导入；ONNX、checkpoint独立导入尚未开放。

非线性函数必须给出近似方法、区间和参数。不会自动把 ReLU、Softmax 或 MaxPool 替换成其他函数。

## 旧输入兼容

原 CLI 的 `--case` 保留以下格式；新应用组件只直接接收统一图。

| schema | 内容 |
|---|---|
| 1 | 8 类内置模型，各 6 个配置 |
| 2 | 单输入自定义图，4 个输入元素 |
| 3 | 2–4 个输入，每个 4 个元素 |
| 4 | 单输入 5–16 个元素，分为 period-4 密文块 |
| 5 | 单输入 1–256 个元素，可变 packed 周期 |

旧 schema 的算子和输出布局限制各不相同，不能套用统一图的全部能力。新项目优先使用统一图；旧文件可以继续使用原格式。

## 中型分块输入 v2

格式 `poseidon-model-graph-v2` 在原图字段之外增加 `resource_profile`：

~~~json
{
  "format": "poseidon-model-graph-v2",
  "resource_profile": "medium-v1",
  "id": "medium_example",
  "inputs": [{"name": "x", "shape": [1024]}],
  "constants": {},
  "nodes": [{"id": "m", "op": "mean", "inputs": ["x"],
             "attrs": {"axes": [0], "keepdims": true}, "outputs": ["y"]}],
  "outputs": [{"name": "mean", "value": "y"}]
}
~~~

输入/输出各最多1024元素、各4个逻辑张量，物理输入/输出分别最多4块。自动选择可容纳的period，或用 `--unified-chunk-period 256` 指定。资源、块偏移、有效尾部和selector参与绑定。见 [capacity-v2案例](../../scripts/baseline/cases/capacity-v2/)。

18模型覆盖逐项、全mean/偏置尾块、密文标量广播、双输入、对齐concat、reshape，各3上下文通过Poseidon数值验收；257元素模型另通过自动P128布局。

medium中间张量/节点/权重分析预算与生成预算分开：公开请求128KiB、常量注册256项，源码/工作量、深度及四块I/O限制继续检查。公开数组按广播形状绑定为周期向量；Linear支持跨输入块、尾部与batch边界，并选择逐行归约或旋转加权求和。

公开张量、跨块Linear、Linear→square→Linear各三个上下文通过数值验收。batched Linear的[128,4]→[128,4]、[1,192,3]→[1,192,3]、[1,2,128,4]→[1,2,128,4]通过；矩形batch的[1,130,3]→[1,130,5]、[1,2,64,4]→[1,2,64,6]和[1,1,65,5]→[1,1,65,7]通过行宽重排及分块映射，三个包可重载。范围见[Linear容量验收](poseidon-capacity-tiled-linear-r430.md)。

基线已验收分块Linear的typed helper复用。跨块 concat、stack、sum、mean 支持静态轴与有效尾块；三个轴/形状上下文分别通过数值验收，另有内轴 stack 及两个输入、三个命名输出的组合回归。接入仍受256个公开常量向量和1024项候选工作预算约束。见[张量容量验收](poseidon-capacity-tensor-acceptance-r441.md)。跨块Conv/Pool已有代表性验收；其他形状/切片组合和Agent定向组合继续开发。resnet20-v1的4096元素物理IO已有代表性验收；完整网络和大中间张量继续开发。

旧v1、schema1–5保持原格式、容量和哈希。新功能Agent生成独立评测。[验收报告](poseidon-capacity-acceptance-r386.md)


## 跨密文空间算子

medium-v1已验收跨块Conv1d/2d（含分组/深度卷积）、AvgPool1d/2d及代表组合。当前公开张量版本为v7，包含池化流与并行行归约；v5起的文件编码支持重复向量无损压缩。各版本按对应原请求与绑定重建。[范围与结果](poseidon-capacity-spatial-acceptance-r449.md)。当前结果为规则基线，真实Agent验收另计。

## PyTorch中型组合验收

可信本地FX已映射sum／mean、cat／concat／concatenate、stack，轴和张量列表在捕获时固定。516输入元素的归约融合、260元素的Linear融合已通过Poseidon CPU数值验收与来源包重载。578元素空间残差已编译，仍因Runtime中间密文保留导致分配失败；释放补丁待批准。dot、mm、bmm及两密文张量广播新增各三个上下文。资源配置上限和具体组合可执行性分别检查。[最新范围](poseidon-pytorch-broadcast-acceptance-r508.md)。


## 4096元素配置

poseidon-model-graph-v2选择resnet20-v1，使用unified-capacity-chunked-inputs-v3。输入／输出各最多4096逻辑元素、各最多16物理块，最多4逻辑输入和4命名输出。各张量独立补零，补零计块；4096元素采用period256。

已验收逐项加乘／平方、全sum/mean、非整块尾部、跨输入均值融合、不同shape四输入四输出、公开构造入口。旧medium-v1和capacity V2仍最多四块，历史请求按原ABI重建。源码／常量／工作／深度／沙箱限制保持。其他卷积／布局组合、大中间张量、权重容量和真实Agent生成分别验收。[4096元素报告](poseidon-extended-capacity-acceptance-r518.md)。

## 显式大周期 V4

`resnet20-v1` 可选择512/1024/2048/4096周期，使用 `unified-capacity-chunked-inputs-v4` 和公开张量v8。逻辑输入／输出各4096元素及16物理块限制保持，增加的周期同时绑定rotation密钥、公开常量、输入复制和输出解码。已验收逐项、切片／归约、尾块、多输入多输出及原尺寸首层诊断。配置 `poseidon-cpu-eva-w45-compact-v1` 为大常量IR提供精确文本压缩，保留原始IR和新请求身份。[详细范围](poseidon-wide-trace-acceptance-r574.md)。


## 通用张量配置

统一图 v2 使用 `resource_profile: "general-tensors-v1"`：

| 项目 | 上限 |
|---|---:|
| 输入／输出张量 | 各 4 |
| 输入／输出总元素 | 各 4096 |
| 张量 rank | 1–32 |
| 单个中间张量元素 | 65536 |
| 原始／分解后节点 | 1024／4096 |
| 公开权重总元素 | 1000000 |
| 输入／输出密文块 | 各 16 |

元素、节点、权重和物理资源与既有最高配置相同。 已编译程序包可进行[资源分析](application-agent-component-v1.md#编译程序包的资源分析)，检查中间密文存储下界；逻辑容量、编译成功及实际内存分别判断。Conv1d/2d、Pool 等保留各自的 rank 要求；逻辑张量 rank 与 Expr 对象数组的构造维度分别检查。静态 transpose/permute 通过固定公共路由映射逻辑 C-order 元素，包括跨块和尾块。

使用新配置后，编译、密态执行和数值验收仍分别进行。旧 legacy-v1、medium-v1、resnet20-v1 的资源身份和请求格式保持。

## 命名输出共享

`poseidon-model-graph-v2` 的 `outputs` 可用多个名称引用同一个 `value`。应用选择 `options={"output_binding_policy": "exact-values-v1", "generation_guidance": "explicit-v15"}`，CLI 增加 `--output-binding exact-values-v1 --unified-guidance explicit-v15`，即可共享相应的物理输出块。

逻辑元素上限按全部名称计算；密文块上限按首个结果的块数计算。每个名称保留 shape、有效尾部和返回位置，嵌套输出结构仍由来源绑定保存。不同 `value` 分别绑定。旧布局与默认行为保持。

ARM Poseidon CPU 的 period=4 已验收同值重复、交错别名和结构化输出，各 3 个上下文；规则12/12（含3个兼容例）、真实Agent9/9首轮通过。其他配置周期及最大rank/容量不计为本次密态通过。[结果](agent-output-alias-acceptance-r1486.md)。

自动布局规划同样读取共享策略，以实际输出块数筛选周期。period=16/32 的三个代表上下文已分别通过规则与真实Agent数值验收；它们复用已有数学模型。[自动布局结果](agent-output-alias-planner-acceptance-r1488.md)。

## 通用输入扩展

`general-tensors-v1` 的逻辑 rank 为 1–32，输入和输出各最多 4096 元素、4 个张量，单个中间张量最多 65536 元素。最高已有节点、权重和物理预算沿用；各算子分别检查 rank、轴与布局。rank 增加不扩大元素容量或乘法深度。

可信 PyTorch v2 提供工厂参数、显式 float32→float64 及静态输出结构；来源清单最多 8 个平铺 Python 文件、总源码 64 KiB、工厂参数 8 KiB、输出 4 个张量叶子。旧 JSON、schema 1–5 与 v1 源码清单保留原接口。

`descriptors-compact-origins-v3` 适用于布局映射较多的图：描述符和类型清单使用固定引用，映射表使用可还原的列格式。执行绑定仍采用原始公开数组，生成上下文与可信来源的大小限制保持。来源紧凑JSON保留声明顺序。[高rank/容量验收](agent-generic-capacity-offline-acceptance-r1505.md)列出当前通过范围及来源记录阻塞。

声明式图的 NumPy 权重包可选 `descriptors-v1`。可信请求上限 4 MiB，提示上下文与候选各 128 KiB；原权重参与哈希和 trusted tracing，生成模型使用公开常量绑定。该选项属于请求能力，与架构、工具链裁剪、验收等级分别配置。

[使用说明](operator-decomposition-and-python.md#参数化-pytorch-模型) · [当前验收](agent-generalization-acceptance-r622.md)。

## 逐节点 PyTorch 近似

可信本地 PyTorch v5 在固定输入签名、公开权重和原资源配置下，按节点、模块和算子默认值选择近似。支持共享模块调用和登记的函数式写法，统一输出原图、分解图与选择来源；完整配置见 [PyTorch 输入说明](operator-decomposition-and-python.md#逐节点与模块近似)。新增规则／人工表达验收见 [r721](agent-pytorch-approximation-acceptance-r721.md)，原 LayerNorm 规则编译失败单列保留。

## PyTorch 静态索引

可信 PyTorch v6 支持正步长切片、整数与负整数位置、单个省略号、单元素轴、select/narrow 和单轴公开整数向量；index_select 支持位置及关键字调用。固定签名、公开状态、NPY 权重和输出结构沿用原接口，v6 容量图采用 v9 路由。各项元素、节点和物理预算保持。

15 个索引模型及两个重复输出回归已有规则候选的 Poseidon CPU 数值验收；新入口真实 Agent 生成单列。[输入用法](operator-decomposition-and-python.md#静态张量索引) · [验收范围](agent-pytorch-index-acceptance-r761.md)。

## PyTorch 静态矩阵规范化

v7 接入向量、矩阵和多维批次广播，支持固定公开向量、矩阵及批量矩阵。数学计算规范化到现有高层／基础图，实际大小按资源配置检查。输入／输出、节点、权重与物理密文上限保持；JSON 高层 matmul 的原 rank 约束保留。30例规则候选已完成 Poseidon CPU 数值与复用验收，新入口真实Agent评测另计。[用法](operator-decomposition-and-python.md#pytorch-矩阵运算) · [结果](agent-pytorch-matrix-acceptance-r815.md)。

## PyTorch 实例归一化

v7 来源清单接入 InstanceNorm1d／2d 和函数式 instance_norm。模块支持带 batch 与单样本；函数支持 rank 3–4 的 N,C,spatial 输入。公开 affine、冻结统计量及完整 NPY 状态由可信工具绑定；当前输入统计量指定 rsqrt 近似。30 个来源上下文已通过规则候选 Poseidon CPU 编译、数值与代表包复用，真实 Agent 生成单列。

输入元素、输出元素、节点、权重与物理资源上限沿用原配置。[用法](operator-decomposition-and-python.md#pytorch-instancenorm) · [范围与结果](agent-pytorch-instance-acceptance-r847.md)。

## JSON 静态矩阵输入

显式格式 `poseidon-operator-graph-v2` 支持向量、矩阵、多维批次广播及固定公开矩阵两侧乘法，并复用 PyTorch v7 的静态矩阵规范化。形状、元素、权重和密文资源按所选配置检查；点积标量采用一个元素的张量。旧 v1 范围保持。

36 例已通过导入与独立参考，新格式的真实编译、密态和 Agent 验收待运行。[格式与示例](operator-decomposition-and-python.md#json-矩阵计算图) · [当前结果](agent-json-matrix-static-acceptance-r913.md)。


## 公开量算术输入

JSON 高层图 v3 和可信 PyTorch v8 支持加、减、乘的公开标量／固定张量左侧写法及合法广播。PyTorch 同时接入运算符、函数和 Tensor 方法，源模型、权重及规范化记录绑定到导入包。两端已通过公开导入与独立明文参考检查；编译、密态和真实 Agent 验收单列。[用法](operator-decomposition-and-python.md#公开量与张量算术)。

现有资源配置、算子 rank／轴规则和物理限制保持。JSON v1/v2、PyTorch v1–v7 与旧 schema 接口沿用原行为。

## PyTorch v10 算子结果接口

可信清单使用 `poseidon-trusted-pytorch-v10`，沿用工厂、NPY权重、dtype、输入/输出与资源配置。结果形状查询扩展到AdaptiveAvgPool、InstanceNorm和padding；函数式AdaptiveAvgPool映射到现有分解。旧v1–v9行为保持。新来源、参考和来源包具有独立版本与哈希绑定。

[范围与用法](operator-decomposition-and-python.md#pytorch-v10-算子结果与形状查询) · [接入检查](agent-pytorch-operator-shape-v10-acceptance-r994.md)。本次没有扩大资源、算子数学、参数或后端执行范围。

## PyTorch v11 常用调用

v11 来源清单接入静态 alpha、eval Dropout 系列和融合 Attention，允许由 Linear、形状操作产生 Q/K/V；支持公开布尔、有限加性及因果 mask。近似配置、资源与输出结构沿用现有接口，旧版本行为保持。

当前公开接入 33/33、528 组明文参考；应用入口 8/8。编译、Poseidon 密态和真实 Agent 生成均为待验收。类型、shape、布局、物理容量继续分别检查。[用法](operator-decomposition-and-python.md#pytorch-v11-常用调用) · [当前范围](agent-pytorch-common-v11-acceptance-r1045.md)。

## PyTorch v21 幂运算与公开除法

可信本地来源支持指数 2／4 的静态幂和有限非零公开标量除法，包括 Tensor 原地形式、
标量中间值及共享视图写回。公开状态除数采用 float64 标量 NPY 哈希绑定；
除数和倒数的绝对值各不超过 1024。来源清单默认保持 v18，v21 显式选择。

八个新来源模型与两个兼容模型已有规则候选的 Poseidon CPU 数值及程序包复用证据，
各次编译绑定分别保留；本批真实 Agent 生成数为 0。
[输入写法](operator-decomposition-and-python.md#pytorch-v21-幂运算与公开除法) ·
[验收结果](agent-pytorch-exact-arithmetic-acceptance-r1348.md)。

## PyTorch 公开常量表达式

可信本地来源 v23 支持公开张量的形状变换、静态切片和布尔转浮点，v24 增加 float/double/to 类型转换，v25 增加公开加、减、乘、除和取负。公开值由 NPY 状态权重绑定；输入相关计算保留在图中。默认来源为 v18，latest=v28；各版本也可显式选择。v23/v24 来源的真实生成及原样重放结果见[来源验收](agent-source-intake-phase-summary-r1426.md)；v25 的[规则验收](agent-public-arithmetic-offline-results-r1386.md)与[真实生成](agent-public-arithmetic-live-acceptance-r1388.md)分别记录。[写法与限制](operator-decomposition-and-python.md#pytorch-v25-公开常量算术)。


## PyTorch v28 静态广播

`poseidon-trusted-pytorch-v28` 是当前 `latest`。支持 `Tensor.expand_as` 和 `Tensor.broadcast_to`，与 v27 的结构化输入、空间算子、公开权重和命名输出共用接入流程。目标形状为公开静态数据，按已开放的 dtype、shape、alias、轴与资源规则检查。

广播加法、转置视图加别名写入与归约、标量均值广播接 Linear 这三个数学上下文，合计六个来源写法，均已通过当前来源参考、真实生成或原样重放、Poseidon CPU 数值验收与程序包复用。[用法](operator-decomposition-and-python.md#pytorch-v28-静态广播) · [验收](agent-static-broadcast-phase-summary-r1433.md)。

## 静态张量变换输入

JSON 高层图 v5 和显式可信 PyTorch v30 接收 repeat/tile、repeat_interleave、flip、roll。数据为加密浮点张量，次数、轴、位移和 output_size 为公开静态值。首版 repeat_interleave 使用统一的正整数次数，flip 采用互不重复的轴。rank、元素、分解节点和物理密文资源分别按所选配置检查。

每种运算三个上下文已完成真实 Agent 生成和 Poseidon CPU 数值比较；18 项规则候选包含 3 项旧 v29 兼容。源规范化保留复制、别名和后续写入的语义，程序包已通过重载、新输入、重复执行和缓存复用。[范围与用法](operator-decomposition-and-python.md#静态重复与重排) · [验收](agent-static-tensor-transforms-acceptance-r1454.md)。默认 v18、latest v28，新增范围通过 v30 显式选择。

## PyTorch v31 公开参数变换

显式 `poseidon-trusted-pytorch-v31` 支持公开状态的 repeat/tile、repeat_interleave、flip、roll。
支持 float32、float64、int32、int64、bool；逻辑 dtype、显式舍入、复制语义和派生数组分别绑定。
次数、轴和位移使用公开静态值，分配前检查所选资源配置的 rank、轴和公开常量总量。

v31 沿用密文张量变换、源码包、结构化输入及输出。15 个新来源上下文及 3 个原 v30 来源
完成规则候选 Poseidon CPU 数值与程序包验收；15 个新上下文的真实 Agent 全部首次通过。
本批实际密态形状为 rank 1/2，较大配置分别验收。
默认 v18、latest v28 保持。[用法](operator-decomposition-and-python.md#pytorch-v31-公开参数变换) ·
[结果](agent-public-tensor-transforms-acceptance-r1458.md)。


## PyTorch v32 公开常量构造

显式 `poseidon-trusted-pytorch-v32` 接收 `torch.zeros_like/ones_like/full_like` 和
`Tensor.new_zeros/new_ones/new_full`，按静态公开形状、dtype 和有限填充值构造常量。
支持 float32、float64、int32、int64、bool，分配前检查 rank、轴和剩余常量总量。
公开数组与来源、逻辑 dtype、舍入结果及内容哈希绑定。

每种调用三个上下文完成真实 Agent 生成、Poseidon CPU 数值和程序包复用；
规则候选另含三个原 v31 来源兼容。真实密态范围为 rank 1/2，rank 32 和资源边界有独立控制测试。
默认 v18、latest v28 保持。[用法](operator-decomposition-and-python.md#pytorch-v32-公开常量构造) ·
[结果](agent-public-constants-acceptance-r1463.md)。

## 显式容量布局 v10

CLI 使用 `--unified-packing capacity-public-tensors-v10`，组件使用
`options={"capacity_packing_version": "capacity-public-tensors-v10"}`。
它与 `--source-format` 分开选择，模型元素、rank、节点和密文资源上限沿用所选资源配置。

v10 按既有预算选择较浅的 Linear 路由，合并广播的块内旋转，支持完整末轴行的非二幂归约。
归约行须完整位于一个输入块；跨块行使用既有路由。
`planning_costs` 标注局部估计、常量计数范围和未知项；编译及执行状态由真实验收确定。

当前规则基线覆盖 10 个不同模型，包含两个原失败模型、广播尾块、宽度 3/5 归约、跨块归约和矩形 Linear。
[验收记录](agent-capacity-shallow-acceptance-r1539.md)。
