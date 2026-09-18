# Agent 支持的输入、算子、模型与配置

代码核对日期：2026-09-18。此表描述**当前输入校验器和运行器接受的范围**，
不表示所有形状、组合或构造语法均已完成真实 Agent/密态测试，更不是全输入形式化等价证明。
运行环境以 Ubuntu x86_64 为准；启动方式见 [Agent README](../../scripts/README.md)。

## 1. 输入文件与通用约束

公共入口是数据型 JSON，不直接接收任意 PyTorch `model.py`、ONNX、`.pt`、
pickle 或自然语言。程序按 JSON 构造受信任的 CPU float64 PyTorch reference。
固定公开权重、加密输入、静态 shape、推理模式；不支持训练及加密数据相关分支。

| schema | 输入 | 输出与布局边界 |
|---|---|---|
| 1 | 内置 8 类模型，configuration 0–5；通常 [4]，flatten_linear 为 [2,2] 或 [1,4] | 预设模型确定输出，最多 4 个逻辑元素 |
| 2 | 自定义单输入；允许 shape 为 [4]、[2,2]、[1,4]、[1,2,2]、[2,1,2]、[1,1,4] | 最终必须为 1–4 元素向量；空间输出需显式 flatten |
| 3 | 2–4 个独立加密输入；每个输入 shape 取同 schema 2 集合 | 同 schema 2；输入次序由 inputs 列表声明 |
| 4 | 单逻辑张量，rank 1–4，总元素 5–16，各维 1–16 | C-order 分为 2–4 个 period-4 密文；最终向量 1–4 元素 |
| 5 | 单逻辑张量，rank 1–4，总元素 1–256，各维 1–256 | packed 输出最多 256 元素；转为 scalar-neuron 布局后最多 16 个元素/密文 |

Schema 5 的 slot period 取能容纳输入的最小值：4、8、16、32、64、128、256；
先按 C-order 展平并补零。不能把这个 period 当作 CKKS 总 slot 数。
所有输入 shape、输出 shape 与真实 layout 仍需同时通过检查。

自定义图通用规则：

- 字段严格检查；节点按依赖顺序排列，不允许前向引用、环或重复名称。
- 最多 64 个节点、32 项公开常量。常量为有限实数或规则数组，绝对值不超过 1024。
- 常量 rank 最多 4；schema 2/3/4 每项常量最多 128 元素，schema 5 为 4096。
- 单 case JSON 最多 128 KiB。模型不接受任意测试输入数组；测试器构造零输入、
  固定有符号输入、固定种子随机输入和范围边界输入。
- schema 4 会继续检查展开后的图，不能只看原图未超过 64 节点就保证通过。

## 2. 输入算子矩阵

“支持”始终附带类型、shape 和布局检查；schema 1 是预设模型而不是自由算子图。

| op | schema 2/3 | schema 4 | schema 5 | 节点附加字段 |
|---|---|---|---|---|
| add / subtract / multiply | 支持 | 支持 | 支持 | 无 |
| negate / square | 支持 | 支持 | 支持 | 无 |
| power | 仅 2、4 次幂 | 同左 | 同左 | exponent |
| linear | 输入须一维，输出宽度 1–8 | 输入须一维，输出宽度 1–8 | 作用于最后一维，输出总元素最多 16 | weight、bias |
| flatten | 展平全部逻辑维度 | 同左 | 同左 | 无 |
| reshape | C-order、元素数不变 | 不支持 | C-order、元素数不变 | shape |
| rotate | 仅仍为 packed [4] 的值 | 不支持 | 不作为输入模型算子开放 | step |
| batch_norm | 固定统计量推理 | 不支持 | 固定统计量推理 | running_mean、running_var、weight、bias、eps |
| concat | 1–8 个密文张量，中间结果最多 8 元素 | 不支持 | 1–8 个密文张量，结果最多 16 元素 | axis |
| conv1d / conv2d | 有界、无 batch、channel-first | 不支持 | 可有前导 batch，输入总元素最多 256，输出总元素最多 16 | weight、bias、stride、padding；可选 dilation、groups |
| avg_pool1d / avg_pool2d | 有界平均池化 | 不支持 | 可有前导 batch，同空间预算 | kernel、stride、padding、count_include_pad |
| permute | 不支持 | 不支持 | 静态逻辑维度排列 | dims |
| transpose | 不支持 | 不支持 | 静态交换两个逻辑维度 | dim0、dim1 |

### 必须遵守的语义细节

- 算术左操作数必须是密文；右操作数可以是密文或公开常量。两密文的 shape
  和布局必须一致，不自动重排。
- schema 2/3/4 的明文广播仅允许标量、长度 1 或匹配 shape；
  schema 5 支持向尾部维度对齐的明文广播，但不能扩大密文 shape。
- schema 2/3 的多维 packed 输入做普通算术前通常须 flatten；schema 4
  多维输入也须先 flatten 再做向量算术。schema 5 不受这个旧版四元素限制。
- Linear 是真实跨元素加权归约：权重 [out,in]，bias 为 [out] 或 null。
  schema 5 支持 [...,in] → [...,out]，但整个输出而非单行仍最多 16 元素。
- rotate 的 step 仅为 -3、-2、-1、1、2、3；正步长是左移读取
  `y[j] = x[(j+step) mod 4]`。schema 5 生成的 DSL 仍会使用 rotation，
  只是 JSON 输入层没有开放独立 rotate 节点。
- reshape 可有一个 -1 推断维度，元素总数必须不变；不表示任意重新 packing。
- concat 除拼接轴外各维必须匹配，允许合法负 axis，不进行广播。
- BatchNorm 按 N,C,...、channel axis=1；公开统计量折叠成仿射常量。
  eps 在 [0,1]，variance 非负且 variance+eps>0；weight/bias 可为 null。
  这不是训练 BatchNorm，也不执行密文开方或除法。
- Conv 是 cross-correlation，不翻转卷积核；明确零 padding、逐轴 stride、
  dilation 和合法 groups。schema 2/3 的空间输入/输出总元素均最多 4；
  schema 5 为输入最多 256、输出最多 16，batch 也计入总预算。
- AvgPool 的边缘除数由 count_include_pad 明确指定；不支持 ceil_mode=True、
  任意 divisor_override 或池化 dilation/groups。

## 3. 模型族与六种预设配置

| 基础模型族 | 数学结构 |
|---|---|
| affine | x*w+b，逐元素仿射 |
| polynomial | x²*w+x*b+c，显式二次多项式 |
| linear | Linear(4,out)，包含归约 |
| mlp2 | Linear(4,a) → square → Linear(a,out) |
| mlp3 | Linear(4,a) → square → Linear(a,b) → square → Linear(b,out) |
| fanout | square(x*w+b)+square(x)*c |
| residual | x+square(x*w+b)*c |
| flatten_linear | flatten → Linear(4,out) |

| configuration | a | b | out | flatten_linear 输入 |
|---|---|---|---|---|
| 0 | 1 | 2 | 1 | [2,2] |
| 1 | 2 | 3 | 2 | [1,4] |
| 2 | 3 | 2 | 3 | [2,2] |
| 3 | 4 | 2 | 4 | [1,4] |
| 4 | 3 | 4 | 2 | [2,2] |
| 5 | 4 | 4 | 4 | [1,4] |

权重和输入由固定种子及固定候选数值构造，不代表六种任意外部权重文件。
这 8×6=48 例是基础目录。affine/polynomial/fanout/residual 不使用 a/b/out
作为网络宽度，它们的 configuration 主要改变固定权重。

`--extended` 选择 16×6=96 例目录，额外八类如下。
新增族以完整 schema 2/3 图表达，不能把名称直接写进 schema 1 的 family。

| 扩展模型族 | 结构 |
|---|---|
| conv1d / conv2d | 小型卷积后展平 |
| avg_pool1d / avg_pool2d | 不同窗口、stride、padding、边缘除数的平均池化 |
| conv_poly_pool | Conv → square → AvgPool |
| dual_affine | 两个独立加密输入分别加权后融合 |
| dual_bilinear | 两个独立加密输入相乘后 Linear 归约 |
| dual_linear | 两输入作差 → Linear → square |

自定义图还可表达：固定统计量 BatchNorm、concat 多分支、静态 reshape、
transpose/permute、不同 rank/batch 的有界 Linear/Conv/Pool、分块 Linear、
重复多项式 block。图被接受不保证其乘法深度与 CKKS 参数相容。
ReLU、SiLU、MaxPool、Softmax、任意 Attention、完整 ResNet/LeNet 不是可直接输入的
受支持模型名称；不自动做语义近似或宣称完整模型支持。

## 4. 已准备的专项目录

以下文件均位于 `scripts/baseline/cases/`。数字是清单长度，不是通过率；
目录间可能有重叠，不能相加当成独立验证模型数量。

| 清单文件 | 例数 | 模型 schema |
|---|---|---|
| advanced-agent-12-manifest.json | 12 | 2 |
| advanced-shapes-manifest.json | 10 | 2 |
| batch-norm-agent-6-manifest.json | 6 | 2 |
| broadcast-agent-manifest.json | 5 | 2 |
| chunked-input-10-manifest.json | 10 | 4 |
| concat-agent-6-manifest.json | 6 | 2/3 |
| concat-axis0-sensitive-1-manifest.json | 1 | 3 |
| construction-exercises-30-manifest.json | 30 | 2 |
| encrypted-zero-manifest.json | 7 | 2/3 |
| layout-guidance-6-manifest.json | 6 | 5 |
| logical-reshape-agent-3-manifest.json | 3 | 2 |
| native-array-exercises-10-manifest.json | 10 | 2 |
| native-function-exercises-11-manifest.json | 11 | 2/3 |
| native-star-exercises-8-manifest.json | 8 | 2 |
| object-arithmetic-exercises-10-manifest.json | 10 | 2 |
| object-unary-exercises-8-manifest.json | 8 | 2 |
| packed-composition-12-manifest.json | 12 | 5 |
| packed-input-12-manifest.json | 12 | 5 |
| packed-native-6-manifest.json | 6 | 5 |
| packed-spatial-14-manifest.json | 14 | 5 |
| packed-spatial-16-manifest.json | 16 | 5 |
| recent-semantics-agent-20-manifest.json | 20 | 2 |
| scalar-conversion-exercises-15-manifest.json | 15 | 2 |
| semantic-gap-manifest.json | 6 | 2/3 |
| tensor-input-12-manifest.json | 12 | 5 |
| tensor-permutation-12-manifest.json | 12 | 5 |

普通自定义清单格式是 `{"schema":1,"cases":[...完整模型对象...]}`，1–96 例，
只接受模型 schema 2/3/4/5；文件最多 1 MiB，id 不得重复。
针对构造语法的专项目录需要匹配其版本化构造模式，不能仅换清单文件就认为覆盖了该语法。

## 5. 编译、API 和执行配置

| 配置 | 可用值 / 默认 |
|---|---|
| provider | 仅 deepseek |
| model | deepseek-flash（默认）、deepseek-v4-flash、deepseek-v4-pro |
| reasoning-effort | low / high / max，默认 high |
| max-tokens | 当前 CLI 默认 384000；允许 1–384000，不是无限预算 |
| api-timeout | 默认及最大 1200 秒，每次请求独立计时 |
| max-repairs | 单例 0–3，默认 3；初次生成加最多三轮语义修复 |
| provider-retries | 0–3，默认 0；README 示例显式选 3；与语义修复分开计数 |
| jobs | 批次 1–10，默认 10；现有 native 密态执行槽数为 2 |
| stream | 默认关闭，传 --stream 开启 |
| loopback-proxy-port | 默认 0，不走显式代理；仅接受当前 Ubuntu 的本地端口 |
| 整体 timeout | 启动器默认 43200 秒；范围 1–604800 秒 |
| compiler-configuration | seal-cpu-eva-w40-v1 或 seal-cpu-eva-w45-v1 |

两项 compiler configuration 都是固定 profile 的 Dacapo EVA → upstream SEAL
HEVM CPU；waterline 分别为 40 和 45。不指定时保留旧 waterline40 请求规则。
这不是安全等级开关，Agent 不得自行修改 profile、scale/level 或编译配置。
继续旧批次时不允许悄悄更换固定编译配置。

模式分别为：`--prepare`（只准备请求）、`--self-test`（脚本化自测）、
`--replay`（重放回答）、`--golden-file`（人工程序）、`--live`/`--deepseek`
（真实付费生成）。一轮生成不等于一次 HTTP 尝试，超时重试可能再次计费。

## 6. DSL 构造模式不是输入模型算子

基础数学图和生成程序的 Python 构造语法是两层。当前入口额外提供：

| 模式参数 | 受限构造能力 |
|---|---|
| --extended-arithmetic | 反向算术、重绑定、增量赋值 |
| --public-construction | 公开有界循环、条件、容器 |
| --function-composition / --closures | helper、嵌套作用域、闭包 |
| --call-binding / --function-literals | 受限参数绑定、默认值、函数值/lambda |
| --public-iteration / --public-sequences / --public-mappings | 公开迭代、序列、映射 |
| --public-numbers / --public-control / --public-strings | 公开数值、控制及字符串处理 |
| --public-polynomial | 公开 Chebyshev/GenPoly 系数构造 |
| --object-arrays / --object-arithmetic / --object-unary / --scalar-conversion | 有界对象存储、算术、单目和标量提取 |
| --native-functions | typed @hc.func helper、真实 frontend 调用 |
| --native-arrays / --native-starred | Expr 对象数组、位置参数展开 |
| --native-array-arithmetic / --native-public-loops | 对象数组算术、公开 range 循环 |
| --native-scalar-augmented / --native-array-mutation | 标量重绑定、别名感知数组原地操作 |

这些是不同版本契约，部分包含前置模式，部分互斥；不能任意同时打开。
schema 5 的 packed-native 还有自己的规则，不代表所有历史 public-construction
契约的并集。最终以 request.json 中固定的 task/rules/layout 为准。

## 7. 验证边界与源码依据

误差判据逐项固定为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
同时记录 MAE、最大绝对误差、非零 reference 的相对误差及适用的 cosine similarity。
不能只凭分类标签或编译通过判断语义正确。

输入接受、规则转换成功、真实 Agent 生成成功、真实密态执行成功是不同门禁。
本清单没有新发起付费调用，也不将 SEAL CPU 的结果当作 Poseidon GPU 通过。
完整 DSL、bootstrap、任意控制流和无限 shape 尚不支持。

源码：`model_catalog.py`、`expanded_model_suite.py`、`model_graph.py`、
`chunked_model.py`、`packed_model.py`、`packed_input_abi.py`、
`spatial_ops.py`、`packed_spatial.py`、`batch_norm_ops.py`、
`logical_reshape.py`、`concat_ops.py`、`custom_batch_manifest.py`、
`compiler_configuration.py`、`run_candidate.py`、`run_agent_batch.py`。
以上均在 `scripts/baseline/`。

固定 Dacapo commit 之外的本地前端修复需按
[补丁说明](../../scripts/baseline/patches/README.md) 重放并重建。
