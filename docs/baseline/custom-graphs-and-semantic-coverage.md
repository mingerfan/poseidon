# 自定义模型输入与语义覆盖：阶段记录（2026-09-07）

GPU 后续更新：[隔离 CUDA 构建与真实 GPU CKKS Add/drop 验证](poseidon-gpu-modswitch-primitive.md)
已通过；下文“尚未安装 CUDA”是历史状态，不再是当前安装门禁。
HEVM opcode 对接和 Agent 的 Poseidon GPU 端到端验证仍未完成。

**历史阶段快照**：下文记录最初 schema-2 接入时的限制，不代表最新能力上限。
后续已验证 13 种图算子、16 类 / 96 例确定性 CPU 密态基线，并新增多输入、
带符号旋转、受限 Conv/AvgPool、近似误差分解和隐藏宽度 5..8。
当前入口及边界见 [使用说明](agent-quickstart.md)、
[96 例基线](expanded-96-models-cpu.md)、
[宽隐藏层](wide-linear-cpu.md) 和
[近似误差分解](approximation-error-decomposition.md)。
这些新增证据不等于新在线 Agent 批次通过，也不等于 Poseidon GPU 已通。

后续增量：[schema-3 多输入用户图已接通隔离候选验证路径](schema3-multi-input-agent-path.md)。
下文 schema-2 的单输入边界仍保持；新增契约的真实 LLM 生成尚未验证。

## 本阶段结论

后续增量：原生减法/取负已在版本化 v1 契约中实现，见
[原生算术验证及 frontend 修复](native-hecate-arithmetic-v1.md)。
下文关于 v0 不允许 `-x` 的描述仍成立；新 schema-2 请求默认选择 v1，
含 rotate 的请求选择 v2。后续带符号旋转和实际密钥检查证据见
[旋转语义 v2](signed-rotation-v2.md)。

**Confirmed fact**：输入不再只能选择 48 个 catalog case。`schema: 2`
允许用户直接描述一个受限的静态有向计算图、连接关系及公开固定权重，
无需导入用户 Python，不加载 pickle/checkpoint，不执行 JSON 中的代码。
仍然通过 PyTorch/FX 分析，最终生成目标仍为 Hecate Python 函数片段。

**Confirmed fact**：自定义 `-(x-offset) -> Linear(4,2)` 的规则基线及独立人工
golden 均已通过 Dacapo 编译和真正的 SEAL CPU 密态执行；缺失归约的反例
能编译、能密态运行，但被独立差分检查拒绝。本阶段没有新付费 LLM 调用。

**未完成**：算子/模型类型翻倍、多输入、广泛 shape/layout、Conv/Pool、
多项式激活近似的双误差报告，以及 Poseidon GPU 执行。目标仍在进行，
不能把本阶段的自定义输入接口称为完整目标完成。

## 凭据只有一个文件

唯一默认凭据文件是执行端项目根目录下的 `.env`，当前只支持 DeepSeek：

```dotenv
DEEPSEEK_API_KEY=
```

非空进程环境变量 `DEEPSEEK_API_KEY` 可显式覆盖文件中的同名变量。
不读取其他凭据文件；模板 `.env.example`
不含真实 key；`.env` 已被 Git ignore。忽略规则不是加密或文件访问权限。

## schema-2 输入规范与运行

完整可编辑例子：`scripts/baseline/cases/custom-subneg-linear.json`。

- 顶层字段严格为 `schema,id,input_shape,constants,nodes,output`。
- `schema` 必须是整数 2；`id` 是安全标识符，不参与模型构造逻辑。
- `constants` 直接提供有限实数 scalar/vector/matrix；每个数组最多 128 个
  元素，数值绝对值不超过 1024，最多 32 个数组。矩阵在 lowering 时拆成行。
- `nodes` 按拓扑顺序，最多 64 个节点；节点引用 `x`、公开数组或前面的值。
  不允许前向引用、环、重定义、私有文件引用或外部命令。
- 当前 graph op：`add,multiply,subtract,negate,square,power,linear,flatten,rotate`。
  `power` 只允许 2、4；`linear` 显式 `weight` 引用与可空的 `bias` 引用。
- `rotate` 显式提供 `step`，仅允许 ±1、±2、±3，作用于 shape `[4]` 的
  period-4 packed ciphertext；正数左旋。六个步长不是六种模型或算子。
- 输入仍是一个密文，逻辑 shape 只允许 `[4]`、`[2,2]`、`[1,4]`。
  多维输入必须先 full row-major flatten；此限制来自仍保留的已验证 ABI。
- Linear 首层输入宽度 4，输出宽度 1..4；后续 Linear 输入/输出宽度 1..4。
  标量/单元素公开量可广播，或公开向量与逻辑 shape 精确一致。
- 每个输出神经元仍为一个 ciphertext；不能把这种结果与 period-4 packed
  输入直接相加，检查器会拒绝没有显式 repacking 的连接。
- 权重会保存为 `weights.npz`，不使用对象反序列化。四组固定测试输入保留为
  零、有符号、固定种子随机、[-1,1] 边界；还不是用户自选输入域接口。

在 WSL 源码根运行，以下两条都不读 API key、不调用模型服务：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_model_batch.py \
  --case scripts/baseline/cases/custom-subneg-linear.json

python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-subneg-linear.json \
  --golden-file scripts/baseline/golden_cases/subneg_linear/golden.py
```

第二条走与 Agent 相同的 AST 白名单、bubblewrap、compiler、artifact gate、
加密执行和差分接口；`--golden-file` 是人工来源，不会被报告成 Agent 生成。

实际请求 Agent 时使用下面的显式 live 命令，它会向选择的 provider 发送
此文件定义的公开图和固定权重、DSL 规则；不发送私钥、测试输入或 reference：

```bash
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-subneg-linear.json \
  --provider deepseek --live
```

这条新自定义案例的 live 命令本阶段**没有执行**。旧 48 例结果不能代替它的
LLM 生成验证。原 API 服务调用、费用与限额规则保持，不自动改用规则答案。

## 减法、取负与独立 reference

schema-2/PyTorch 输入现在接受减法和取负。为了保留已有生成器安全契约，
翻译器先使用精确代数关系 `-a = a*(-1)`、`a-b = a+(-b)`；Hecate Agent
AST 本身依然只允许 `+`、`*`、`rotate(1/2)`。因此“模型输入支持取负”
和“Agent 能直接输出 `-x`”不是一回事。本阶段未增加原生 `-` AST 语法。

`model_graph.evaluate_reference` 使用独立的 Python 双精度运算和逐行点积，
不读取 FX、DSL 或解密结果。先与 PyTorch reference 对照，再冻结为密态比较
目标；行归约使用 `math.fsum`，可与 PyTorch 求和舍入略有不同，预比较门限为
1e-12。密态比较仍冻结为 `abs(actual-reference) <= 1e-5+1e-4*abs(reference)`。

本例 artifact 实际含 `NegateC`，说明 compiler 将乘以 -1 优化为 native
negation；不能承诺每个任意图都会发生该优化。普通乘以公开数可能影响 scale/
level，实际 artifact 和参数仍必须逐例检查。没有性能提升结论。

## 语义清单与测试映射

机器可读清单：`scripts/baseline/dsl_semantic_inventory.py`。
执行该文件只输出 JSON，不导入 native compiler、运行模型或加载凭据。
对应测试：`test_dsl_semantic_inventory.py`，检查测试目标存在和上游名称覆盖。

清单覆盖固定 Dacapo commit 的以下明确范围：Python 的全部 opcode 表、
全部 10 个 Earth op、全部 11 个 CKKS op、`poly.Func.py` 的全部 15 个 `HE_*`
wrapper，以及函数/I/O、常量、mutation、控制流、shape/layout、keys 等横切规则。
这是一份**可检查的范围清单，不是已经完成的所有组合行为规范或形式化证明**。
上游 `__all__` 中出现名字，不足以证明该 API 存在或在当前配置可用。

| 层面 | 现在的证据 | 未完成的验证 |
|---|---|---|
| 加法/乘法/平方/四次方 | 人工、规则、旧 48 Agent 案例的 CPU 密态结果 | 更大 shape/depth、更多参数 |
| 减法/取负 | 新 schema-2、独立 reference、人工 golden、真实 CPU 执行、归约反例 | native Agent AST 语法、系统性算术反例族 |
| rotation | 步长 +1/+2 与对应 key | 负步长、其他步长、缺钥反例、实际 GPU |
| 输入与布局 | 一输入、周期 4、标量输出密文列表 | 多输入、更大 shape、一般 broadcast/repacking |
| Linear/MLP | 首层4输入、宽度1..4，新增任意公开权重和连接 | 一般宽度、多 batch、新结构规模验证 |
| Conv/Pool/Concat/BN | 上游 wrapper 已列入清单，当前明确拒绝 | 独立语义、golden、编译、真实密态执行 |
| ReLU/SiLU/Max | 上游部分 helper 带多项式或 bootstrap | 批准的近似、误差分解、真实 bootstrap 路径 |
| rescale/modswitch/relin | SEAL profile 的 compiler/runtime 实验 | Poseidon GPU 低位宽物理参数映射 |
| bootstrap/upscale | AST/HEVM 门禁拒绝 | 实际安全 backend 验证，不能靠模拟解禁 |

计数分母冻结：旧模型族 **8**，旧模型输入算子语义 **6**（add、multiply、
square、power、linear、flatten），旧生成 DSL primitive **3**（+、*、rotate）。
目标为模型族至少 **16**、输入算子至少 **12**。DSL primitive 数单独观察，
不为凑数添加语言语法。当前输入 op 8 个，而 DSL primitive 仍 3 个；
不能把别名、参数配置、cipher/plain overload 或“custom_graph”标签计作翻倍。
该计数只是覆盖目标，不要求为了凑数量引入没有必要的语言特性。

以后多项式近似报告必须同时保留三个量：原模型明文输出 `f(x)`、明确批准的
近似模型明文输出 `p(x)`、解密输出 `y_fhe`。分别报告 `p(x)-f(x)` 的模型近似
误差和 `y_fhe-p(x)` 的 CKKS 执行误差，并附 `y_fhe-f(x)` 的总误差。当前例子
没有近似替换，不能把未来这项要求标成已完成。

## 本阶段真实实验与反例

所有结果在 `/home/lhy/poseidon-work/results/`：

| 目录 | 结果与含义 |
|---|---|
| `fx-batch-1y3z2dpr` | 规则图通过；4 组加密输入、8 输出；MAE 5.052761610155332e-9，max abs 1.2292731610408225e-8，max nonzero relative 1.4633949585451944e-7 |
| `candidate-replay-0_gia1ei` | 人工 golden 通过实际隔离 candidate 通道；agent_calls=0 |
| `candidate-replay-o__b34p6` | 故意只做两元素而非四元素归约；AST、compile、密态执行通过，numerical_comparison 失败，预期退出1 |

第三个是主动构造的**语义反例成功被检出**，不是环境失败，也没有为其放宽容差。
重现方法是将 `--golden-file` 改为同目录 `wrong_reduction.py`，加 `--max-repairs 0`。

`test_custom_graph_evidence.py` 通过显式环境变量读取以上真实结果，验证 backend、
执行标志、artifact opcode、冻结输入/权重/代码/artifact 的哈希及反例失败层。
未配置证据时会 skip，不将模拟对象称为实际密态执行。

最终完整回归：204 tests，190 通过、14 Torch-only
测试跳过；固定 Nix/Torch 环境 27/27，通过了包括该 14 项在内的全部相关测试。
以上不是新 48-case live 重跑；只有新自定义案例运行了新的真实密态实验。

## Poseidon GPU：当前阻塞和下一步批准边界

只读复核：RTX 4060 8188 MiB，Windows 驱动 616.64；WSL PATH 无 `nvcc`，
`/usr/local` 没有 CUDA 安装目录；系统 CMake 为3.22.1。仓库 RMM 24.12.01
要求 CMake>=3.26.4，随仓库的 CUDA12.5 环境使用 GCC11。
Poseidon optional GPU target 要求 CUDAToolkit/C++17，默认 architecture75，
实际4060必须明确选择合适目标，不能把默认值当作本机验证。

安装候选：从 NVIDIA 官方 CUDA12.5.1 redistributable 中选择 nvcc12.5.82、
cudart12.5.82、CCCL12.5.39、cuobjdump12.5.39，压缩下载共54,212,408 bytes
（约51.7MiB）。这只是最小编译候选，不保证覆盖后续所有 link dependency。
拟放在 `/home/lhy/poseidon-work/deps/cuda-12.5.1`，缓存放 cache，out-of-source
构建放 build-poseidon；使用已隔离的新 CMake 或另行确认，不能混改 Dacapo
LLVM 环境。**尚未安装，需批准后执行**，拟下载上限100MiB、预留1GiB。

版本依据：本仓库 `third_party/rmm/conda/environments/all_cuda-125_arch-x86_64.yaml`
以及 [NVIDIA 官方12.5.1组件清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.1.json)。
官方支持用组件归档建立独立前缀，见 [CUDA12.5.1 Linux安装文档](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-installation-guide-linux/index.html)。
只安装 Toolkit，不安装/替换 WSL Linux display driver，遵循
[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/archive/12.5.0/wsl-user-guide/index.html)。

即使 Toolkit 安装成功，仍需解决 HEVM ModswitchC、GPU multiply/relin、
低位宽 Q/P 与 logical scale/level 映射，再按不降低安全性的参数完成 encode/
encrypt/evaluate/decrypt/decode。当前 artifact 包含 ModswitchC，Poseidon adapter
仍拒绝它；不能直接说“装 CUDA 就完成 GPU”。本阶段未改这些映射或安全参数。

## 本地修改边界与后续

本次新增 model_graph、语义清单、三组测试、自定义 JSON、人工 golden/反例和
本记录；修改 model_catalog、fx_to_hecate、run_model_batch、run_candidate、
test_agent_entry、quickstart。没有修改已有 `.env` key，没有新依赖安装。
分支/HEAD 不变，所有修改保留 working tree；未 commit、push 或 PR。

建议以后获准提交时拆成“自定义图与权重/精确算术/测试入口”和
“语义清单、覆盖门禁与阶段证据”两组。下一阶段继续原完整目标，不跳过
尚未通过的多输入、rotation/key、shape、Conv/Pool、近似误差及 GPU gate。
