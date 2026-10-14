# Poseidon Agent

将模型转换为 Hecate DSL，完成编译与同态数值验证，输出可复用程序包。

执行链：**Hecate → Dacapo-modified → RuntimePlan → CKKS Runtime → Poseidon CPU**。
Agent 用于模型转换和部署准备；生成后的程序包可直接重复执行。

## 模型接入

| 输入方式 | 用途 |
|---|---|
| JSON 计算图 | 直接声明输入、公开权重、算子、连接和命名输出 |
| 可信本地 PyTorch | 捕获 `nn.Module.eval()`，支持工厂参数、子模块和独立 NPY 权重 |
| 受限 Python 源码 | 静态解析已登记的 Python／PyTorch 写法 |

模型采用固定 shape、公开权重和浮点计算数据。输入可组织为 Tensor、tuple/list、字典及关键字参数；输出可保留对应的命名和结构。

PyTorch 清单默认使用 v18，`--source-format latest` 使用 v34。清单保存实际版本，旧显式版本继续兼容。
多文件模型通过 `--source-root` 指定根目录，`--helper` 列出依赖；源码包最多 8 个文件、64 KiB。

[输入格式](../docs/baseline/supported-inputs-models-configurations.md) · [PyTorch 与算子配置](../docs/baseline/operator-decomposition-and-python.md)

## 算子范围

| 类别 | 已接入的计算 |
|---|---|
| 线性与空间 | Linear、dot/mm/bmm/matmul、Conv1d/2d、ConvTranspose1d/2d、resize、平均池化 |
| 图结构 | 残差、分叉、共享中间结果、多输入融合、命名多输出 |
| 算术与归约 | 加减乘、公开除法、平方与静态幂、sum/mean、variance |
| 张量操作 | reshape/view、flatten、permute/transpose、广播、cat/stack、静态 slice/split/chunk/unbind、公开索引、对角线 |
| 重复与重排 | repeat/tile、repeat_interleave、flip、roll |
| 归一化 | 推理 BatchNorm、LayerNorm、RMSNorm、GroupNorm、InstanceNorm |
| Attention | 静态 Q/K/V、投影、公开 mask、因果组合及已登记的 Attention 接口 |
| 公开张量 | bool/int32/int64/float32/float64、类型转换、常量构造、算术、where/masked_fill、公开与加密张量混合拼接 |

非多项式函数采用显式近似配置，包含方法、区间和参数。ReLU、GELU、SiLU、Softmax、max/min 等按对应配置进入算子分解。
实际执行范围同时受 shape、布局、乘法深度和资源检查约束。算子的轴、广播和参数范围见[详细说明](../docs/baseline/operator-decomposition-and-python.md)。

当前 CPU 回归包含 Linear、平方 MLP、归一化、Attention decode、分支与多输出等小模型。
最新公开／加密拼接的真实 Agent 测试为 18/18；多文件 CLI 的三种来源均通过编译、数值与程序包复用。
命名输出共享的真实 Agent 测试为 9/9 首轮通过。[输出共享](../docs/baseline/agent-output-alias-acceptance-r1486.md)
[生成与数值结果](../docs/baseline/agent-public-join-acceptance-r1480.md) · [多文件入口](../docs/baseline/agent-source-cli-acceptance-r1482.md)

## 容量配置

| 配置 | 输入／输出总元素，各自 | 单个中间张量 | 原图节点 | rank |
|---|---:|---:|---:|---|
| `legacy-v1` | 256 | 256 | 64 | 1–4 |
| `medium-v1` | 1,024 | 4,096 | 256 | 1–4 |
| `resnet20-v1` | 4,096 | 65,536 | 1,024 | 1–4 |
| `general-tensors-v1` | 4,096 | 65,536 | 1,024 | 1–32 |

逻辑输入和命名输出各最多 4 个；标量输出通过对应结构表示。rank 与轴按各算子定义检查。
统一图 v2 的多个命名输出可引用同一结果。选择 `--output-binding exact-values-v1 --unified-guidance explicit-v15` 后，共享该结果的密文块，保留输出名称、shape 和返回结构。
medium 配置通常映射到最多 4 个输入／输出密文块，较大配置最多 16 个；尾块补零也计入块数。

使用 `scripts/plan_model.py` 选择布局，`scripts/analyze_resources.py` 或 `scripts/analyze_program.py` 检查资源。
较大输入可显式选择周期 512/1024/2048/4096；配置与实际产物共同决定可执行范围。
[布局与资源接口](../docs/baseline/application-agent-component-v1.md#布局准备)

## 环境配置

运行平台为 Linux x86_64 或 ARM64。现有验收环境为 ARM Ubuntu 22.04，后端采用 Poseidon CPU。

| 依赖 | 版本 |
|---|---|
| LLVM / MLIR | 18.1.2 |
| GCC / GMP / GSL | 13.2.0 / 6.3.0 / 3.1.0 |
| Python / NumPy | 3.10.14 / 1.25.2 |
| PyTorch | x86：2.0.1+cpu；ARM64：2.0.1 CPU |

依赖由 Nix 和锁文件管理。[依赖配置](../docs/baseline/agent-setup.md) · [Poseidon SDK 构建](../docs/baseline/poseidon-runtime-migration.md)

```bash
git clone --branch lhy-agent-dsl https://github.com/mingerfan/poseidon.git
cd poseidon

export POSEIDON_WORK_ROOT="$HOME/poseidon-work"
export POSEIDON_PLATFORM=aarch64-linux
# x86 平台使用 x86_64-linux
```

## 环境验证

```bash
python3 -B scripts/agent.py --backend local doctor

python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
```

`doctor` 检查环境；`--self-test` 使用内置候选完成编译、加密执行和数值比较。

## PyTorch 模型导入

以下命令使用 `my_model/model.py` 中的 `make()` 工厂，输入 `x` 的 shape 为 `[1,4]`，输出命名为 `logits`：

```bash
python3 -B scripts/prepare_pytorch_source.py my_model/model.py \
  --entry model.make --source-format latest \
  --input x:1,4 --output logits \
  --dtype-policy convert-float64-v1 \
  --manifest my_model/source.json

python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest my_model/source.json --trust-local-code \
  --resource-profile general-tensors-v1 --write \
  --output imported-model
```

嵌套包增加 `--source-root my_model` 和各个 `--helper`；`--entry` 使用模块路径，如 `models.model.make`。
`--weight linear.weight=weight.npy` 绑定公开权重。`--call call.json` 描述结构化参数及关键字调用。
float32→float64 转换由 `--dtype-policy convert-float64-v1` 显式选择。

## DSL 生成

在项目根目录的 `.env` 中配置：

```dotenv
DEEPSEEK_API_KEY=<api-key>
```

先准备请求：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case imported-model/model.json --prepare \
  --unified-guidance explicit-v14 \
  --public-weights descriptors-math-roles-v2 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
```

生成并验证：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case imported-model/model.json --live \
  --provider deepseek --model deepseek-flash \
  --reasoning-effort high --stream \
  --api-timeout 1200 --max-tokens 384000 \
  --max-repairs 3 --provider-retries 3 \
  --unified-guidance explicit-v14 \
  --public-weights descriptors-math-roles-v2 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
```

JSON 计算图可直接作为 `--case`。旧 schema 1–5 沿用原入口。
生成和修复使用同一个模型，在给定预算内返回成功或明确失败。

## 验证与程序包

结果保存在平台工作目录的 `results/`，包含 DSL、编译产物、日志和数值报告。
数值验收使用真实 Poseidon CPU 执行，比较四组输入的逐项输出：

```text
abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)
```

```bash
python3 -B scripts/dsl_bundle.py export \
  --evidence /absolute/path/to/result \
  --output /absolute/path/to/program \
  --identity-policy staged-v1

python3 -B scripts/dsl_bundle.py replay \
  --bundle /absolute/path/to/program --execute

python3 -B scripts/dsl_bundle.py execute \
  --bundle /absolute/path/to/program --inputs inputs.json
```

程序包包含候选源码、模型、公开权重、RuntimePlan、OperatorSpec、常量 bundle 和依赖身份。
重载后可直接加密、执行和解密新输入，无须重新生成 DSL；每次接受 1–16 组样本。

编译验收、CKKS 数值验收和原模型应用精度分别记录。Bootstrap 接线、CUDA 实机和完整 ResNet20 属于后续独立里程碑。

## 应用接口

`submit_source → status → result` 提供模型提交、状态查询、取消、预算和程序包复用。
宿主通过固定 `WorkerContext` 中的独立 worker 执行 `run`，成功后调用 `execute(task_id, samples)`。

[组件接口与示例](../docs/baseline/application-agent-component-v1.md)

## 开发与测试

- [Benchmark](../docs/baseline/semantic-benchmark-v1.md)：1,200 个去重小模型，自由生成与定向构造分别计分。
- [示例模型](baseline/cases/)。
- [文档目录](../docs/README.md)。

```bash
python3 -B scripts/agent.py --help
python3 -B scripts/agent.py --backend local candidate -- --help
```
