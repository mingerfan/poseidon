# Agent 使用说明

## 启动方式

以下命令均在 **Ubuntu x86_64 终端**、项目根目录下运行。需要 Python 3.10 或更新版本、项目固定版本的编译依赖及 Dacapo 子模块。本文只说明 Ubuntu 本地运行。

### 配置 API key

在项目根目录创建 `.env`，格式参考根目录的 `.env.example`。已有 `.env` 时只编辑需要的字段，不覆盖原文件。

```dotenv
DEEPSEEK_API_KEY=替换为你的DeepSeek密钥
```

当前只支持 DeepSeek，密钥只需配置在这一个文件中。不要提交 `.env`，也不要把路径配置写入其中。

### 检查环境

```bash
python3 scripts/agent.py --backend local doctor
```

`doctor` 只检查平台、依赖路径和工具是否存在，不表示编译或密态执行已通过，也不会自动安装依赖。

依赖准备好后，可先运行不调用 API 的单例自测：

```bash
python3 scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
```

`--self-test` 使用脚本化候选验证检查链路，不代表真实 Agent 生成测试。

### 单个模型

```bash
python3 scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json \
  --live --provider deepseek --model deepseek-flash \
  --reasoning-effort high --api-timeout 1200 \
  --max-repairs 3 --provider-retries 3 --stream
```

`--live` 会产生付费 API 调用；`--deepseek` 是同一开关的别名。改用自己的模型时，将 `--case` 替换为模型 JSON 路径。

### 批量运行

使用下文的 `manifest.json`，先查看批次，不调用 API：

```bash
python3 scripts/agent.py --backend local batch -- \
  --case-manifest manifest.json --plan
```

确认后启动付费批次：

```bash
python3 scripts/agent.py --backend local batch -- \
  --case-manifest manifest.json \
  --live --provider deepseek --model deepseek-flash \
  --jobs 10 --reasoning-effort high --api-timeout 1200 \
  --provider-retries 3 --stream
```

未传 `--case-manifest` 时使用内置模型批次。

### 路径与常用选项

- `candidate` / `batch` 前面是启动器参数，`--` 后面是运行器参数；后面的相对文件路径均按 **Ubuntu 项目根目录** 解析。
- 工作目录默认是当前用户的 `~/poseidon-work`。可在启动器参数中添加 `--work-root '/data/poseidon-work'`，或设置 `POSEIDON_WORK_ROOT`；该设置不会自动搬移或安装依赖。
- `--timeout` 放在 `candidate` / `batch` 前，控制整个任务的期限，默认 43200 秒；`--api-timeout` 放在后，控制单次 API 请求期限。
- `--dry-run` 放在 `candidate` / `batch` 前，仅显示启动命令，不启动任务。
- `--loopback-proxy-port 6478` 放在 `--` 后，仅在代理确实监听于当前 Ubuntu 的本地端口时使用。
- `candidate` 模式下，将 `--live` 替换为 `--prepare` 可只准备模型请求，不调用 API。
- 输入 schema 5、需要 native helper / 数组 / 公开循环构造时，可显式添加 `--native-array-mutation --compiler-configuration seal-cpu-eva-w45-v1`；其他构造模式需按入口帮助选择，不要任意混用。
- 完整参数：`python3 scripts/agent.py --help`、`python3 scripts/agent.py --backend local candidate -- --help`、`python3 scripts/agent.py --backend local batch -- --help`。

## 接受的输入

完整的逐 schema 算子矩阵、模型族、六种预设配置和专项目录见
[支持的输入、模型与配置清单](../docs/baseline/supported-inputs-models-configurations.md)。

`--case` 接收 **JSON 模型描述文件**，不是自然语言、任意 `model.py`、ONNX、`.pt` 或 pickle 文件。模型采用静态 shape、公开固定权重、加密输入；算子和尺寸须在下述范围内。

### 单个模型的格式

| `schema` | 输入内容 | 输入范围 |
|---|---|---|
| 1 | 内置模型的 `family` 和 `configuration` | 8 类模型，每类配置编号 0–5 |
| 2 | 自定义单输入静态计算图 | 一个输入，4 个逻辑元素 |
| 3 | 自定义多输入静态计算图 | 2–4 个独立加密输入，每个输入 4 个逻辑元素 |
| 4 | 自定义分块输入计算图 | 一个 rank 1–4 的逻辑张量，总元素数 5–16 |
| 5 | 自定义可变周期 packing 计算图 | 一个 rank 1–4 的逻辑张量，总元素数 1–256 |

Schema 1 的 `family` 可取 `affine`、`polynomial`、`linear`、`mlp2`、`mlp3`、`fanout`、`residual`、`flatten_linear`。`id` 必须为 `family-configuration`，例如：

```json
{
  "schema": 1,
  "id": "linear-1",
  "family": "linear",
  "configuration": 1
}
```

### 自定义模型示例

将以下内容保存为执行端项目目录下的 `model.json`，然后使用 `--case model.json`：

```json
{
  "schema": 5,
  "id": "matrix-affine",
  "input_shape": [2, 3],
  "constants": {
    "gain": [0.5],
    "bias": [0.375]
  },
  "nodes": [
    {"id": "scaled", "op": "multiply", "inputs": ["x", "gain"]},
    {"id": "result", "op": "add", "inputs": ["scaled", "bias"]}
  ],
  "output": "result"
}
```

字段含义：

- `id`：模型标识；批量输入中不能重复。
- `input_shape`：单个逻辑输入的固定 shape，元素数为各维度的乘积。
- `constants`：公开的有限实数标量或矩形数组，包括权重、偏置等。
- `nodes`：按依赖顺序排列的算子列表；`inputs` 引用输入名、常量名或先前节点名。单输入图的输入名为 `x`。
- `output`：最终输出引用。
- Schema 3 用 `inputs: [{"name": "left", "shape": [4]}, {"name": "right", "shape": [4]}]` 替代 `input_shape`，节点按声明的名字引用不同输入。

Schema 5 接受的算子名为：`add`、`subtract`、`multiply`、`negate`、`square`、`power`、`linear`、`flatten`、`reshape`、`permute`、`transpose`、`batch_norm`、`concat`、`conv1d`、`conv2d`、`avg_pool1d`、`avg_pool2d`。其他 schema 的允许集合和布局限制不同，不能直接套用整个列表。

算子附加字段按算子确定，例如 `linear` 使用 `weight`、`bias` 引用公开常量，`power` 使用 `exponent`，`reshape` 使用 `shape`，`permute` 使用 `dims`。可参考同目录下的 [模型案例](baseline/cases/)。

主要输入限制：

- `power` 只接受指数 2 或 4；不自动将 ReLU、SiLU、MaxPool 等替换成多项式。
- Schema 5 最多 64 个图节点、32 个公开常量条目；常量和生成程序另有大小限制。
- Schema 5 的 packed 逐元素输出最多 256 个逻辑元素；Linear、Conv、concat 等 scalar-neuron 布局最多 16 个输出密文，不能据此假设任意 batch 或输出尺寸都被接受。
- shape、broadcast、通道、卷积窗口及分支合并的布局必须满足对应算子的检查；不接受动态 shape 或依赖加密数据的分支。
- 模型 JSON 不接收任意测试输入数组；现有运行器为模型构造固定的零输入、有符号输入、固定种子随机输入和边界输入。

### 批量输入格式

将以下内容保存为 `manifest.json`，交给 `--case-manifest`：

```json
{
  "schema": 1,
  "cases": [
    {
      "schema": 5,
      "id": "square-four",
      "input_shape": [4],
      "constants": {},
      "nodes": [
        {"id": "squared", "op": "square", "inputs": ["x"]}
      ],
      "output": "squared"
    }
  ]
}
```

批次外层 `schema: 1` 是清单格式版本，不是模型版本。`cases` 中须放入 1–96 个完整的 schema 2/3/4/5 自定义模型对象，不接受文件路径列表或 schema 1 的内置模型引用；每个模型 `id` 唯一，清单文件不超过 1 MiB。
