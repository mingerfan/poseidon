# Poseidon Agent

将小型模型转换为 Hecate DSL，并通过编译和同态计算检查结果。生成的程序包可以保存、重放，也可以交给宿主应用使用。

当前执行后端是 **SEAL CPU**，支持 Ubuntu x86_64 和 ARM64。Agent 用于模型转换与部署准备。

## 功能范围

| 模型或计算 | 支持范围 |
|---|---|
| 全连接网络 | Linear、展平后接 Linear、使用平方或多项式激活的小型 MLP |
| 卷积小网络 | Conv1d/Conv2d、平均池化、固定统计量 BatchNorm 及其组合 |
| 多分支网络 | 残差连接、分叉、共享中间结果、多输入融合、多输出 |
| 张量运算 | 加减乘、归约、拼接、切片、形状变换和旋转 |
| 非线性计算 | 显式多项式；SiLU 等近似需要给定系数或选择对应接口 |

推荐输入是 **JSON 计算图**。输入形状固定，权重公开，待计算的数据加密。图内最多 4 个输入、4 个输出、64 个节点；输入和输出分别最多 256 个元素。实际可运行的深度还受编译配置限制。

Python 输入通过静态解析转换为 JSON 计算图，支持的函数和算子见下方输入说明。

[模型格式与算子](../docs/baseline/supported-inputs-models-configurations.md) · [Python 与高层算子输入](../docs/baseline/operator-decomposition-and-python.md)

## 环境要求

运行环境为 Ubuntu 22.04。以下命令在仓库根目录执行。

系统依赖：Python 3.10 以上、Git、curl、bash、GNU timeout、flock、sha256sum 和 bubblewrap。

| 依赖 | 版本 |
|---|---|
| LLVM / MLIR | 18.1.2 |
| GCC / SEAL / GSL | 13.2.0 / 4.0.0 / 3.1.0 |
| Python / NumPy | 3.10.14 / 1.25.2 |
| PyTorch | x86：2.0.1+cpu；ARM64：2.0.1 CPU |

C++ 和 Python 依赖由 Nix 和锁文件管理。[完整依赖说明](../docs/baseline/agent-setup.md)

## 安装与构建

获取源码：

```bash
git clone --branch lhy-agent-dsl https://github.com/mingerfan/poseidon.git
cd poseidon
```

查看本机架构对应的构建计划：

```bash
python3 -B scripts/setup_agent.py --platform auto --plan
```

下载依赖、初始化子模块并构建：

```bash
python3 -B scripts/setup_agent.py --platform auto \
  --work-root "$HOME/poseidon-work" \
  --apply --init-submodule \
  --download-budget-mib 3072 --disk-budget-gib 40 \
  --self-test
```

示例设置下载预算为 3 GiB、磁盘预算为 40 GiB。构建并发为 2，链接并发为 1；缓存和日志保存在工作目录。

运行时选择平台：

```bash
export POSEIDON_WORK_ROOT="$HOME/poseidon-work"
export POSEIDON_PLATFORM=aarch64-linux
# x86 Ubuntu 将上一行改为：export POSEIDON_PLATFORM=x86_64-linux
```

`--platform auto` 仅用于安装器。运行器采用上面的环境变量，或显式的 `--platform` 参数。

## 环境验证

```bash
python3 -B scripts/agent.py --backend local doctor

python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
```

`doctor` 检查环境；`--self-test` 使用内置候选完成本地编译、加密计算和结果比较。

多输入、多输出示例：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/unified-two-input-two-output.json \
  --self-test --compiler-configuration seal-cpu-eva-w45-v1
```

## DSL 生成

在仓库根目录的 `.env` 中配置 DeepSeek API 密钥：

```dotenv
DEEPSEEK_API_KEY=<api-key>
```

检查模型并准备生成请求：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case model.json --prepare \
  --compiler-configuration seal-cpu-eva-w45-v1
```

通过 DeepSeek API 生成 DSL：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case model.json --live --provider deepseek --model deepseek-flash \
  --reasoning-effort high --stream \
  --api-timeout 1200 --max-tokens 384000 \
  --max-repairs 3 --provider-retries 3 \
  --compiler-configuration seal-cpu-eva-w45-v1
```

`--model` 和 `--max-tokens` 指定模型与输出上限。生成和修复使用同一个模型，最多修复 3 轮；任务结束后返回结果或失败原因。

`model.json` 使用[统一图格式](../docs/baseline/supported-inputs-models-configurations.md)。已有 schema 1–5 文件仍可从这个 CLI 运行。

## 结果与程序包

运行器会输出结果目录，里面包含生成的 DSL、编译产物、错误日志和数值报告。目录位于工作目录的 `results/`；ARM 的实际工作目录是 `$POSEIDON_WORK_ROOT/platforms/aarch64-linux`。

CLI 成功表示：程序通过检查、编译，并在四组测试输入上完成真实 SEAL 计算及数值比较。误差条件为：

```text
abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)
```

将通过验收的结果导出：

```bash
python3 -B scripts/dsl_bundle.py export \
  --evidence /absolute/path/to/result \
  --output /absolute/path/to/new-program

python3 -B scripts/dsl_bundle.py replay \
  --bundle /absolute/path/to/new-program --execute
```

程序包包含 `candidate.py`、模型、公开权重、请求和配置清单。重放使用匹配的源码和 SDK。`replay` 默认检查文件完整性，`--execute` 启用重新编译与验收。

## 应用集成

应用接口提供 `submit → status → result`，并支持取消、预算控制和相同请求的程序包复用。宿主在独立 worker 中调用 `run` 执行任务。

默认返回经过数值验收的程序包；可显式选择仅编译验收，结果会标明等级。接口用法见[应用集成](../docs/baseline/application-agent-component-v1.md)。

## 开发与测试

- [模型集与 Benchmark](../docs/baseline/semantic-benchmark-v1.md)：1200 个去重小模型、自由生成和定向构造测试。
- [开发文档目录](../docs/README.md)：接口、编译配置和验收记录。
- [示例模型](baseline/cases/)。

命令行参数：

```bash
python3 -B scripts/agent.py --help
python3 -B scripts/agent.py --backend local candidate -- --help
```
