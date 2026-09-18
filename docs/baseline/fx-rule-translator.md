# 受限 PyTorch/FX → Hecate：可批量运行的规则对照

## What：交付的是什么

这是一条不调用 LLM 的 deterministic translator 基线：

`数据型模型描述 → 真实 PyTorch Module → FX 图检查/规则 lowering → Hecate
函数片段 → AST/type/layout gate → Dacapo → HEVM/CST → SEAL CPU → 解密差分`

转换器根据 FX operator、参数和 shape 工作，不读取测试 family 名称或按
case ID 挑选手写 golden。独立测试还覆盖了不属于 catalog 的自定义 PyTorch
计算图和根模块 nn.Linear。它不是 Agent，也不能算 Agent contribution。

第一版的 CLI 输入是严格校验的 JSON 描述文件（例如
`scripts/baseline/cases/linear-example.json`），指定受限 catalog 的 family 与
configuration。受信 `model_catalog.py` 构造 CPU float64/eval PyTorch 模型。
模型随机初始化被隔离，模型权重由固定版本的 NumPy、固定种子和固定取值
规则覆盖；每个结果目录另存实际权重的 `weights.npz`，禁止 pickle 加载。

Python API `fx_to_hecate.translate(model, input_shape)` 接受可信 `nn.Module`。
CLI **尚不导入任意 model.py，也不支持任意自然语言/ONNX/FX JSON 输入**。
FX tracing 会执行模型 Python；检查 FX 图不是检测所有外部副作用的沙箱。
这里只对受信 factory/测试代码使用，不能将该接口暴露给未知 Python 上传。

## Why：与 Agent 任务的关系

这建立同输入、同 Hecate 目标、同 SEAL backend 的规则对照，并验证生成后的
程序能接受真实编译/运行反馈。若未来 Agent 只复制这些静态算子映射，它未必
优于 compiler frontend。后续需要实测 unsupported-op 改写、反馈修复、布局
选择等收益，不能把规则转换器的正确率记到 Agent 名下。

## How：模块和边界

| 文件（相对源码根） | 消费 → 产生 |
|---|---|
| `scripts/baseline/model_catalog.py` | 受限 JSON descriptor → 真实 PyTorch model、固定输入 |
| `scripts/baseline/fx_to_hecate.py` | 可信 Module / input shape → 已检查 FX、Hecate 源码、常量和 layout manifest |
| `scripts/baseline/hecate_contract.py` | 函数片段/公开常量 → 静态结构/类型检查；不证明数值正确 |
| `scripts/baseline/trace_translated.py` | 仅受信规则输出 → 实际 Hecate frontend MLIR/CST |
| `scripts/baseline/run_model_batch.py` | 单例描述/固定 catalog → 分阶段诊断、真实编译执行结果、总体指标 |
| `scripts/baseline/seal_cpu_golden.py::execute_artifact` | 已校验 HEVM/CST、输入与输出 slot selector → 原有 SEAL runtime 执行/解密 |
| `scripts/baseline/test_fx_to_hecate.py` | 规则单测、独立明文 slot interpreter；**不是 FHE 执行证据** |
| `scripts/baseline/test_batch_reporting.py` | 分母/失败分类测试、实际批量产物和数值证据回归 |

实际 Hecate tracing 位于独立有超时/资源限制的进程。当前 tracer 只用于
我们自己的确定性转换器输出；`generator` 字段不是身份验证，AST 检查、空
builtins 和 Nix 环境也不是 OS 能力隔离。未来接 Agent 前需补充受约束的
tracing 执行边界，不可将任意文件改个 generator 标签就当作可信源码执行。

### 输入支持范围

- 一个加密输入，公开、有限、固定的 CPU float64 模型参数；所有 module
  必须已经处于 eval，检查器不通过自动 `.double()` 或 `.eval()` 改变输入。
- 原始四元素向量 `[4]`；以及显式 `torch.flatten(x)` 把 `[2,2]` / `[1,4]`
  按 C/row-major 顺序展平成四元素。后两者不是已支持一般 batch 推理。
- `operator.add/mul`、`torch.add/mul/square`、固定 power 2/4、精确
  `nn.Linear` / `torch.nn.functional.linear`、上述 full flatten。
- Linear 输出宽度 1–4；后续 Linear 可消费这种按神经元分 ciphertext 的向量。
- 只接受明确验证的 broadcast 规则；其他 module、kwargs、in-place、ReLU、
  随机算子、数据分支、任意 reshape/其他方法明确拒绝，不进行静默近似。

### 两种内部布局

`packed`：四个输入按周期 4 重复在一个 ciphertext 中。逐元素计算仍用
一个 ciphertext。第一次 Linear 每行执行公开权重乘法、rotate 1/2 和加法，
得到一个在全部 slot 上重复的点积。

`scalars`：每个神经元占一个 broadcast ciphertext；平方逐 ciphertext 做，
后续 Linear 在这些 ciphertext 间做 scalar-weight multiply/add。无需隐式
重排或解密。两种布局直接相加目前拒绝，不能猜测它们对齐。

转换输出 manifest 明确记录 input/output shape、常量来源、ciphertext
数量和 `[result_index, slot_index]` selector。final result register 从 HEVM
metadata 获取，不把返回顺序误当成寄存器号。

### 48 例的构成及留出边界

8 family × 6 configuration：仿射、显式多项式、Linear、两层 polynomial MLP、
三层 polynomial MLP、fan-out 合并、residual、flatten→Linear。
宽度配置覆盖 1–4；不同配置有不同固定权重。每例使用零、固定有符号、seed42
随机、范围边界四组输入。这个集合仍很小，并不覆盖一般 shape/batch/Conv。

fan-out 和 residual 的 12 例标注 `holdout_for_agent=true`。它们可以运行规则
对照，但不得放入未来 Agent prompt 示例。当前尚无 Agent prompt/training，
所以不能把这个标签本身称作“Agent unseen 泛化已通过”。

## Evidence：复现和报告

Confirmed（2026-09-05）：完整批量结果
`/home/lhy/poseidon-work/results/fx-batch-yw9qpdtx` 为 **48/48 通过**。
FX 转换/静态检查、Hecate tracing、编译、真实 SEAL 执行、数值正确率均为
48/48。每例四组输入，共 192 次密态输入执行、640 个输出值逐项比较。
全体最大绝对误差为 **3.972002460272961e-8**，未放宽容差。

| family | 通过/总数 | 最大绝对误差 |
|---|---:|---:|
| affine | 6/6 | 9.0099828e-9 |
| polynomial | 6/6 | 3.1456125e-8 |
| Linear | 6/6 | 9.3443384e-9 |
| 两层 MLP | 6/6 | 2.4179624e-8 |
| 三层 MLP | 6/6 | 2.8914826e-8 |
| fan-out | 6/6 | 3.9720025e-8 |
| residual | 6/6 | 1.9536176e-8 |
| flatten→Linear | 6/6 | 1.2746947e-8 |

单独 `--case` 文件入口也实际通过：
`/home/lhy/poseidon-work/results/fx-batch-9fo2xy9m`。较早的五例 smoke 结果
`fx-batch-2ystv3sq` 保留为历史记录；之后完善了 reference hash 冻结保护，
完整 48 例和单文件例使用完善后的驱动。没有按失败结果筛掉任何计划案例。

回归验证：宿主 baseline suite 共发现 85 项，76 项通过，9 项因宿主未安装
Torch 而跳过；这 9 项另通过 `--unit-tests` 在固定 CPU Torch 环境中全部
执行通过。不是把 skip 计为通过。原手写 `--suite base` 的 add、mul_plain、
Linear 三例也重新完成真实 trace/compile/encrypt/execute/decrypt/compare，
结果为 `/home/lhy/poseidon-work/results/seal-cpu-golden-98f3gsf8`，确认共用
执行器的输出 selector 重构未破坏原入口。

Inference：对于这个受限静态算子/布局集合，规则转换器已经可以完成自动映射；
尚没有证据表明加入 Agent 会提高成功率。Unconfirmed：任意 PyTorch 程序、
Python 层隐式随机性/副作用检测、其他尺寸/batch、真实 Agent 生成/修复、
跨密钥统计稳定性，以及 Poseidon GPU 正确性和性能。

依赖、profile、waterline 和容差沿用已经验证的版本：SEAL 4.0.0、LLVM/MLIR
18.1.2、Torch 2.0.1+cpu、NumPy 1.25.2、N=32768、14×60-bit 模数、tc128、
waterline=40；`abs(error) <= 1e-5 + 1e-4*abs(reference)`。不安装新依赖，
不改安全参数，不执行 bootstrap，不使用 GPU。

```powershell
# 单个数据型描述文件
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_model_batch.py --case scripts/baseline/cases/linear-example.json
# 五个代表性案例
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_model_batch.py --smoke
# 全部 48 例
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_model_batch.py
# 必须在已安装的隔离 CPU Torch 环境内执行的九项单元测试
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_model_batch.py --unit-tests
```

结果写入新的 `/home/lhy/poseidon-work/results/fx-batch-*`，不覆盖旧记录。
源码、测试和需要提交的修改全部在 `D:\Code Space\Poseidon`。大构建仍最多
`-j2`；这里只构建很小的已有 key/metadata helper。外层有 900 秒硬超时，
各 trace/compile/execute 子进程也有独立硬超时。

每例保留：`model.json`、`weights.npz`、原逻辑输入/packed 输入/独立 reference
的 `arrays.npz`、`generated.py`、`translation.json`、FX/Earth/CKKS/HEVM/CST、
命令与日志、`decrypted.npy`、`execution.json`、`case-report.json`。
`report.json` 汇总阶段、分子/分母和失败层。输入、权重、reference 在 trace
之前记录 hash，后续不能重新覆盖这个基准 hash；即使发生早期失败也要查验。
私钥仅在 mode-700 的 native 结果目录内，**不要分享 private-keys**。

报告层：model_description、model_reference、fx_translation、dsl_trace、
compiler、artifact_gate、seal_runtime、numerical_comparison、integrity。
这些表示失败被观测到的关卡；仅凭退出码不宣称已经确定环境/编译器 bug 的
根因。输入拒绝、共享基础设施失败和 pipeline 失败分别统计。
共享环境失败时保留全部计划案例，不从最终分母中删掉未执行项。运行中
status=running 的报告只表示进度，不能当作最终 48 例成功；中断也不算成功。
Agent call 数量始终为零，未实现反馈修复轮次，因此不报告 Agent 首次/修复成功率。

宿主回归命令（在 WSL 源码根运行，所有结果路径均为已有只读证据）：

```bash
timeout -k 3s 5m env PYTHONDONTWRITEBYTECODE=1 \
  POSEIDON_NATIVE_COMPILER_RESULTS=/home/lhy/poseidon-work/results/native-compiler-k73nx6wz \
  POSEIDON_PYTHON_COMPILER_RESULTS=/home/lhy/poseidon-work/results/python-compiler-r2ekrtjt \
  POSEIDON_SEAL_GOLDEN_RESULTS=/home/lhy/poseidon-work/results/seal-cpu-golden-98f3gsf8 \
  POSEIDON_SEAL_SEMANTICS_RESULTS=/home/lhy/poseidon-work/results/seal-cpu-golden-l7iu78ak \
  POSEIDON_FX_BATCH_RESULTS=/home/lhy/poseidon-work/results/fx-batch-yw9qpdtx \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -v
```

与显式随机 FX operator 不同，Python 层随机值可能在 tracing 时被常量折叠；
本检查器不能据 FX 图证明所有 Python 源码无随机性/副作用。catalog 的
`forward` 由受信代码固定且无随机操作；任意自定义模型仍必须先经过人工/更强
隔离检查，不能把目前的 API 用于未经审核的代码。

本轮本地变更：新增 model factory、FX translator、单例 JSON、tracing wrapper、
batch driver、两份测试文件和本文档；仅将已有 SEAL worker 的输出选择逻辑
抽成可复用接口，旧 golden 仍走同一个原 SEAL runtime。所有改动仍留在
working tree；不 commit/push/PR。以后经批准可按“转换器与案例”和“批量执行、
诊断与文档”拆分提交，现有环境/Poseidon 修改不混入这两组。

## 下一门禁

连接可配置模型接口与结构化生成/反馈包之前，需要：确认模型服务、模型版本、
凭据存放和费用上限；隔离 Agent tracing；保护 reference/安全参数/权重的
所有权；最多三轮修复；以本规则转换器作同输入同后端对照。不要将未实现的
Agent 配置或 pipeline mock 测试宣称为自动合成成功。
