# pt_agent：从 PyTorch 模型生成性能感知的 FHE DSL 程序

pt_agent 读入一个 PyTorch `nn.Module`，生成 PT-DSL 程序（受限 Python 子集，只解析、不执行）。导出后端把 PT-DSL 转成可独立交付的结构化 Hecate 包，再交给 Hecate/DaCapo 编译器，得到 RuntimePlan。

```
PyTorch nn.Module ──(intake: make_fx/ATen)──► PT-Graph ──(planners: rule | agent)──► PT-DSL
     ──(export + copied kernels)──► 结构化 Hecate 包 ──(trace_model / compile_model)──► RuntimePlan
```

设计、各轮 spec 和当前状态见 `DSL/DSL_agent/docs/pt_agent/`（PLAN.md、specs/、STATUS.md）。

## 目录

| 目录 | 内容 |
|---|---|
| `intake/` | PyTorch 前端（ATen 层 make_fx 跟踪 + 核心算子映射表）、PT-Graph |
| `dsl/` | PT-DSL 的 IR、解析器、打印器、类型检查 |
| `backends/`、`kernels/` | Mock / Count / Emit / DAG 后端；mux 卷积、Chebyshev 与组合 minimax 激活、池化、全连接 |
| `lower.py`、`cost.py`、`placement.py` | lowering、代价模型（复现编译器的 depth-dp 自举放置） |
| `export/` | 结构化交付包：逐层模型、显式近似系数、接口、固定内核及独立入口；权重保存在包外 |
| `planners/` | 规则规划器、LLM Agent 循环、精调工具、冻结确认 |
| `toolchain.py`、`compiler_profiles/` | Hecate trace、`hecate-opt` 调用、随仓库提供的编译参数文件 |
| `experiments/`、`tests/` | 各轮实验脚本和测试 |

## 1. 环境

本机使用的是 ARM Linux（aarch64），通过 nix-portable 提供纯净的 Nix shell，并在其中使用固定版本的 Python 虚拟环境（torch 2.0.1 CPU、numpy）。

1. **Nix 与 Python 环境**：由 `DSL/DSL_agent/scripts/baseline/hecate_python_env.py` 创建，位于 `<poseidon-work>/platforms/<platform>/venvs/hecate-2.0.1-cpu`。
   ```bash
   cd DSL/DSL_agent/scripts
   python3 baseline/hecate_python_env.py --plan              # 查看将要安装的内容
   python3 baseline/hecate_python_env.py --install-approved  # 安装（需要网络）
   ```
2. **Hecate/DaCapo 编译器**：源码是子模块 `third_party/ckks-runtime/third_party/dacapo`（其中包含 depth-dp 自举放置），还需要应用本地补丁 `DSL/DSL_agent/patches/dacapo-local-13file.patch`。该补丁修改 13 个文件，包括 Python 前端 `python/hecate/hecate/expr.py`、`tools/frontend.cpp` 和 `tools/optimizer.cpp`；它已在 DaCapo 提交 39c1fb2 和 6e75be2 上验证可以干净地应用。
   ```bash
   git submodule update --init --recursive third_party/ckks-runtime
   cd third_party/ckks-runtime/third_party/dacapo
   git apply ../../../../DSL/DSL_agent/patches/dacapo-local-13file.patch
   # 按该目录的 README / install.sh 用 CMake 构建，得到 <build>/bin/hecate-opt
   ```
   本机的构建记录在 `<poseidon-work>/platforms/aarch64-linux/results/agent-current-dacapo-build-r1712/`（plan.json、build.log）。
3. **一次性运行入口**：所有需要 torch 或编译器的命令，都通过 `pt_agent.tools.nixrun` 在 Nix shell 中执行，并且一次只能运行一个（nix-portable 使用独占锁）。

## 2. 位置配置（环境变量，均可选）

默认布局是 `<workspace>/Code/Poseidon`（本仓库）、`<workspace>/pt_agent_results`、`<workspace>/poseidon-work`。路径定义集中在 `paths.py`。

| 变量 | 含义 | 默认值 |
|---|---|---|
| `POSEIDON_WORK_ROOT` | poseidon-work 根目录（Nix、虚拟环境、数据、编译输出） | `<workspace>/poseidon-work` |
| `POSEIDON_PLATFORM` | 平台目录名 | `aarch64-linux` |
| `PT_AGENT_RESULTS_ROOT` | pt_agent 运行结果的根目录 | `<workspace>/pt_agent_results` |
| `PT_HECATE_BUILD` | 含 `bin/hecate-opt` 的编译器构建目录 | `<poseidon-work>/platforms/<platform>/results/agent-current-dacapo-build-r1712/build` |
| `PT_DACAPO_ROOT` | DaCapo 源码目录：Python Hecate 前端、组合 minimax 系数、ResNet20-SiLU 的模型定义和权重 | `<repo>/third_party/ckks-runtime/third_party/dacapo` |
| `PT_DACAPO_UPSTREAM` | 未修改的上游 DaCapo 克隆（corelab-src/dacapo），只有 R9a 统计专家 DSL 代码行数时使用 | `<repo>/third_party/dacapo` |
| `PT_RESULTS` | 单次外层运行的输出目录（只由外层读取） | 无 |

`nixrun` 会把除 `PT_RESULTS` 以外的变量转发进 Nix shell。

## 3. 数据与权重（不在仓库中）

| 内容 | 默认位置（相对 `<poseidon-work>/platforms/<platform>/results/`） | 格式 |
|---|---|---|
| CIFAR-10 | `agent-cifar-offline-records-r1804/cifar10-records/{train,test}-NNNNN.records` | 官方二进制格式：每条 1 字节标签 + 3072 字节 CHW 像素；每个文件 1000 条，NNNNN 为起始下标 |
| 校准与加密测试的选图列表 | `agent-resnet20-original-accuracy-r1805/frozen-selections.json` | `calibration` 和 `encrypted_test` 两个列表，每项为 index、shard、row、label |
| ResNet20 权重 | `agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th` | PyTorch state_dict |
| ResNet20-SiLU 权重与组合 minimax 系数 | `<PT_DACAPO_ROOT>/examples/data/resnet20.silu.model`、`<PT_DACAPO_ROOT>/python/poly/poly/data/sgn151527.txt`（随子模块提供） | DaCapo 提供 |
| cnn_plain / mlp 权重 | `<PT_AGENT_RESULTS_ROOT>/models-v2/` | 用 `python3 -m pt_agent.tools.nixrun -- -m pt_agent.models.train_small --out-dir <dir> --model cnn_plain`（mlp 同理）重新训练 |

**数据划分**：测试集第 9000–9999 张为循环内的留出验证集；第 8000–8999 张为冻结时的确认集；第 0–7999 张只用于最终报告。训练集只用于校准和训练。

**验证引擎**：按 R8f2 §8–§9，`tuner.tune` 默认使用 `engine="tensor"` 进行 float64 张量精调，开启结果、结构化前缀及小函数三层进程内缓存；可用 `cache=False` 关闭。面向 LLM 的 `agent_tools.evaluate`、`validate` 默认使用 Mock，冻结确认、独立精确检查和最终报告也使用槽级 Mock。Tensor精调对最终程序额外执行一次完整Mock复核，核对feasible/agree后使用Mock统计形成对外反馈；搜索决策仍来自Tensor过程。冻结tag反馈全文一致性验收及当前状态见 STATUS.md。

## 4. 常用命令

在 `DSL/DSL_agent/scripts` 下执行：

```bash
# 测试；缺少历史结果、数据或编译器时，依赖它们的测试会跳过，并说明缺了什么
python3 -m pt_agent.tools.nixrun --seconds 3000 -- -m unittest discover -s pt_agent/tests -p 'test_*.py'

# 规则规划器：模型 → PT-DSL → 编译（不调用 LLM）
python3 -m pt_agent.tools.nixrun --seconds 3000 -- -m pt_agent.run --inside --out <dir> --model resnet20 \
    --models-dir <PT_AGENT_RESULTS_ROOT>/models-v2 --planner rule --deg 127 --margin 1.5 \
    --stages plan,check,mock,count,export,compile

# 已有 PT-DSL → 可迁移的包和外部 NPZ
python3 -m pt_agent.tools.nixrun -- -m pt_agent.export --program <program.ptdsl> --model resnet20 \
    --models-dir <PT_AGENT_RESULTS_ROOT>/models-v2 --out <new-dir>

# 在交付包中检查接口；trace/compile 要求新的输出目录
python3 <new-dir>/ResNet20_PTAgent_DSL/trace_model.py --plan

# LLM Agent（需要 DeepSeek API 密钥，放在仓库根目录的 .env 中，由 baseline/agent_credentials.py 读取）
python3 -m pt_agent.experiments.r8b_dev --round <name> --tuner on
```

交付包包含 program.ptdsl、model.py、显式近似配置、权重 schema、接口和相对路径入口，无 pt_agent 运行依赖。
权重 NPZ 位于包的同级 weights/ 目录；调用方提供 hecate 前端。包内 README 给出完整 trace、编译和自检命令。
编译结果写在 `<poseidon-work>/platforms/<platform>/results/pt-agent-*` 下，包括 RuntimePlan 和编译日志。
旧 `emit` 阶段保留供对照使用。`compile` 阶段要求 `export`。

## 5. 已知限制

- **本机 CPU CKKS 执行**（`toolchain.execute_cpu`，只用于不含自举的小程序）仍走旧的 baseline SDK 选择文件，以及不在仓库历史中的 `normal_cpu_overlay_sandbox`。没有这些时，相关测试会跳过。整网真实 CKKS 需要能执行自举的运行时（GPU）。
- 非可信的候选沙箱（`trace`/`compile` 的 `trusted=False`）同样依赖旧 SDK；pt_agent 的流程使用的是可信运行器（`trusted=True`）。
- 只支持单密文布局（中间结果需要放进 16384 个槽位），kernel 目前覆盖 CIFAR 尺度的 CNN/MLP。
