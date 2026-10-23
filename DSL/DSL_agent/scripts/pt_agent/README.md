# pt_agent：从 PyTorch 模型生成性能感知的 FHE DSL 程序

pt_agent 读入一个 PyTorch `nn.Module`，生成 PT-DSL 程序（受限 Python 子集，只解析、不执行）。可信 lowering 把 PT-DSL 展开成 Hecate 程序，再交给**不做任何修改的** Hecate/DaCapo 编译器，得到 RuntimePlan。

```
PyTorch nn.Module ──(intake: make_fx/ATen)──► PT-Graph ──(planners: rule | agent)──► PT-DSL
     ──(lower + kernels)──► Hecate 程序 ──(toolchain: hecate-opt)──► RuntimePlan
```

设计、各轮 spec 和当前状态见 `DSL/DSL_agent/docs/pt_agent/`（PLAN.md、specs/、STATUS.md）。

## 目录

| 目录 | 内容 |
|---|---|
| `intake/` | PyTorch 前端（ATen 层 make_fx 跟踪 + 核心算子映射表）、PT-Graph |
| `dsl/` | PT-DSL 的 IR、解析器、打印器、类型检查 |
| `backends/`、`kernels/` | Mock / Count / Emit / DAG 后端；mux 卷积、Chebyshev 与组合 minimax 激活、池化、全连接 |
| `lower.py`、`cost.py`、`placement.py` | lowering、代价模型（复现编译器的 depth-dp 自举放置） |
| `planners/` | 规则规划器、LLM Agent 循环、精调工具、冻结确认 |
| `toolchain.py`、`profiles/` | Hecate trace、`hecate-opt` 调用、随仓库提供的编译参数文件 |
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
| `PT_DACAPO_ROOT` | 提供 Python Hecate 前端的 DaCapo 源码目录 | `<repo>/third_party/ckks-runtime/third_party/dacapo` |
| `PT_RESULTS` | 单次外层运行的输出目录（只由外层读取） | 无 |

`nixrun` 会把前 5 个变量转发进 Nix shell。

## 3. 数据与权重（不在仓库中）

| 内容 | 默认位置（相对 `<poseidon-work>/platforms/<platform>/results/`） | 格式 |
|---|---|---|
| CIFAR-10 | `agent-cifar-offline-records-r1804/cifar10-records/{train,test}-NNNNN.records` | 官方二进制格式：每条 1 字节标签 + 3072 字节 CHW 像素；每个文件 1000 条，NNNNN 为起始下标 |
| 校准与加密测试的选图列表 | `agent-resnet20-original-accuracy-r1805/frozen-selections.json` | `calibration` 和 `encrypted_test` 两个列表，每项为 index、shard、row、label |
| ResNet20 权重 | `agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th` | PyTorch state_dict |
| ResNet20-SiLU 权重 | `third_party/ckks-runtime/third_party/dacapo/examples/data/resnet20.silu.model`（子模块内） | DaCapo 提供 |
| cnn_plain / mlp 权重 | `<PT_AGENT_RESULTS_ROOT>/models-v2/` | 用 `python3 -m pt_agent.tools.nixrun -- -m pt_agent.models.train_small --out-dir <dir> --model cnn_plain`（mlp 同理）重新训练 |

**数据划分**：测试集第 9000–9999 张为循环内的留出验证集；第 8000–8999 张为冻结时的确认集；第 0–7999 张只用于最终报告。训练集只用于校准和训练。

## 4. 常用命令

在 `DSL/DSL_agent/scripts` 下执行：

```bash
# 测试；缺少历史结果、数据或编译器时，依赖它们的测试会跳过，并说明缺了什么
python3 -m pt_agent.tools.nixrun --seconds 3000 -- -m unittest discover -s pt_agent/tests -p 'test_*.py'

# 规则规划器：模型 → PT-DSL → 编译（不调用 LLM）
python3 -m pt_agent.tools.nixrun --seconds 3000 -- -m pt_agent.run --inside --out <dir> --model resnet20 \
    --models-dir <PT_AGENT_RESULTS_ROOT>/models-v2 --planner rule --deg 127 --margin 1.5 \
    --stages plan,check,mock,count,emit,compile

# LLM Agent（需要 DeepSeek API 密钥，放在仓库根目录的 .env 中，由 baseline/agent_credentials.py 读取）
python3 -m pt_agent.experiments.r8b_dev --round <name> --tuner on
```

编译结果写在 `<poseidon-work>/platforms/<platform>/results/pt-agent-*` 下，包括 RuntimePlan 和编译日志。

## 5. 已知限制

- **本机 CPU CKKS 执行**（`toolchain.execute_cpu`，只用于不含自举的小程序）仍走旧的 baseline SDK 选择文件，以及不在仓库历史中的 `normal_cpu_overlay_sandbox`。没有这些时，相关测试会跳过。整网真实 CKKS 需要能执行自举的运行时（GPU）。
- 非可信的候选沙箱（`trace`/`compile` 的 `trusted=False`）同样依赖旧 SDK；pt_agent 的流程使用的是可信运行器（`trusted=True`）。
- 只支持单密文布局（中间结果需要放进 16384 个槽位），kernel 目前覆盖 CIFAR 尺度的 CNN/MLP。
