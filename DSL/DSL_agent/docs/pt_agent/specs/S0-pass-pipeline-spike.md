# S0：验证"已管理的 earth MLIR → 跳过 scale 管理 → RuntimePlan"的流水线（不改编译器）

用户决定（2026-10-23）：做 DSL 管理模式（PLAN §4.B 的 S0–S2），并把 S0 提前到 R8f2 之后、U1 之前。

本轮是一次只读验证：
- 不改编译器、前端和运行时，也不改 pt_agent 的任何模块；
- 只新增一个实验脚本和对应结果目录；
- 不调用 LLM，不读取报告集；
- 预计半天。

## 0. 背景（Claude 于 2026-10-23 核对 DaCapo 源码）

- earth 方言有 `rescale`、`modswitch`、`upscale`、`bootstrap` 运算，类型 `!earth.ci<scale * level>` 带 scale 和 level。编译器自己产生的 `lowered.earth.mlir`（由 `--enable-debug-printer` 输出）就是已管理的形式，常量以 `dense<...>` 内联。
- `tools/optimizer.cpp` 中的 `earth::registerEarthPasses()`、`ckks::registerCKKSPasses()`、`registerConversionPasses()` 把以下 pass 注册成了命令行可用的形式：
  - `check-smu`（earth，检查 scale 管理）；
  - `convert-earth-to-ckks`；
  - `convert-upscale-to-mulcp`；
  - `emit-runtime-plan`（ckks）。它的选项涵盖 RuntimePlan 导出的全部参数：prefix、plan-id、target-id、capability-version、operator-spec-id、operator-spec-version、operator-spec-sha256、context-id、device-count、ntt、boot-profile、boot-implementation、inline-payload-max-bytes、plan-format、bundle-format、pretty、report-io。
- 自带流水线（如 eva）在 scale 管理之后依次执行：CSE → canonicalize → [打印 earth.mlir] → convert-earth-to-ckks → convert-upscale-to-mulcp → CSE → [打印 ckks.mlir] → canonicalize → 可选的内存规划 → emit-runtime-plan。
  - `--pass-pipeline` 由 MlirOptMain 提供，`optimizer.cpp:128` 只注释掉了 `registerAllPasses()`，所以标准的 cse、canonicalize 是否可用需要实际确认。

## 1. 要回答的问题

- **Q1**：把编译器自己产生的已管理 `lowered.earth.mlir` 作为输入，只运行"转换 + 导出"部分的自定义流水线，得到的 RuntimePlan 是否与原流水线的 RuntimePlan 相同？
- **Q2**：`check-smu` 能否检查外部给出的已管理 earth MLIR：对正确的输入通过；对人为破坏的输入报错？
- **Q3（附带）**：把已管理的 earth MLIR 送进自带的 `--eva` 或 `--pars` 流水线，编译器是否会保持原有管理不变（幂等），还是会再插入或改动 rescale、modswitch？这决定 S1 能否直接复用自带流水线。

## 2. 输入

使用以下已管理的 earth MLIR，覆盖不同的编译模式和规模：

1. P1 回归的编译目录 `work_result("pt-agent-r5d-p1-20261023T0*")` 中已有的 `lowered.earth.mlir`，以及同目录下原流水线导出的 `lowered._hecate_golden.runtime-plan.json` 和 `lowered.ckks.mlir`。每种编译模式（eva、dacapo-optimized、dacapo-depth-dp、pars 等）各取一个。
2. mlp 和 ResNet20（R8e7 冻结程序）的 dacapo-depth-dp 编译。如果 P2 的编译目录里 `lowered.earth.mlir` 已被 prune 删除，就用 P2 的包重新 trace 和编译一次，本次**不执行 prune**，保留 earth.mlir、ckks.mlir 和 RuntimePlan。这两次编译沿用 trusted 运行器的资源限制。

## 3. 步骤

脚本 `pt_agent/experiments/s0_pipeline_spike.py`（新增）在 Nix 内运行，hecate-opt 取自 `toolchain.host_paths()`。

### 3.1 列出可用的 pass

先运行 `hecate-opt --help`，记录可用的 pass 名称及选项，并确认 `--pass-pipeline` 是否可用。重点确认以下 pass 是否存在：check-smu、convert-earth-to-ckks、convert-upscale-to-mulcp、emit-runtime-plan、cse、canonicalize。

### 3.2 Q1：自定义流水线

对每个输入，按原流水线 earth.mlir 之后的顺序构造流水线：

```
builtin.module(
  cse, canonicalize,
  func.func(convert-earth-to-ckks, convert-upscale-to-mulcp),
  cse, canonicalize,
  func.func(emit-runtime-plan{prefix=<out>/lowered._hecate_<fn> plan-format=json bundle-format=files
                              <其余选项取自 toolchain.compile_argv 中对应模式的 --runtime-plan-* 参数>}))
```

- 写法以 hecate-opt 实际接受的语法为准。如果 cse 或 canonicalize 不可用，就去掉，并在报告中注明。
- 如果 `--pass-pipeline` 不可用，就改用逐个 pass 的命令行开关；仍然不行，Q1 判定为"需要改编译器"。
- 对比新、旧 RuntimePlan：
  - steps 总数、各 op 的计数、boot 数、values 数、输入数、输出数；
  - 逐条比较 steps 的 op、level、scale 字段；
  - bundle 中常量的个数，以及每个常量用 `np.array_equal` 比较。不做文件哈希比较。
- 同时用 `lowered.ckks.mlir` 做参照：新流水线中间打印的 ckks.mlir 与原文件的 op 序列是否相同。

### 3.3 Q2：check-smu

对每个输入：

1. 原样运行 `check-smu`，应当通过；
2. 生成以下变体，每种运行一次，应当报错：
   - 删除第一个 `earth.rescale`，后续类型保持原样；
   - 把第一个 `earth.modswitch` 的 `downFactor` 加 1；
   - 删除第一个 `earth.bootstrap`（只对有自举的输入）；
   - 修改某个运算结果类型中的 level，使之与操作数不一致。

对每个变体，分别记录：
- MLIR 解析（verifier）是否先报错；
- `check-smu` 是否报错；
- 不经过 `check-smu` 直接走 Q1 的流水线时，是否会静默导出 RuntimePlan。

### 3.4 Q3：自带流水线的幂等性

把已管理的 earth MLIR 分别送进 `--eva` 和 `--pars`（参数取 toolchain 的对应模式），对比输出的 earth.mlir 和 RuntimePlan 与输入或原结果是否相同。只记录现象，不据此修改任何东西。

## 4. 判定

- **S0 通过**：Q1 至少对 dacapo-depth-dp 和 pars 两种模式的输入得到相同的 RuntimePlan；并且 Q2 中，`check-smu` 或 MLIR verifier 至少能检出"删除 rescale"和"改 level 类型"两种破坏。
- **部分通过**：Q1 可行，但 Q2 的检查不完整。汇报后由 Claude 和用户决定，S1 是否需要在 pt_agent 内自建检查器。
- **不通过**：Q1 必须改编译器才能实现。停下来汇报，S1、S2 取消。

## 5. 汇报

- 3.1 的可用 pass 清单；
- 每个输入的 Q1 对比表、Q2 的变体结果矩阵、Q3 的现象；
- 实际使用的流水线命令行；
- 判定结论；
- STATUS.md 新增 S0 一行。

## 6. 不要做

- 不修改 DaCapo、hecate-opt、Python 前端、运行时和 pt_agent 的已有模块。
- 只新增 `experiments/s0_pipeline_spike.py` 和结果目录。
- 不删除任何已有编译目录中的文件，变体写在新目录中。
- 不调用 LLM，不读取报告集。
