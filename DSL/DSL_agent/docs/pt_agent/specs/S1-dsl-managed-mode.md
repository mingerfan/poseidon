# S1：DSL 管理模式——由 pt_agent 放置自举、rescale、modswitch，写出已管理的 earth MLIR（第一版复现编译器）

用户决定（2026-10-23 加入计划，10/25 提到最前）：在 DSL 阶段描述并管理自举和 rescale。PLAN §4.B、§4.C，S0 已证明可行。

本轮约束：
- **不修改编译器、Python 前端和运行时**；
- 不调用 LLM，不读取报告集（既有回归测试按 AGENTS.md 的例外执行）；
- **现有的 compiler 管理模式完全不变**：Emit 文本、编译命令、RuntimePlan 都不变。

## 0. 背景

### 0.1 S0 的结论

- 已管理的 earth MLIR 经过自定义流水线
  ```
  builtin.module(cse,canonicalize,func.func(convert-earth-to-ckks,convert-upscale-to-mulcp,cse))
  builtin.module(func.func(canonicalize,emit-runtime-plan{...}))
  ```
  导出的 RuntimePlan，与原流水线逐步相同（6 个输入，含 ResNet20 的 23575 步、226 次自举）。
- 破坏管理的 20 个变体都被 MLIR verifier 在解析阶段拒绝。
- `check-smu` 在 Release 构建中只有 assert，NDEBUG 下不生效。
- 自带的 eva/pars 流水线对已管理的输入不幂等。

### 0.2 编译器在 dacapo-depth-dp 模式下的管理（`tools/optimizer.cpp` 中的 "dacapo" pipeline，`--boot-placement=depth-dp`）

依次执行：
1. **`RemoveBootstrap`**；
2. **`GreedyBootstrapPlacement`**（`depth_dp=true`，`lib/Dialect/Earth/Transforms/GreedyBootstrapPlacement.cpp`，约 280 行）：
   - 计算乘法深度；rotate/negate 的深度推迟到其最早的使用者；
   - 统计跨越每个深度的活跃密文数，用动态规划选出切点；
   - 按深度稳定排序运算；
   - 逐个运算处理：
     - 到达切点时，对 level 超过 `safeLevel` 的活跃密文执行 `refresh`：upscale 到 rescalingFactor，bootstrap，再 `normalize` 到 waterline；
     - 乘法前对操作数执行 `processOperandsEVA(waterline)`；
     - 乘法之后执行 `normalize`：upscale `factor + waterline − scale`，然后 rescale；
     - 使用多次的常量会被克隆；
     - 结果类型由 `inferReturnTypes` 得出；
   - 首尾由 `refineInputValues` 和 `refineReturnValues` 处理；
3. **`EarlyModswitch`**（约 110 行）：从后往前，把所有使用者共同需要的 modswitch 前移到操作数上，或者合并进已有的 modswitch；
4. **canonicalize、CSE**（MLIR 自带）。

相关的语义定义在 `lib/Dialect/Earth/IR/`（`ForwardManagementInterface.cpp`、`HEParameterInterface.cpp`）、`Transforms/Common.cpp`，以及 `include/hecate/Dialect/Earth/IR/*.td`。参数（rescalingFactor、levelUpper/LowerBound、bootstrapLevelUpper/LowerBound、waterline、output_val）来自所用的 profile（qwen48 编译配置和 `toolchain.compile_argv`）。

### 0.3 pt_agent 现状

代价模型已经复现了第 2 步的切点和自举计数（E6 40/40，U2/U3 本机编译全部相等），但没有生成实际的管理运算，也没有写出 MLIR。

## 1. 新组件

### 1.1 管理器（`pt_agent/manage/greedy.py`，新增）

- **输入**：lowering 之后的运算 DAG（由记录型后端产生，与 DagBackend 的口径相同：运算种类、操作数、旋转步数、常量）、函数的输入输出，以及 profile 参数。
- **输出**：一个已管理的运算序列。每个值带有 `(scale, level)`；序列中插入 `upscale`、`rescale`、`modswitch`、`bootstrap`；常量按编译器的方式克隆。
- **逐条移植** §0.2 的第 1–3 步：
  - 切点动态规划、深度推迟、稳定排序、refresh/normalize、`processOperandsEVA`、`inferReturnTypes`、常量克隆、`refineInputValues`/`refineReturnValues`；
  - EarlyModswitch。
  
  每个函数的注释中写明对应的 C++ 源文件和行号。
- 第 4 步（canonicalize 和 CSE）**不移植**，交给自定义流水线开头的 `cse,canonicalize`。这与原流水线在 earth 阶段末尾执行的 canonicalize 和 CSE 相同。
- 与代价模型共用切点结果：`placement.depth_dp` 已经实现了切点，管理器应调用它或与之逐项一致（测试保证），不能出现两份不同的实现。

### 1.2 全局约束检查器（`pt_agent/manage/check.py`，新增）

把 `SMUChecker.cpp` 中的断言（它在 Release 中被关闭），以及各运算类型推导中的约束移植成 Python 检查，在写出 MLIR 之前运行。至少检查：
- 每个运算的 `(scale, level)` 与类型推导规则一致；
- add/sub 两侧的 scale 和 level 相同；
- level 在 `[levelLowerBound, levelUpperBound]` 内，自举输入的 level 符合 bootstrap 上下界；
- scale 不超过该 level 的模数容量，并且不低于 waterline（输出值另按 output_val 规则）；
- 函数输入输出的类型与 `refine*` 的结果一致。

违反时报错，给出运算序号和对应的 PT-DSL op id 与行号。

### 1.3 earth MLIR 发射器（`pt_agent/manage/earth_emit.py`，新增）

- 写出 generic 形式的 earth MLIR 文本，包括 `func.func` 的签名、`arg_scale`/`init_level`/`res_scale` 属性、每个运算的结果类型 `tensor<1x!earth.ci<scale * level>>` 或 `!earth.pl<...>`，以及常量的 `value` 和 `rms_var` 等属性。
- 语法和属性以编译器在 `--enable-debug-printer` 下输出的 `lowered.earth.mlir` 为准。常量属性的计算方式（例如 `rms_var`）与 Python 前端相同，查 `python/hecate/hecate/expr.py` 和 frontend 中的实现。
- 不运行 Python 前端，不导入 hecate。

### 1.4 编译命令

`toolchain.py` 新增 `managed_compile_argv(...)`，生成 S0 的两段式流水线。`--ckks-config` 和全部 `emit-runtime-plan` 选项都取自现有 `compile_argv` 对应模式的参数，不另立来源。原有的 `compile_argv` 不变。

## 2. 接入

1. **PT-DSL**：header 增加可选字段 `manage="compiler" | "dsl"`，默认 `"compiler"`。`"dsl"` 本轮只允许与 `refresh="compiler"`、`placement="depth-dp"` 同时使用，其他组合在 check 阶段报"暂不支持"。parser、printer 和 typecheck 同步；省略该字段时，打印结果与原来逐字相同。
2. **代价模型**：`manage="dsl"` 时的 `estimate` 与 `"compiler"` 完全相同（同一个切点）。测试断言两者逐项相等。
3. **Mock 和张量引擎**：不变。管理运算不改变明文数值，自举在 Mock 中本来就是恒等。
4. **run.py 的 compile 阶段**：`manage="dsl"` 时依次执行：记录型后端 → 管理器 → 检查器 → 发射器 → `managed_compile_argv`。读取的 plan_stats 和 plan_latency 与原来相同，编译后 prune（也删除我们写出的 earth MLIR）。
5. **P2 包**：
   - 新增 `core/manage/`（复制管理器、检查器和发射器）以及 `config/management.json`。该文件记录管理模式、profile 参数、切点、自举数，以及每个函数值的 `(scale, level)` 摘要，供阅读和复现；
   - `trace_model.py --mode managed` 运行 forward（记录型后端）、管理器、检查器和发射器，写出 `managed.mlir`，不导入 hecate；
   - `compile_model.py --mode managed` 使用新的命令模板；
   - 默认 `--mode compiler`，行为与原来逐字相同；
   - `model.py`、`ops.py` 和 kernel 不变。

## 3. 验收与测试（新增 `tests/test_s1_*.py`）

**程序集合 M**（都是 depth-dp 模式）：
- 4 个冻结程序；
- R6c、R7c-B；
- U2 的 6 个多块变体；
- U3a 中本机编译过的 3 个（resnet20_relu、squeezenet_silu_avgpool、squeezenet_relu_maxpool）；
- 一组小的单元程序：只有加法；乘法链；跨切点的残差；常量多次使用；输出前没有乘法。

1. **compiler 模式不变**：P2 的 7 个程序、U2 和 U3 的已有程序，Emit 文本、常量和编译命令与 b7662b94 完全相同；省略 `manage` 的 PT-DSL 打印结果逐字不变；4 个冻结模型的 validate、estimate 和 confirm 逐项相同。
2. **已管理 IR 与编译器逐项一致**：对 M 中的每个程序：
   - 用编译器原流水线编译，保留 `lowered.earth.mlir`；
   - 我们发射的已管理 IR 和编译器的 `lowered.earth.mlir`，各自只运行 `cse,canonicalize` 之后，按运算顺序比较：运算种类、操作数的引用关系、结果类型的 `(scale, level)`、属性（rotate offset、modswitch downFactor、upscale 因子），以及常量数值（`np.array_equal`）。SSA 名字和 location 不参与比较。
   - 如果顺序不同但 DAG 同构，报告为"同构"，并说明原因；如果不同构，停下来汇报第一个差异的位置。
3. **RuntimePlan 一致**：M 中每个程序，`manage="dsl"` 的 RuntimePlan 与 `manage="compiler"` 的 RuntimePlan，按 S0 的 Q1 口径比较：步数、各运算计数、自举数、values 数、输入输出数、逐步的 op/level/scale，以及常量数值。要求全部相同。
4. **检查器**：
   - 对 M 的已管理 IR 全部通过；
   - 对 S0 中的 4 类破坏，加上"scale 超过容量""自举输入 level 越界"两类，都能报错，并带运算序号和 PT-DSL 行号，而且在送进编译器之前就报出。
5. **切点只有一份实现**：管理器的切点与 `placement.depth_dp` 对 M 中所有程序逐项相等。
6. **P2 包**：M 中至少 4 个程序（含 1 个多块程序）的包，在 `--mode managed` 下 trace 和编译成功，RuntimePlan 与 compiler 模式相同。包内的管理代码不导入 pt_agent，也不导入 hecate。
7. 既有 481 项测试全部通过。

**资源**：编译沿用 trusted 运行器的限制；每个程序比较完成后，立即 prune 编译目录和我们写出的 MLIR。

## 4. 汇报

- 改动文件；
- §3 各项结果。§3.2 和 §3.3 每个程序一行：运算数、自举数、比较结论（相同/同构/不同）；
- 从 C++ 移植时遇到的语义细节（例如常量克隆、深度推迟、EarlyModswitch 的合并），以及对应的源码位置；
- STATUS.md 新增 S1 一行。

## 5. 不要做

- 不修改编译器、前端、运行时；不修改 compiler 模式下的任何输出。
- 本轮不做 pars/agent 刷新模式下的 DSL 管理，也不做任何优于编译器的管理（那是 S2）。
- 不调用 LLM；不下载、不安装；不放宽资源限制。

## 6. 修订 §1.2：检查器的口径（Claude 于 2026-10-25，取代 §1.2 与 §3.4 中的相应内容）

### 6.1 层数用 Earth 的口径

检查器只在 **Earth 语义**上工作：类型 `!earth.ci<scale * level>` 中的 `level` 是**已消耗的层数**，0 表示顶层，rescale 和 modswitch 每次加 1。编译器的 GreedyBootstrapPlacement 也是这个口径：`lastBootLevel = bootstrapLevelUpperBound − bootstrapLevelLowerBound`，并与 `type.getLevel()` 比较。

profile 中的 `levelLowerBound`/`levelUpperBound` 属于转换到 CKKS 和物理层之后的口径，**不在检查器的范围内**。这一部分由 §3.3 的 RuntimePlan 逐步比较，以及编译器自身的转换来保证。合法的 Earth IR 中出现 level 0 是正常的。

### 6.2 不移植 SMUChecker，检查以下三类不变量

`SMUChecker` 验证的是 ScaleManagementUnit 分析的分组一致性。这个分析服务于 optimized/ELASM 流水线，不能作为 depth-dp 管理是否合法的判据：它对编译器自己生成的 depth-dp IR 也报出 42 处（MLP）和 392 处（ResNet20）不一致。**不移植它，不做任何 SMU 分组检查**，也不需要处理管理运算前后的分组。

检查器只检查以下三类：

**A. 逐运算的类型规则**：每个运算结果的 `(scale, level)`，必须等于按编译器 `inferReturnTypes` 规则（及相关接口）由操作数类型和属性推出的值。覆盖 add、sub、mul（cc 与 cp）、negate、rotate、constant、rescale、modswitch、upscale、bootstrap。

这正是 MLIR verifier 和类型推导在编译器中实施的局部规则。移植时注明源码位置。

**B. depth-dp 管理自身的不变量**：取自 `GreedyBootstrapPlacement.cpp` 中的 `emitError` 条件与 refresh/normalize 的约定：
- 函数的密文输入，level ≤ lastBootLevel；
- 到达每个切点时，所有活跃密文的 level ≤ lastBootLevel，且 scale = waterline；
- depth-dp 下，乘法的密文操作数 level < lastBootLevel；
- 自举的输入 scale = rescalingFactor（refresh 先 upscale 到 factor）；
- 所有密文值的 level 在 [0, lastBootLevel] 内。

**C. 函数边界**：输入输出的类型，以及 `arg_scale`、`init_level`、`res_scale` 属性，与移植的 `refineInputValues`/`refineReturnValues`（`Common.cpp`）的结果一致。

**容量检查**：只有在编译器源码中能找到明确的"scale 与层数容量"规则时（需引用文件和行号），才加入。找不到时不做这项检查，并在汇报中说明，由 A、B、C 加上 §3.3 的 RuntimePlan 比较来保证。

### 6.3 修订 §3.4 的负例（6 类）

| 破坏方式 | 应由哪类检查报出 |
|---|---|
| 删除一个 rescale | A |
| modswitch 的 downFactor 加 1 | A |
| 删除一个自举 | B（后续 level 超出 lastBootLevel，或乘法操作数 level 过高） |
| 某个结果类型的 level 加 1 | A |
| 自举输入未 upscale 到 rescalingFactor | B |
| 函数输入的 level 超过 lastBootLevel | B 或 C |

检查器必须在把 MLIR 送进编译器之前报出，并且给出运算序号和 PT-DSL 的 op id、行号。

对 M 中编译器自己生成的已管理 IR（`lowered.earth.mlir`），用同一个检查器检查，必须**全部通过**。这是检查器本身的正确性测试：如果编译器的合法输出被误报，说明规则移植错了，停下来汇报。

## 7. 补充：earth 尾部 pass 序列对齐编译器（Claude 于 2026-10-25，MLP 首次 IR 差异后）

**现象**：冻结 MLP 的 DSL IR 与编译器的 `lowered.earth.mlir` 相比，多出两条重复的 modswitch 和一个重复常量（见 `s1-mlp-20261025T181607368492/DUPLICATES.json`）。五个小程序已经通过。

**源码核对**（`tools/optimizer.cpp` 的 "dacapo" pipeline，第 598–626 行）。`EarlyModswitch` 之后，编译器依次执行：

```
canonicalize → cse → [check-smu，仅在启用时] → cse → canonicalize
→ [打印 lowered.earth.mlir]
→ convert-earth-to-ckks → convert-upscale-to-mulcp → cse
→ [打印 lowered.ckks.mlir] → canonicalize → emit-runtime-plan
```

所以作为基准的 `lowered.earth.mlir` 已经完整经过 `canonicalize,cse,cse,canonicalize`。而 S1 的比较入口和直接转换入口只做了 `cse,canonicalize`，次序与次数都不同。canonicalize 先把运算折叠成相同的形式，后面的 cse 才能合并它们，所以会残留重复项。

**决定：两处都对齐到编译器的序列。**

1. **`managed_compile_argv`**：流水线改为与编译器 `EarlyModswitch` 之后完全相同的序列（不含打印和 check-smu）：
   ```
   builtin.module(canonicalize,cse,cse,canonicalize,
                  func.func(convert-earth-to-ckks,convert-upscale-to-mulcp,cse))
   builtin.module(func.func(canonicalize,emit-runtime-plan{...}))
   ```
   仍然分成两段，以便保存中间的 ckks.mlir。第一段中 pass 的嵌套层级（module 级还是 func 级）与编译器的 `pm.addPass` 或 `pm.addNestedPass` 一致。
2. **§3.2 的比较口径**：
   - DSL 发射的 IR 先运行 `canonicalize,cse,cse,canonicalize`，再与编译器的 `lowered.earth.mlir` 比较；
   - 基准文件原样使用，它已经是这个序列的输出；
   - 另外对基准再运行一次同样的序列，确认它幂等（输出与输入相同）。如果不幂等，汇报，但不影响判定。
3. **这不等于移植 canonicalize 或 CSE**。管理器仍然只移植 GreedyBootstrapPlacement 和 EarlyModswitch，与 §1.1 一致。尾部的通用 MLIR pass 由编译器自己执行。
4. **接续**：
   - 从 MLP 接续，继续 §3 的集合 M；
   - 已通过的 5 个小程序的历史结果保留，在最终验收时**用新序列重跑一次**（成本很低），保证 M 中所有程序使用同一条流水线；
   - S0 的 6 个输入不需要重跑：它们的基准已是尾部序列的输出，新序列对它们幂等，这一点由第 2 点中的幂等检查覆盖。
5. 对齐之后，如果 MLP 或其他程序仍有差异，按 §3.2 停下来汇报第一个差异的位置，不再额外增加 pass。

## 8. 补充：常量 `rms_var` 属性的精确复现（Claude 于 2026-10-26，CNN plain 的 16 处属性差异后）

**现象**：按 §7 的序列，MLP 已完整通过。CNN plain 两侧都是 4685 个运算、72 次自举，结构、类型和 829 个常量载荷都相同，但有 16 处 `rms_var` 不同（首处为 0.13461690297668655 对 0.13461690297668372）。用 libm 逐项 FMA 累加可以精确重现编译器的全部 16 个值（`FMA_PROOF.json`）。

**源码依据**：`include/hecate/Dialect/Earth/IR/EarthOps.td` 第 145–157 行，ConstantOp 的 builder：

```cpp
double rms = 0.0;
for (auto && dval : arr) rms += dval * dval;   // aarch64 GCC 构建默认把这一步收缩为 fma(dval, dval, rms)
rms /= arr.size();
rms = std::sqrt(rms);
```

NumPy 的 `np.sum(x*x)` 使用成对求和，乘法和加法也分别舍入，所以最后几位不同。`rms_var` 只被 `estimateValue`（optimized/ELASM 的值估计）和 `PrivatizeConstant` 读取。depth-dp 模式下应当不影响 RuntimePlan，但它是 IR 的一部分，**比较不放宽，仍然精确复现**。

**修复**：

1. 发射器按 builder 的顺序计算：
   - 从 0.0 开始，按数组顺序逐项累加，每一步都是一次融合乘加 `rms = fma(d, d, rms)`；
   - 然后计算 `rms / n`，再用 `math.sqrt`（IEEE 正确舍入，与 `std::sqrt` 相同）。
   
   FMA 通过 `ctypes` 调用系统 libm 的 `fma`。找不到 `fma` 时直接报错，不能退回到 NumPy 求和。
2. **累加方式作为平台参数**：上面的"每步 FMA"对应当前平台上的编译器构建（aarch64、Nix GCC）。在 profile 中记录 `rms_accumulation: "fma"`，并支持 `"separate"`（先乘后加，各自舍入），用于未来在不做收缩的构建上使用。当前 profile 只使用 `"fma"`。不对不同平台做猜测：新平台需要按第 3 条的测试确定该值。
3. **测试**：
   - FMA_PROOF 中的 16 个值逐位相同；
   - MLP、CNN plain 以及 5 个小程序中，编译器 `lowered.earth.mlir` 里**所有**常量的 `rms_var`，与发射器的计算逐位相同（比较 float64 的位模式）；
   - 用 `"separate"` 模式重算同一批常量，至少有一个值与编译器不同，证明测试能区分两种模式；
   - libm 中没有 `fma` 时报错（mock 掉 ctypes 的查找）。
4. **性能**：逐项调用 ctypes 会比较慢，ResNet20 约有 4500 个常量，每个 16384 项。允许按常量值去重缓存 `rms_var`（相同载荷只算一次）。只要结果逐位相同，也允许其他精确实现。记录发射耗时。
5. P2 包的 `core/manage/` 同步这一实现。包内只使用标准库的 `ctypes` 和系统 libm，不导入 pt_agent。

**接续**：从 CNN plain 接续 §3 的集合 M，已通过的 MLP 保留。如果还有其他属性或数值的差异，按 §3.2 停下来汇报，不放宽比较。

## 9. 补充：ResNet20 的 IR 差异——逐 pass 定位与修正（Claude 于 2026-10-26）

**现象**：compiler 23331 个运算，DSL 23349 个，多出 18 个。首处差异在第 307 个运算：

| | 运算 |
|---|---|
| 编译器 | `modswitch(op280, downFactor=1)`，op280 已处于 level 3 |
| DSL | `modswitch(op257, downFactor=4)`，op257 处于 level 0 |

两者结果都是 level 4，语义等价，但结构不同：编译器形成了"先降 3、再降 1"的 modswitch 链。两侧检查器都通过，编译器基准是幂等的。MLP、CNN plain 已完整通过，它们没有残差这种"一个值被多处以不同 level 使用"的结构。

**Claude 的初步判断**（需要验证）：差异来自 `EarlyModswitch` 的移植。
- 原 pass 从后往前遍历时，会把 modswitch 前移到操作数上（新建 `ModswitchOp` 并 `setOperand`）。这些新建的 modswitch **立即成为该操作数的使用者**，MLIR 的 use-list 是动态更新的。遍历到更靠前的运算时，`getUses()` 能看到它们，并继续吸收或合并（"Modswitch movement can be absorbed into modswitch"，以及对使用者的 downFactor 递减）。
- 如果移植时在遍历前静态计算使用者集合，或者对"是否全部使用者都是 modswitch"、使用次数（同一运算两次使用同一个值算两次）的判断与 MLIR 不同，就会在残差、捷径这类多使用者结构上产生不同的链。
- 也不排除差异来自 `GreedyBootstrapPlacement` 中的 `processOperandsEVA`（操作数对齐时插入 modswitch 的位置和基准值），或者 `live` 集合（SetVector，按插入顺序）的迭代次序。

### 9.1 逐 pass 定位（不改编译器，只用已注册的 pass）

对 ResNet20 的前端 trace（`candidate_trace.mlir`），用 hecate-opt 的 `--pass-pipeline` 分阶段输出编译器的中间 IR。pass 的选项与 dacapo-depth-dp 模式相同，取自 `toolchain.compile_argv`：

1. 阶段 a：`remove-bootstrap`，接 `greedy-bootstrap-placement{waterline=… output-val=… depth-dp=true}`；
2. 阶段 b：在 a 之后加 `early-modswitch`；
3. 阶段 c：在 b 之后加 `canonicalize,cse,cse,canonicalize`。c 应当等于现有的 `lowered.earth.mlir`，用来确认阶段划分正确。

管理器也在相同的两个点输出中间结果：GreedyBootstrapPlacement 之后，以及 EarlyModswitch 之后。**两侧中间 IR 都先只做 `canonicalize`**，或者都不做后处理，口径一致即可；然后按 §3.2 的方式逐项比较 a 和 b。由此确定第一处差异出现在哪个 pass，以及对应 C++ 源码的哪一段逻辑。

如果第 1 条中某个 pass 选项名在命令行上与源码不同，按 `hecate-opt --help` 的实际名字使用，并记录下来。

### 9.2 修正

- 按定位结果修正对应的移植（属于 AGENTS.md 的自写错误例外）。修正必须是**更忠实地复现 C++ 语义**，例如在 EarlyModswitch 的反向遍历中动态维护 use-list，并按 MLIR 的使用计数口径（每个操作数槽各算一次）判断"全部使用者都是 modswitch"。**不能**为了匹配而加入针对特定模式的后处理，也不能增加 canonicalize/CSE 之外的 pass。
- 为定位到的语义细节补充单元测试：构造能重现该行为的最小程序，例如一个值同时被"降 3 层的乘法"和"降 4 层的残差加法"使用，阶段 a、b 的中间 IR 与编译器逐项一致。
- 修正后：
  - 已通过的 MLP、CNN plain 和 5 个小程序**重跑**，必须仍然全部通过；
  - 然后从 ResNet20 接续集合 M。

如果 9.1 表明差异来自编译器某个 pass 依赖于 MLIR 内部实现细节、在 Python 中无法确定性复现（例如依赖指针地址的哈希顺序），停下来汇报证据，由 Claude 和用户决定是否改为"同构即可"的验收口径。

每次比较完成后 prune 中间 IR。

## 10. 补充：管理器的输入必须是前端预处理后的共享 DAG（Claude 于 2026-10-26，取代 §9 中"初步判断"的方向）

**定位结论**（codex，§9.1）：差异源自管理器的**输入**，不在 EarlyModswitch。`tools/frontend.cpp` 的 `save()`（第 124–131 行）在写出 MLIR 之前执行：

```
cse → canonicalize → earth::privatize-constant（func 级）→ canonicalize
```

编译器收到的是这串预处理之后的 DAG，其中重复构造的 `-x` 等已经合并。管理器用的是 lowering 原始的记录图，仍然保留这些重复。最小复现显示：重复构造 `-x` 时，阶段 a、b 都不同；复用同一个 `-x` 时，a、b 都完全相同。

**决定：不在 Python 中移植 CSE、canonicalize 和 privatize-constant，而是由 hecate-opt 执行与前端 `save()` 相同的 pass 序列**。这几个 pass 都已在 hecate-opt 中注册，不需要修改编译器，得到的 DAG 与编译器的输入在构造上一致。§1.1 中"canonicalize 和 CSE 不移植"的原则不变。这里是**调用**编译器已有的 pass，不是移植。

### 10.1 新的 DSL 管理流程

```
lowering（记录型后端）
 → 原始 earth 发射器：写出与前端 builder 相同形态的未管理 IR
   （类型 ci<0 * 0>、pl，常量带 value 与 rms_var（§8），函数签名与前端相同）
 → hecate-opt：builtin.module(cse,canonicalize,func.func(privatize-constant),canonicalize)
   （与 frontend.cpp 的 save() 相同，嵌套层级一致）
 → 解析得到的 generic 形式 MLIR，得到共享 DAG
 → 管理器（GreedyBootstrapPlacement、EarlyModswitch 的移植）→ 检查器 → 已管理 IR 发射器
 → 编译器尾部序列与转换（§7）
```

1. **原始 earth 发射器**：与 §1.3 的发射器共用代码，只是不带管理运算，也不带 scale/level 类型。
2. **MLIR 解析器**（`pt_agent/manage/mlir_read.py`，新增）：只需支持前端和本流程会出现的 generic 运算子集：`earth.add/sub/mul/negate/rotate/constant`、`func.return`，以及函数签名和属性。遇到不认识的运算就报错，不静默跳过。
3. **预处理的命令行**：由 `toolchain.py` 新增的 `frontend_preprocess_argv(...)` 生成，与 `managed_compile_argv` 一样只有一个来源。
4. **P2 包**：`trace_model.py --mode managed` 增加 `--hecate-opt` 参数，用于这一步预处理。包仍然不导入 hecate 的 Python 前端，也不导入 pt_agent。

### 10.2 新增的验收（阶段 0）

在 §3.2 之前增加一步：对集合 M 的每个程序，"原始发射 + hecate-opt 预处理"的输出，与经 Python 前端 trace 得到的 `candidate_trace.mlir`（P2 包的 compiler 模式）逐项比较，口径同 §3.2（运算种类、引用关系、类型、属性、常量数值和 rms_var，忽略 SSA 名字和 location），必须完全相同。这一步保证管理器的输入与编译器的输入一致，之后的差异只可能来自管理器本身。

### 10.3 代价模型

**不变**。`estimate` 与冻结路径仍使用原来的记录图，因为冻结 Agent 的预测值不能改变。在汇报中统计：对集合 M，代价模型的切点和自举数，与"预处理后的 DAG 上的切点和自举数"是否相同。历史上自举数都与编译结果相等，预计相同；如有不同，列出程序，但不修改代价模型。§2.2 的"两种管理模式下 estimate 相等"保持不变。

### 10.4 接续

1. 用 §9 的最小复现（重复构造 `-x`）验证新流程：阶段 0、a、b 都与编译器相同。
2. MLP、CNN plain 和 5 个小程序在新流程下重跑，必须仍然通过。
3. 从 ResNet20 接续集合 M。

如果 EarlyModswitch 或 GreedyBootstrapPlacement 的移植另有差异，再按 §9.1 逐 pass 定位。

## 11. 补充：大中间 IR 的压缩与流式传递（Claude 于 2026-10-26，SqueezeNet-SiLU 磁盘准入停止后）

**现象**：集合 M 中 18/20 完整通过。SqueezeNet-SiLU 在原模式下 trace 成功，得到 1,923,338,498 字节的 MLIR。随后原始 trace 和新的 raw IR 同时留在磁盘上，使预处理启动前的空闲空间低于 10 GiB 的准入线。清理后空闲 14,371,901,440 字节。还有 1 例待执行。

**决定：采用压缩中间 IR、通过 stdin 交给现有 pass**。不转服务器，不放宽 10 GiB 准入线和任何资源限制。依据：
- hecate-opt 的位置参数默认值就是 `-`（stdin）；
- S0 的 Q2 已经用"解压后经 stdin 输入"的方式运行过，路径经过验证；
- MLIR 文本中大部分是 dense 常量，压缩率高。

### 11.1 规则（只用于大文件，阈值为 256 MiB；小文件维持现状）

1. **我们自己写出的 IR**（raw IR、已管理 IR）：由发射器直接以 gzip 流写成 `*.mlir.gz`，不先落盘未压缩的版本。
2. **前端 trace 得到的 `candidate_trace.mlir`**：trace 完成后立即压缩为 `.mlir.gz`，校验能解压（只检查格式，不做哈希比较）后删除未压缩的文件，再进入下一步。
3. **调用 hecate-opt**：由 Python 用 `gzip` 模块分块解压，通过 `subprocess` 的 stdin 管道送入（输入参数写作 `-`）。需要文本输出的 pass（预处理、尾部序列）用 `-o -` 输出到 stdout，再由 Python 分块写成 `.mlir.gz`。`emit-runtime-plan` 照常写出 RuntimePlan 和 bundle 文件。不依赖系统的 `gzip` 命令。任何一端出错（返回码非 0、管道断开），整步视为失败，不能产出半个文件。
4. **读取 IR**：`mlir_read` 和 §3.2 的比较都直接读取 `.mlir.gz` 流。
5. **生命周期**：每个中间文件在最后一个使用它的步骤完成后立即删除。一个程序全部比较完成后，按 `toolchain.prune` 的规则清理，并删除我们写出的 `.mlir.gz`。准入检查（≥10 GiB 空闲）在每个重步骤之前执行，口径不变。
6. **等价性**：用 MLP 和 CNN plain 各走一遍压缩路径，解压后的预处理输出、已管理 IR 和尾部序列输出，与非压缩路径的输出**文本逐字相同**，RuntimePlan 的比较结论也相同。这一项进入测试。

### 11.2 接续

在压缩路径下完成 SqueezeNet-SiLU 和剩余的 1 例。已通过的 18 例保留结果，不需要重跑。

如果压缩之后单个程序仍然无法在 10 GiB 准入线之上完成，停下来汇报峰值占用和瓶颈步骤。这时剩余程序转为服务器作业，并在汇报中标注"本机未完成"，不放宽限制。

最后执行既有回归（481 项及 S1 新增测试），然后汇报 §4 的全部内容。

### 11.3 补充：调试位置的口径（Claude 于 2026-10-26）

**现象**：MLP、CNN plain 的普通与压缩路径，4 次完整的 IR 和 RuntimePlan 比较都通过；已管理 IR 和尾部序列文本都逐字相同。但两份预处理文本各有 5 行调试位置不同：普通路径中写的是原始文件名，压缩路径中写的是 `<stdin>`。这些行是没有显式位置的结构（模块、函数、参数、return）从解析器继承的默认位置。

**决定：采用"显式固定位置"**，不放宽 §11.1 第 6 条的"逐字相同"。

1. 原始 earth 发射器（§10.1）给**模块、函数、函数参数和 `func.return`** 显式写上 `loc(unknown)`，使它们的位置与输入渠道（文件或 stdin）及文件名无关。
2. 所有 PT-DSL 运算产生的 earth 运算，保留现有的位置信息（指向 PT-DSL 的 op 和行号），不能改成 unknown，因为报错和检查器的定位要用到它们。
3. 已管理 IR 发射器按同一规则处理。
4. 这一改动只影响调试位置。§3.2 和阶段 0 的比较本来就忽略 location，RuntimePlan 的比较口径也不包括位置，所以**已通过的 18 例结果保持有效**，不重跑整个集合 M。
5. **重跑**：MLP、CNN plain 的普通与压缩路径等价测试（共 4 次比较）。预处理、已管理和尾部序列三类文本都必须逐字相同，RuntimePlan 的比较结论也相同。另外为第 1 条补一个单元测试：同一份 raw IR 分别通过文件和 stdin 交给预处理，输出文本逐字相同。
6. 然后按 §11.2 在压缩路径下完成 SqueezeNet-SiLU 和剩余的 1 例，执行回归，并按 §4 汇报。

### 11.4 补充：超大输入改用临时常规文件（Claude 于 2026-10-26，SqueezeNet-SiLU 退出 134 后）

**现象**：32 项测试和 MLP/CNN 的两项严格文本等价都通过。SqueezeNet-SiLU 的 raw IR 为 2,397,613,997 字节，已压缩。经 stdin 交给 hecate-opt 时，在 `getMemoryBufferForStream → getNewUninitMemBuffer` 申请内存时退出 134，还没有进入解析或 pass。峰值 RSS 为 2294.53 MiB，磁盘最低空闲 11,626,680,320 字节。

**原因**：LLVM 从流读取时，要把整个输入读进一块连续的堆内存，读取过程中还要扩容，峰值需要输入大小的数倍。从常规文件读取则使用内存映射，不需要这份拷贝。U3a 中 SqueezeNet 在 compiler 模式下，用同一个 hecate-opt 从文件输入本机编译成功过，可以作为参照。

**决定：采用临时常规文件的例外**，不导出服务器作业。修订 §11.1 第 3 条：

1. **门槛**：解压后大于 1 GiB 的输入，不再走 stdin。Python 先把它分块解压到**一个**临时常规文件中，放在本程序的运行目录下，然后把这个路径作为 hecate-opt 的输入参数。不超过 1 GiB 的输入仍走 stdin（已验证）。
2. **输出不变**：需要文本输出的 pass 仍然 `-o -`，由 Python 写成 `.mlir.gz`；`emit-runtime-plan` 照常写出文件。
3. **生命周期**：临时文件在 hecate-opt 返回后立即删除。无论成功还是失败，都在 `finally` 中删除。同一时刻最多存在一个这样的临时文件。
4. **准入**：在解压之前检查：空闲空间 ≥ 10 GiB + 临时文件的预计大小（取 raw IR 的原始字节数）。不满足就停下来汇报，不尝试执行。6 GiB 的进程内存限制、8 GiB 等其他限制都不变。
5. **等价性**：对 MLP 和 CNN plain，把门槛临时设为 0，强制走临时文件路径，输出文本与 stdin 路径、普通路径逐字相同（依赖 §11.3 的固定位置）。这一项进入测试。
6. **接续**：在这一路径下完成 SqueezeNet-SiLU 和剩余的 1 例。如果仍然因为内存或磁盘失败，停下来汇报峰值和所在步骤；这时剩余程序转为服务器作业，标注"本机未完成"，不放宽限制。然后执行回归，按 §4 汇报。
