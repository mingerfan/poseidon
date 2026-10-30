# PT-Agent 后端对接清单

2026-10-28 初版，2026-10-30 更新（含 R121 编译器修正）。依据当天 `git fetch` 后 mingerfan/poseidon、ckks-runtime、dacapo-modified 各分支的代码与记录；来源见第 6 节。在线版本：https://claude.ai/artifact/JrRJHfMBeJWoA3fDF3u3w9

## 结论

- PT-Agent 已能为 CNN 和 Transformer 生成可编译、模拟精度达标的程序，但还没有任何程序在真实 CKKS 上运行过。流程目前止于 RuntimePlan。
- 我们的 RuntimePlan 现在不能直接在 GPU 上跑。我们用旧编译器（DaCapo `39c1fb2` + 13 文件补丁 + R121 自举段修正）和 qwen48 约定；后端 GPU 用 dacapo-modified `8928817`、OperatorSpec V2 和手动 Q50/P25 自举约定。
- 我们已按 Qwen25 R121 的编译器修正（自举后不再多留一层）对齐默认编译器，自举少 20%–40%；这项修正是否进正式编译器见 Q5。
- 自举数主要由“每次自举后可用几层”决定。我们现在只有 4 层。按成本模型测算，可用 6 层时 R1 从 36512 次降到 12256 次，可用 9 层时各程序都在 5000–7000 次。
- 层数和安全性互相制约。N=65536、128 位安全时，模数总位数上限约 1772；我们现在 Q 已用 1620 位，最多再多约 2 层。后端 25 Q 的配置层数够，但安全级别为 none。
- 请后端回答第 2 节的 11 条问题，最关键的是 Q1–Q3（参数、安全级别、自举配置）。

## 1. 两边现状对照

| 项目 | PT-Agent 现在 | 后端最新 | 状态 |
|---|---|---|---|
| 编译器 | DaCapo `39c1fb2` + 13 文件补丁 + R121 自举段修正（`DSL/DSL_agent/validation/dacapo-r121-boot-window.patch`） | dacapo-modified `8928817`（native-scale-planning）/ `1d40e76`（framework）；我们的版本是它的祖先 | 版本不同 |
| OperatorSpec | v1，context `qwen-bootstrap48` | V2（placement 要求 V2） | 要升级 |
| CKKS 参数 | N=2^16；Q 为 30 个素数共 1620 位（52 / 43×8 / 58×18 / 60×3）；scale 2^43；P 未知 | 手动 Q50/P25、N=2^16、素数 ≤32 位、scale 40；Qwen3 GPU 测试另用 N=2^14、Q50/P8 | 需拍板 |
| 每次自举后可用层数 | 4（第 1 层进、第 5 层出；自举本身用掉 24 层） | 25 个 Q（Q6/s40 进、Q31/s40 出） | 需拍板 |
| 安全级别 | 未认证；Poseidon 自带安全表只到 N=32768 | 全部为 `sec_level_type::none` 或 `--unsafe-*` | 需拍板 |
| rescale 规则 | waterline 45（旧规则） | 默认 `threshold`：按实际素数删 Q | 语义不同 |
| 计划格式 | JSON + 文件 bundle | 默认二进制 `.runtime-plan.bin`；语义版本 V1–V4 | 兼容 |
| 槽数 | DSL 头 `slots=16384` | N=2^16 满槽 32768 | 要对齐 |
| 延迟表 | qwen48 表（DaCapo/HEAAN） | 手动 profile 为占位数；`profiled_HEAAN_GPU_blb5.json` 是候选实测表 | 缺实测 |
| GPU runner | 无 | `poseidon_gpu_plan_e2e` 只支持 1 输入 1 输出；Qwen3 多输入路径中途解密再加密 | 缺通用版 |

### 可用层数对自举数的影响（成本模型测算，2026-10-30）

程序不变、未按新层数重新精调；每次自举的耗时按原表固定。只作量级参考。

| 每次自举后可用层数 | Qwen R0 | Qwen R1 | Agent 冻结程序 | Agent 精调程序 |
|---|---:|---:|---:|---:|
| 4（当前） | 54848 | 36512 | 56800 | 32960 |
| 5 | 36896 | 22560 | 38208 | 20864 |
| 6（N=2^16、128 位安全下的上限） | 21536 | 12256 | 21824 | 10240 |
| 7 | 13184 | 7552 | 13696 | 10016 |
| 9 | 6336 | 6336 | 6624 | 7008 |
| 11 | 6816 | 5216 | 7232 | 6592 |

上表按旧编译器规则计算。我们的默认编译器已按 R121 修正后，可用 4 层时：R0 38849、R1 22721、Agent 冻结程序 20929（§6.19 精调后），自举少约 30%–40%，不需要改参数。

“6 层上限”的前提是 P 为 1 个 60 位素数。若 P 为 2 个 60 位素数，N=2^16 下已无余量；P 更大时现有参数已超出 128 位标准。

## 2. 请后端回答的问题

### 参数与自举（决定自举数，最优先）

- **Q1 CKKS 参数集**：正式的 N、完整 Q 素数链、P 素数链、scale、槽数各是多少？
- **Q2 安全级别**：是否要求 128 位安全？要求的话，N=2^16 下每次自举后最多约 6 层；不要求的话，论文需写明所用级别。
- **Q3 自举配置**：输入层、输出层、输入与输出 scale、分量数各是多少？用哪个自举实现（手动 S2C-first degree59/DA2，还是生产 degree-22）？
- **Q4 编译器版本**：固定 `8928817` 还是 `1d40e76`？之后升级是否通知？
- **Q5 R121 自举段修正**：Qwen25 R121 用的实验编译器改了一处：scale 等于 waterline 时，自举后的段不再预留一层，每段可用层数从 3 变成 4。我们已按它对齐，三方核对一致，自举少 20%–40%。这项修正会进正式编译器吗？最新主线 `8928817` 重写了放置算法，从代码看包含等价逻辑，请确认。
- **Q6 OperatorSpec**：请提供 V2 正式文件。Transformer 能否和 ResNet20 用同一份？

### 精度（决定真实 CKKS 能否算对）

- **Q7 自举输入范围**：输入上界 B 是多少？|x| ≤ B 时自举的最大误差是多少？
- **Q8 自举误差**：能否提供带 arcsine 修正的自举？不能的话，请给出自举实际映射的精确公式，我们在 Mock 中照着模拟。

### 运行（决定能否端到端验证）

- **Q9 GPU 运行器**：需要支持多输入多输出、按 RuntimePlan 生成密钥、中途不解密。由谁实现，什么时候能用？
- **Q10 机器和显存**：哪台机器、哪张卡、多少显存给 PT-Agent 用？
- **Q11 延迟表**：能否按 Q1 的参数实测每一层的 add、mul、rotate、rescale、modswitch、boot 延迟？

## 3. 我们拿到答案后做的事

- **Q1–Q6**：把 13 文件补丁合并到固定的编译器提交并重建；改成本模型（rescale 规则、自举约定、层数）和 DSL 槽数，重算 R0、R1 和 Agent 程序的自举数，恢复“预测 / compiler / managed”三方一致检查。
- **Q7–Q8**：把自举输入上界作为 Agent 和精调的硬约束；Mock 按真实自举映射模拟。CKKS 容差 1e-5 + 1e-4·|ref| 不放宽。
- **Q9–Q11**：换上实测延迟表重跑优化；先用 ResNet20 做第一次 GPU 真实运行，再做 BERT，最后 Qwen。

## 4. 逐项细节

**A1 编译器版本（有差异）**
- 我们需要：一个固定的、后端认可的编译器提交，用它编译 PT-Agent 导出的包。
- 现状：后端最新为 `8928817`（"pin the single-prefix compiler"）。我们的 13 文件补丁要在它上面重新应用，再在 ARM 本机重建（Nix、-j2）。
- 影响：PT-Agent 只换用后端的编译器版本，不修改编译器本身。切换版本要用户批准，补丁可能需要合并。

**A2 正式的 OperatorSpec 和 CKKS context（需拍板）**
- 我们需要：一份 V2 OperatorSpec 和对应的 compiler profile，作为唯一目标配置：`context.rns_moduli` 完整素数链（含 P）、层数上下界、槽数。
- 现状：手动 Q50/P25 可由 `generate_resnet20_manual_profiles.py` 生成，但它是为 ResNet20 定的。我们当前 CPU 参数的 P 不在 operator-spec 里，Qwen CPU 运行程序也不在本仓库。

**A3 原生自举 profile（需拍板）**
- 我们需要：`BootProfile` 的全部字段：input_level_min/max、input_scale_log2、output_level、output_scale_log2、分量数；以及 GPU 侧用哪个 builder 生成。
- 现状：手动 profile 为 Q6/s40 → Q31/s40。我们的 CPU 约定是第 1 层进、第 5 层出，自举用掉 24 层，链中第 6–8 层的 43 位素数很可能被自举最后一步占用。

**A4 自举输入范围（缺失）**
- 我们需要：自举输入的绝对值上界 B，以及该范围内的最大绝对误差。
- 现状：PT-Agent 已有 `boot_input_bound` 和 `boot_abs_error` 检查，Transformer 目标里目前为 null，只报告、不约束。
- 影响：拿到 B 后，Agent 和 tuner 把它当作可行性条件。超界的程序需要把缩放并入相邻权重（PT-Agent 侧要补的功能）。

**A5 自举的近似误差怎么进精度判据（需拍板）**
- 现状：手动 profile 使用 `arcsine_degree=0`，输出是正弦近似，相对理想恒等的最大偏差约 0.00454。
- 冲突：PT-Agent 的 CKKS 判据是对 Mock 参考 1e-5 + 1e-4·|ref|，且不放宽。Mock 把自举当作恒等，每次自举都会带来约 4.5e-3 的偏差，整网必然超差。
- 选项：(a) 后端提供 arcsine 修正（arcsine_degree>0）的 profile；(b) PT-Agent 的 Mock 按 profile 实际的映射模拟自举。两种都不放宽容差。

**A6 rescale 语义与 PT-Agent 的代价模型（有差异）**
- 现状：新编译器默认 threshold 规则，按实际素数删 Q，Boot 前按需对齐。PT-Agent 的代价模型和 S1 managed 模式都复现旧编译器（waterline、depth-dp、EarlyModswitch）。
- 影响：compiler 模式可以直接换编译器。代价模型和 managed 模式要移植到新规则，三方一致检查才能恢复；移植前 GPU 路径只用 compiler 模式。

**A7 输入输出编码约定（有差异）**
- 我们需要：runner 按计划中每个 external input 的 level/scale 编码加密，按 final output 的描述解码；输入初始 level 是否用 `--input-level`（如 Q34）。
- 现状：导出包里有 `prepare_inputs.py` 和 `unpack_outputs.py`，给出槽位排布和 ValueId 对应。槽数要从 16384 改为 32768。

**A8 通用 GPU runner（缺失）**
- 我们需要：读入 RuntimePlan、OperatorSpec 和 bundle；按计划生成 relin、旋转（BERT 需 215 种）和自举密钥；支持多输入多输出；输出解密结果和报告。
- 现状：`poseidon_gpu_plan_e2e` 要求 1 rank、1 卡、1 输入、1 输出。Qwen3 的多输入多输出路径用 `decrypt_reencrypt`，违反“不做中间解密再加密”的规则，不能用于我们的结果。

**A9 显存与密钥规模（需拍板）**
- 现状：手写 ResNet20 GPU 版要求 28 GiB 空闲显存（26 GiB 池）。N=2^16 下 215 种旋转密钥的总量需要核算。
- 影响：如果放不下，PT-Agent 的代价模型要把旋转密钥数作为约束。

**A10 GPU 实测延迟表（缺失）**
- 我们需要：与 A2 参数一致的每层 add / mul / rotate / rescale / modswitch / boot 延迟，格式同 compiler profile 的 `latencyTable`。
- 影响：Agent 和 tuner 按这张表做优化。换表之后要重跑规则基线和 Agent。

**A11 安全参数（需拍板）**
- 现状：两边所有配置都没有标准安全级别。N=2^16 时 128 位安全的 Q+P 上限约 1772 位；后端 Q50/P25（素数 ≤32 位）总位数约 2400，超出该上限。
- 选项：(a) 128 位安全、N=2^16：可用层数最多约 6；(b) 128 位安全、N=2^17：层数足够，内存与时间约翻倍；(c) 用非标准安全级别，论文写明。

## 5. 建议的对接顺序

1. **先定参数和自举（Q1–Q3）**：它决定所有自举数，拿到后先用成本模型重算各程序。
2. **ResNet20**：单输入单输出，现有 `poseidon_gpu_plan_e2e` 稍改就能跑；目标是冻结程序在 GPU 上跑完 40 张图，CKKS 对 Mock 满足 1e-5 + 1e-4·|ref|，预测类别一致。
3. **MLP 或小 CNN**：校准延迟表，检查预测延迟和实测的偏差。
4. **BERT-tiny**：多输入（token 密文和 mask），需要通用 runner 和显存核算。
5. **Qwen2-tiny**：等后端解决 Qwen 路径的加法精度问题后再做。

## 6. 来源

| 内容 | 分支 / 提交 | 路径 |
|---|---|---|
| native scale 规划、25Q 预算、ResNet20 18 次自举、GPU 子路径精度 | poseidon `experiment/native-scale-planning` `280e4715` | `test-results/resnet20-scale-trace-20261027/budget25.md` |
| 编译器用法、OperatorSpec V2、手动 Q50 profile、arcsine_degree=0 | ckks-runtime `378bd95` | `integrations/dacapo/README.md` |
| 原生 Boot 接入方式、runner 边界、剩余工作 | poseidon `framework` `a42fdb8e` | `docs/ckks-runtime-api-next-steps.md` |
| GPU 单输入单输出测试 runner | poseidon `framework` | `src/poseidon/tests/runtime_api/poseidon_gpu_plan_e2e.cpp` |
| BootProfile 字段、configure_native_bootstrap | poseidon `integration/runtime-gpu` | `src/poseidon/runtime_api/poseidon_gpu_api.cpp` |
| 新旧编译器自举数对比、Qwen GPU 回归失败 | poseidon `framework` `590bbc29` | `test-results/model-regression-20261021/results.zh-CN.txt` |
| Qwen3 整模型路径（Host 解密再加密、sec none） | poseidon `integration/runtime-gpu` | `scripts/qwen3_dsl/README.md` |
| 手写 ResNet20 原生自举整网 GPU（HEVM） | poseidon `hsb-resnet20-bootstrap-expansion` | `Benchmark/dsl/resnet20_no_hoist/README.md` |
| R121 自举段修正（编译器补丁与三方核对） | poseidon `lhy-agent-dsl` `fb03206d` | `DSL/DSL_agent/validation/dacapo-r121-boot-window.patch`、`DSL/DSL_agent/docs/pt_agent/STATUS.md` |
| CPU qwen-bootstrap48 参数（Q 链 1620 位、自举 L1→L5） | poseidon `lhy-agent-dsl` | `DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config/operator-spec.json` |
| Poseidon 安全参数表（只到 N=32768） | poseidon `lhy-agent-dsl` | `src/poseidon/basics/util/hestdparms.h` |
| 可用层数对自举数的测算 | 本地运行 2026-10-30 | `pt_agent_results/A1-qwen-explore-relaxed-20261030/levels/LEVELS.json` |
| PT-Agent 当前编译参数与 qwen48 约定 | poseidon `lhy-agent-dsl` | `DSL/DSL_agent/scripts/pt_agent/toolchain.py`、`compiler_profiles/qwen48/` |
