# Hecate v1 原生算术验证与 Dacapo frontend 修复

日期：2026-09-07。源码：`D:\Code Space\Poseidon`。
本记录继承完整目标：自定义模型/权重、算子与模型族至少翻倍、完整语义验证矩阵、
多输入/rotation/shape/Conv/Pool/近似误差，以及 Poseidon GPU。仅本次原生算术
增量完成验证，不代表整个目标完成。

## 已验证结论

- Hecate 生成契约 v1 允许直接 `-cipher`、`cipher-cipher`、`cipher-public`。
- 正确程序经过 AST gate、真实 Hecate tracing、Earth/CKKS/HEVM/CST、真实 SEAL
  CPU 加密求值解密，与独立 reference 比较通过。
- 交换减法两侧的反例静态/编译/运行都成功，但解密差分失败，符合预期。
- 旧 v0 契约不变；旧五类代表模型的真实密态回归全部通过。
- 没有 API 调用、bootstrap、参数降级或 GPU 执行。

## 版本与接口位置

`hecate_contract.validate_function(..., contract=...)` 默认仍是
`hecate-function-v0`。v1 保留单输入、周期4、输出1..4 ciphertext、直线 SSA、
公开只读常量和资源限制，仅增加原生 subtraction/negation。

`candidate_contract.make_request` 对 schema-1 catalog 生成原来的
`hecate-function-synthesis-v0` 请求；对 schema-2 自定义图生成
`hecate-function-synthesis-v1`。任务版本和对应规则文本一并进入 request hash。
provider 与 validator 同时校验这个对应关系，不能只改版本号或规则绕过门禁。
response JSON 的 `schema:1` 是传输包结构版本，与输入图 schema 和 DSL 契约版本
不是同一概念。这里的 v0/v1 都是本项目受限契约，不是上游 Hecate 发布版本。

`candidate_trace.evaluate_tree` 的新表达式分别调用真实 Hecate `__neg__` 和
`__sub__`；没有执行生成的 Python，没增加任意方法、函数或 import 能力。
公开量在左、public-only 取负、数字字面量、`~x`、`not x`、`+x`、`-=`,
不受支持的旋转和函数调用仍拒绝。negate/subtract 都计入256-operation上限。

## 新发现的 compiler 问题：症状到结论

症状：`-x-c0` 在 tracing 后的 compiler verifier 失败：

```text
'earth.negate' op result #0 must be tensor of HE Cipher Type ...
but got 'tensor<1x!earth.pl<0 * 0>>'
```

原始证据：
`/home/lhy/poseidon-work/results/candidate-replay-n4rvjtfq/attempt-00/compile.log`。
该轮状态为 compiler 失败，不是 API 故障，也不是 FHE 数值误差。

**Confirmed fact**：`third_party/dacapo/tools/frontend.cpp::createBinary` 的
subtraction case 无条件对右操作数创建 `earth::NegateOp`。当右操作数是
`earth::ConstantOp`（plaintext）时，构造结果为 plaintext，而 NegateOp
继承的输出要求是 ciphertext，导致非法 IR。

修复：当右操作数是明确的 dense float public constant 时，先将每个常量
元素精确变为相反数，再构造新的 ConstantOp，最后创建 AddOp。对于 ciphertext
右操作数，仍创建原来的 NegateOp。代数关系是 `x-p=x+(-p)`，其中 p 是公开
固定量，不涉及读取、解密或猜测密文内容。

同时将 valueMap 中取出的 mlir::Value handle 改为按值保存，避免添加中间值
引起 SmallVector 扩容后继续访问之前的元素引用。

只重编译既有 `HecateFrontend` CMake target（parallel2），没有重新安装依赖或
重编 LLVM/SEAL。真实运行记录新增 frontend binary 与 frontend source SHA-256，
并在执行期间核验 binary 没有变化。

本地 Dacapo 基准 commit 仍是
`4616402710f39df3e5f5bd7930a6c036025aaac3`，但 `tools/frontend.cpp` 有本地修改。
不能再把当前 frontend 称为“完全未修改的上游 frontend”；SEAL runtime 未修改。
主仓库 gitlink 和 `.gitmodules` 没有变化。

为避免补丁只存在于未提交 submodule 工作树中，主仓库还保存：
`scripts/baseline/patches/dacapo-plaintext-subtraction.patch`。
已执行 `git apply --check --reverse` 检查它与当前修改匹配，没有实际撤销修改。

## 真实测试结果

统一参数与容差沿用现有 SEAL CPU profile、tc128 参数与 key helper，
`abs(actual-reference) <= 1e-5+1e-4*abs(reference)`，没有调整。
每例4组输入：零、有符号、固定种子随机、[-1,1]边界。
reference 来自 `model_graph.evaluate_reference` 独立计算并先与 PyTorch 对照，
不读取 DSL 或解密值。这里没有 ReLU/SiLU 等近似模型替换。

以下目录均在 `/home/lhy/poseidon-work/results/`：

| 程序 | 目录 | 输出值数 | MAE | 最大绝对误差 | 结果 |
|---|---|---:|---:|---:|---|
| 原生取负/公开常量减法，再 Linear(4,2) | `candidate-replay-gq2y3dwl` | 8 | 5.1278e-9 | 1.1413e-8 | 通过 |
| 原生密文减法：x²-x*w | `candidate-replay-yovkx6a5` | 16 | 5.2109e-9 | 3.0665e-8 | 通过 |
| 反例：x*w-x² | `candidate-replay-8vm9gphy` | 16 | 0.92105 | 约3.5 | numerical_comparison 拒绝 |

正确两例最大非零相对误差分别3.3201e-8、1.2703e-7。三例均记录相同 frontend
SHA-256：`3a7b4672fd5a3692945574cf4534b11ead09860a21a855a70e03471c105c9d81`。
artifact gate 确认正确两例实际包含 HEVM NegateC，而不仅仅在源码写了减号。
所有记录 `agent_calls=0`、`llm_generation_validated=false`、
`backend=upstream_SEAL_HEVM_CPU`、`poseidon_gpu_validated=false`。

真实旧链路回归：`fx-batch-r4t_0c4u`，affine-0、linear-1、mlp2-1、mlp3-1、
flatten_linear-1 全部通过，即20次旧案例输入密态执行。不是付费48模型重跑。

完整离线回归216项：202通过，14项Torch专用测试在通用环境跳过；固定Nix/Torch
环境27/27通过，覆盖了上述14项。原生算术新增9项静态测试与3项真实证据核验。
后者仅在明确指定实验目录时运行，没有证据时skip，不伪造成功。

## 复现

```bash
cd '/mnt/d/Code Space/Poseidon'
# 仅在本地补丁已经检查/应用且需要重编 frontend 时执行，不安装依赖。
python3 scripts/baseline/rebuild_hecate_frontend.py

python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-subneg-linear.json \
  --golden-file scripts/baseline/golden_cases/subneg_linear/native_golden.py

python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-cipher-subtract.json \
  --golden-file scripts/baseline/golden_cases/cipher_subtract/golden.py

# 预期退出1，失败层必须是numerical_comparison，而不是compiler/runtime。
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-cipher-subtract.json \
  --golden-file scripts/baseline/golden_cases/cipher_subtract/wrong_order.py \
  --max-repairs 0

python3 scripts/baseline/run_model_batch.py --smoke
```

以上命令都不读取provider凭据、不调用服务商。真实Agent可通过同一schema-2图的
`--live`使用v1，但本阶段未做新live实验，不能声称新语法的LLM生成率已经验证。

## 尚未完成的目标

语义清单现已关联原生算术的静态、真实执行及反例测试，但算子/模型族翻倍仍
未达到。当前模型输入算子仍8类（不是因为native语法增加又多算2类），模型族
基准8类仍需扩充到至少16。生成DSL primitive从3类扩为5类，单独记录不混算。

下一步仍是rotation/key、多输入、shape/layout与更一般Linear/MLP，继而小型
Conv/Pool以及近似误差分解；每项都需要相同完整证据。CUDA最小组件安装仍等待
用户批准，未自动执行。Poseidon Modswitch/低位宽参数/relin等GPU工作没有在
本轮完成，SEAL结果不能替代它。

## Git 与本地交付

分支`feat/agent-dsl-correctness`、主仓库HEAD不变；原修改、sparse checkout与
skip-worktree保留。新增原生golden/反例、自定义减法图、v1测试、重编入口和
可复现frontend补丁；修改validator/request/provider/AST-dispatch/报告哈希及
语义清单。没有commit、push、PR、安装、sudo或API费用。

以后经批准可拆为两组：`fix(dacapo): fold public subtraction constants before Earth IR`
（含主仓库补丁记录），以及`feat(agent): version native Hecate arithmetic contract`。
尚未提交是用户要求保持working tree；不能将主仓库gitlink升级视为默认操作。
