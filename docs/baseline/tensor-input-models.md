# 保留多维语义的模型输入与输出

这一步扩展schema5，不改变既有请求规则、密钥配置或旧schema行为。
目的不是多放几个shape名字，而是保留原模型的轴含义、广播含义和输出shape。

## 新增能力与实现

1. rank1–4的逐元素add/subtract/multiply、negate、square、power2/4不再强制先flatten。
2. 公开常量按PyTorch尾轴规则广播到密文张量的既有shape；尺寸1可扩展，
   但不允许反过来扩大密文shape。密文与密文仍要求相同shape和packing。
3. reshape保持C行主序和元素总数，允许一个可整除的`-1`推断维度。
   它改变逻辑shape而不是槽顺序；没有偷偷加入transpose或slot permutation。
4. 多维Linear只作用于最后一维，保留前导维度。例如`[2,2,8]`与`weight[2,8]`
   产生`[2,2,2]`，不是把32元素全部作为同一个8元素向量。
5. 解密结果同时保留原有扁平数值文件和`decrypted-logical.npy`，后者恢复原模型输出shape。
   `arrays.npz`中的`reference_logical`来自独立原模型reference，而不是解密值。

源码位置：

- `packed_model.py`：rank/广播/末维Linear/输出资源检查。
- `logical_reshape.py`：schema5显式使用256元素预算，旧调用默认8元素不变。
- `fx_to_hecate.py`：广播公开系数；packed输入的多维Linear构造分组块对角权重，
  scalar-neuron输入则按连续组分别做点积。输出命名顺序为原输出张量的C行主序。
- `model_graph.py`：独立Python索引广播、逐组点积reference；不复用块对角lowering。
- `run_model_batch.py::prepare_case`：对照原始PyTorch输出shape与独立reference，冻结两种视图。
- `candidate_worker.py`/`run_candidate.py`：真实密态执行后恢复并检查多维输出绑定。
- `test_tensor_models.py`/`test_tensor_evidence.py`：语义、来源、数值、shape和历史兼容性测试。

生成目标仍是Hecate标量/向量密文表达式。模型中的reshape不是允许Agent任意调用Python
`reshape`；shape变换由可信前端/layout处理。公开广播已经反映在提供给Agent的常量注册表中。

## 当前边界

- 一个逻辑输入，1–256元素，rank1–4，固定shape、固定公开权重。
- packed_prefix输出最多256值；scalar_neurons模式整个张量最多16个输出密文，
  不是每一组都可以独立拥有16个而无总数上限。
- 操作数、常量数、AST、编译深度和runtime资源预算仍须通过现有检查。
- 没有开放扩大密文shape的广播，也未开放transpose/permute、动态shape或密文控制流。
- 旧native构造语法与可变周期接口的全部组合、较大Conv/Pool等仍是后续工作。
- backend仍为`upstream_SEAL_HEVM_CPU`，不是Poseidon GPU；无bootstrap、安全参数或门限变更。

## 确定性真实密态证据

报告：`/home/lhy/poseidon-work/results/tensor-model-goldens-3pf44dof/report.json`。
SHA256：`b34d6ba134a251abaaf9fbea8fc9773effde6e74eeec37f316782c5ddb89c5b7`。

12个正确程序全部通过，48组输入、1012个数值，最大绝对误差`5.218337151280394e-10`。
另2个错误程序均在数值比较阶段被拒绝：错误归约误差约0.1875，误用另一组输入误差约0.660765。
这些程序来自确定性FX规则，不是Agent。全部使用stock SEAL tc128安全参数和命名waterline45配置。
规则批次自动清理`7883508251`字节临时密钥，结果与编译产物保留。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --tensor
```

## 真实Agent测试入口

固定清单：`scripts/baseline/cases/tensor-input-12-manifest.json`。
模型涵盖行/列/中间轴广播、rank2/3/4 Linear、分组MLP/residual、全零权重行、
packed输入reshape和scalar-neuron输出reshape。

下列命令会付费，仅在获得目标内相应授权后执行；凭据从本地`.env`加载。

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/tensor-input-12-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

测试输入、reference、.env及密钥不会发送给模型。每例最多3轮修复，每次生成最多3次网络重试，
即12例最多48次生成/192次HTTP尝试；保留原始失败证据，不用规则答案代替真实Agent响应。

## 本轮真实Agent结果

报告：`/home/lhy/poseidon-work/results/agent-batch-e7vjdf2m/report.json`。
SHA256：`aa2780ae3db57b983564be4ef13322596716262efd6162bc9285d68ac9736d5f`。

- 12/12首轮通过parse/check/compile/execute/数值比较，0修复、0网络重试。
- 48组真实密态输入、1012个输出值；最大绝对误差`6.27567123268058e-10`，
  按输出数值加权MAE `5.122037459150001e-11`。门限仍为`1e-5 + 1e-4*abs(reference)`。
- 12次官方DeepSeek `deepseek-flash/high`调用；输入34927、输出113276、总计148203 tokens。
  所有调用有usage回执；没有读取或估算人民币账单金额。
- API并发10、密态并发2，单次1200秒，输出上限384000 tokens。
- 12例均验证`decrypted-logical.npy`与原始输出shape、C行主序选择及扁平解密值一致。
- 结束后自动删除`6716396740`字节临时密钥，原始响应、源码、IR/HEVM/CST、reference与解密输出保留。
  加上规则基线，共清理约13.60 GiB临时密钥；原密钥不可恢复，但可重跑实验生成新密钥。

| 案例 | 输入shape | 保留的输出shape |
|---|---|---|
| 行/列广播 | `[3,5]` | `[3,5]` |
| rank3广播 | `[2,3,5]` | `[2,3,5]` |
| rank4广播 | `[2,2,4,8]` | `[2,2,4,8]` |
| rank2 Linear | `[2,8]` | `[2,3]` |
| rank3 Linear | `[2,2,8]` | `[2,2,2]` |
| rank4 Linear | `[1,2,2,16]` | `[1,2,2,2]` |
| 分组MLP | `[2,7]` | `[2,2]` |
| 分组residual | `[2,3,5]` | `[2,3,2]` |
| 零权重行 | `[2,8]` | `[2,2]` |
| scalar-neuron reshape | `[2,2,8]` | `[4,2]` |
| packed reshape | `[3,5]` | `[1,3,5]` |

独立只读审计：`agent-lineage-audit-iwc9qo13/report.json`，12/12覆盖，0次新API和密态执行，
0个遗留密钥目录，`all_goal_requirements_complete=false`。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-e7vjdf2m/report.json
```

网络探针`deepseek-tls-probe-adk8ncrs`记录：默认路由3/3 TLS超时，6478 CONNECT代理3/3
通过证书校验并得到无凭据请求预期的401。付费批次使用6478，未发生传输重试。

扩展回归时发现旧`test_spatial`的算子清单遗漏已实现的BatchNorm/reshape/concat，
已补齐精确清单；没有删除测试或放宽空间算子覆盖断言。该失败不是本轮密态计算失败。

最终回归155项全部通过，0跳过。第一次扩展回归有7项历史证据审计未配置路径；
随后根据现有文档定位并重新读取真实spatial/grouped-conv/encrypted-zero报告，
启用这7项后再次运行，全部通过，没有把未执行项计入成功。
本轮20个涉及文件的Python/JSON语法和定向空白检查也通过。

结论仅限这些已验证模型和当前有界接口，不是任意输入上的形式化证明或全DSL完成声明。
