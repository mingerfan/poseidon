# 静态permute/transpose与真实密文槽位重排

## 语义与实现

schema5 / periodic-packed-v1新增`permute(dims)`与`transpose(dim0,dim1)`。
输入/输出仍为静态rank1..4、最多256个逻辑元素；已有scalar-neuron总量上限16不变。
负轴先按rank规范化，permute必须恰好列出每一轴一次；非法轴、布尔轴、动态参数均拒绝。
transpose可交换相同轴，此时是恒等变换。

这不是reshape的别名。reshape只按当前逻辑C行主序重新解释尺寸；permute改变元素顺序。
例如`[[0,1,2],[3,4,5]]`转置后是`[[0,3],[1,4],[2,5]]`，不是直接reshape到3×2。

`tensor_permutation.py`提供三部分：

- 验证静态轴和维度；
- compiler lowering从原始元素坐标scatter到新C行主序位置，生成source-order和掩码；
- 独立reference从输出坐标递归gather原始嵌套数组，不使用lowering的索引表或掩码。

`fx_to_hecate.Emitter.permute`区分两种物理表示：

1. **packed_prefix**：显式旋转和掩码后相加，真正改变密文槽位。
2. **scalar_neurons**：按新逻辑顺序重排已有密文引用，不把引用位置误认为槽位。

packed路径满足`y[j] = sum_delta mask_delta[j] * rotate(x, delta)[j]`。
`j`是目标槽位；`source(j)`是该输出元素在原输入中的槽位；
`delta = (source(j)-j) mod P`，`P`是实际slot period；正向rotation读取`x[j+delta]`。
每个逻辑输出位置恰有一个掩码为1，padding位置全部为0。旋转只用已有密钥允许的正2次幂步长，
复合位移由多个旋转组成，共享已经构造的中间值。没有新增密钥类型、负步长特例或安全参数。

PyTorch方法的tuple/variadic/keyword形式和`torch.permute/transpose`函数形式都有实际
Torch执行与等价源码/常量/layout测试。只接入显式新版ABI，旧接口继续拒绝新增变换。
当前这些函数的输入张量使用位置参数；轴可以使用上述位置/关键字形式，不宣称接受全部PyTorch签名变体。
这里保留张量数值和逻辑shape，不承诺PyTorch物理stride、view alias或原地存储语义。

## 固定模型与测试

`permutation_model_cases.py`与`cases/tensor-permutation-12-manifest.json`定义12例：

| 案例 | 输入shape | 输出shape |
|---|---|---|
| 非方阵transpose | `[2,3]` | `[3,2]` |
| rank3 permute | `[2,3,5]` | `[5,2,3]` |
| rank4 permute | `[2,3,2,4]` | `[2,2,4,3]` |
| 256元素transpose | `[16,16]` | `[16,16]`，元素顺序改变 |
| 负轴permute | `[2,3,4]` | `[4,2,3]` |
| transpose→Linear | `[3,8]` | `[8,2]` |
| NHWC→NCHW→Conv | `[1,4,4,2]` | `[1,2,2,2]` |
| Linear→scalar transpose | `[2,4]` | `[3,2]` |
| Conv→scalar permute | `[2,1,4,4]` | `[2,2,2,2]`，元素顺序改变 |
| permute→BatchNorm | `[2,3,4]` | `[2,4,3]` |
| transpose→concat | `[2,3]` | `[3,4]` |
| permute→reshape→Linear | `[2,3,4]` | `[4,2]` |

离线测试枚举多个rank1..4形状的所有轴排列，检查非零padding不泄漏、多个重复period一致、
正旋转方向、独立reference与Torch一致、逆变换、恒等变换以及旧接口隔离。
离线槽位解释器只用于诊断，不当作真实密态运行。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --permutation
```

真实Agent付费入口（只在目标授权范围内运行）：

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/tensor-permutation-12-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

最多48次生成/192次HTTP尝试；只发送公开合成模型、公开常量、layout和规则，不发送测试输入、
reference、规则答案或凭据。密态执行并发2，API并发10；编译/后端仍为Dacapo/SEAL CPU。
不将模型级permute计为新增上游DSL primitive；其DSL表达仍使用既有rotate/multiply/add。

## 确定性真实密态基线

报告：`/home/lhy/poseidon-work/results/tensor-permutation-goldens-mzox_ovk/report.json`。
SHA256：`6eb796008ce3a82f1357a4a74215c42f0d44ad7beaad2416cb5935b02a5a1059`。
生成器`deterministic_tensor_permutation_fx`，0次API调用；不是Agent结果。

- 12/12正确程序通过；48组真实密态输入、1816个输出值。
- 最大绝对误差`9.788436605706607e-08`，原门限`1e-5 + 1e-4*abs(reference)`未改变。
- 两个语法合法反例均真实执行后被数值比较拒绝：错误旋转步长误差约2.0，
  错误scalar输出顺序误差约0.10777。
- 已自动清理7597236941字节临时密钥，保留IR/HEVM/CST、模型、reference和解密证据。
- 含本轮规则证据的182项回归全部通过、0跳过。

这些变换仍受AST/算子数/常量数量与乘法深度预算约束；允许单个形状不意味着任意长组合都可运行。
保持安全参数、scale/level管理与数值门限不变，遇到预算或后端问题明确报错。

## 真实Agent结果与失败诊断

报告：`/home/lhy/poseidon-work/results/agent-batch-ks9rbv17/report.json`。
SHA256：`5e8a22652c0f69f4e5bd260beab9102d2f2e2bc629b4beb1ff05b404dc6fc3a5`。

- 首次parse/check/compile/execute均12/12，首次数值正确10/12（83.33%）。
- 反馈修复后12/12通过；共3轮修复，平均每例0.25轮。不是首轮100%。
- 15次官方DeepSeek `deepseek-flash/high`调用，0网络重试、0次缺失usage回执。
- usage：输入116560、输出464458、总计581018 tokens；未读取或估算人民币账单。
- 正确程序48组密态输入、1816个输出值；计入失败尝试共60组密态执行。
- 成功结果最大绝对误差`6.640800753743292e-08`，按输出元素数加权MAE
  `7.51304730732527e-10`。原数值门限及安全参数未变。

| 首稿失败案例 | 实际原因与修复 | 调用数 |
|---|---|---:|
| `perm-negative-axes` | mask-then-rotate程序的旋转方向反了；修复稿在相同掩码下改变旋转符号，密态比较通过 | 2 |
| `perm-nhwc-conv` | 首稿漏掉NHWC→NCHW重排；第一次修复只交换乘法操作数，仍计算同一个错误模型；第二次修复通过 | 3 |

失败均位于`numerical_comparison`，不是TLS、parse或编译失败。所有失败响应、源码、产物和
数值结果均保留。`test_original_numeric_failures_have_specific_layout_causes`进一步核实：

- 负轴案例两稿的掩码集合相同，各对应旋转量模P互为相反数。
- NHWC案例前两稿的解密值均与“只reshape、不permute”的诊断反模型吻合，却不符合原模型。
  该反模型仅用于只读诊断；从未替换不可变reference或被用于接受错误候选。

这说明通过编译并不等于模型语义一致。现有通用数值反馈能最终修复这些案例，但一次无效修复
暴露出提示中布局信息不够明确的问题。后续应为规则加入版本化的节点输入布局和重排掩码语义说明，
保持本批旧请求与首次成功率不变；本轮尚未实施或声称验证该提示改进。

独立审计`agent-lineage-audit-gtkar04j/report.json`通过12/12，0次新增API或密态执行，
0个遗留密钥目录；仍为`poseidon_gpu_validated=false`、`all_goal_requirements_complete=false`。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-ks9rbv17/report.json
```

网络探针`deepseek-tls-probe-j7iz5228`：默认路由3/3连接超时、6478代理3/3通过证书校验并返回
无凭据请求预期的401。实际付费批次显式使用6478，没有关闭证书验证或修改系统代理。

本批自动清理6811822118字节临时密钥，加上规则基线共14409059059字节，约13.42 GiB。
原随机密钥不可恢复，但可重跑实验重新生成；原始响应、模型、编译和数值证据保留。

最终184项回归全部通过、0跳过，包括历史真实密态证据与上述首稿失败的独立诊断。
改动保持本地，未commit/push或切换分支；完整DSL、真实bootstrap及Poseidon GPU执行目标仍未完成。

后续独立实验已加入版本化节点布局说明，设计与新增证据见
[节点布局语义说明 v2](node-layout-guidance.md)。本页的历史请求、报告哈希与首次成功率不随之改写。
