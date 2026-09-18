# 大输入上的固定统计BatchNorm与concat组合

## 已实现的语义

schema5 / periodic-packed-v1现在接受`batch_norm`和`concat`，复用Hecate基础密文算子，
后端仍是Dacapo SEAL CPU。没有宣称上游`poly.HE_BN/HE_Concat`已通过直接调用验证。

- BatchNorm采用PyTorch推理公式`gamma * (x - mean) / sqrt(var + eps) + beta`，
  channel轴为1，支持输入rank2..4、总元素数<=256。统计量/权重是公开固定向量；
  不允许训练、即时样本统计、密态开方或动态shape。缺省gamma/beta为1/0。
- 确定性前端只在公开数据上折叠gain/offset；原始reference逐个元素直接应用原公式，
  不读取折叠系数。负gamma、零gamma、非零epsilon和无affine均有案例。
- concat支持rank1..4、1..8分支、正/负轴及非拼接轴尺寸一致；总输出<=16。
  packed输入通过one-hot乘法和rotate-and-sum提取成scalar-neuron密文，
  已有scalar-neuron只重排引用，二者可混合。输出是scalar-neuron，不伪装成重新packed。
- concat后的Linear继续按最后一轴运算；BatchNorm前后都可接受本接口支持的Conv。
- 旧BatchNorm/concat默认8元素上限保持不变；只有显式新版ABI选择扩大配置。
- 现有scalar-neuron预算、FHE安全参数、数值门限、compiler waterline均不变。

实现位于`batch_norm_ops.py`、`concat_ops.py`、`packed_model.py`、`fx_to_hecate.py`、
`model_graph.py`。`run_candidate.py`补充记录两个语义模块的来源哈希。
测试与固定案例位于`packed_composition_cases.py`、`test_packed_composition.py`、
`test_packed_composition_evidence.py`及`cases/packed-composition-12-manifest.json`。

| 固定案例 | 输入shape | 输出shape |
|---|---|---|
| BatchNorm rank2 | `[3,5]` | `[3,5]` |
| BatchNorm rank3 | `[2,3,5]` | `[2,3,5]` |
| BatchNorm rank4 | `[2,2,4,8]` | `[2,2,4,8]` |
| BatchNorm 256元素 | `[2,4,4,8]` | `[2,4,4,8]` |
| 无affine | `[2,3,3]` | `[2,3,3]` |
| 全零gamma | `[2,3,5]` | `[2,3,5]` |
| Conv→BatchNorm（部分零gamma） | `[2,1,4,4]` | `[2,2,2,2]` |
| BatchNorm→Conv，256输入 | `[2,1,8,16]` | `[2,1,2,4]` |
| packed分支负轴concat | `[2,3]` | `[2,6]` |
| Conv分支channel轴concat | `[2,1,4,4]` | `[2,2,2,2]` |
| packed/scalar混合concat | `[2,4]` | `[4,4]` |
| BatchNorm→concat→Linear | `[2,3]` | `[2,2]` |

嵌套`nn.BatchNorm2d`和functional BatchNorm路径还做了等价源码/常量/layout及
固定state不变检查；这是离线接口验证，不额外计入12例真实Agent数量。

## 确定性真实密态证据

报告：`/home/lhy/poseidon-work/results/packed-composition-goldens-4jn3zxd_/report.json`。
SHA256：`3f7e421fcb727076baaf2c2af4f66bd8d886acce575d6bfccf432780e3b4d1af`。
生成器为`deterministic_packed_composition_fx`，0次API调用，不将规则程序标记为Agent。

- 12个正确模型全部通过；48组真实密态输入、2228个输出值。
- 最大绝对误差`8.381486615860467e-10`。
- 两个语法合法反例均完成真实密态执行，然后被数值比较拒绝：
  错误BN gain误差约0.86428，错误concat输出顺序误差约1.49916。
- 14次程序验证结束后自动清理8074359097字节临时密钥，原始报告与数值证据保留。
- 回归172项通过、0跳过，包括历史真实密态证据。

最初新增的3个离线测试调用FX时遗漏显式新版ABI清单而失败；已修正测试调用，
生产`prepare_case`路径无需修补，没有放宽shape或安全校验来通过测试。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --composition
```

## 真实Agent测试入口

下列命令会付费，应在目标范围授权内运行。模型描述、公开常量、layout和DSL规则发给provider；
不发送测试输入数组、reference、规则答案、密钥或.env。

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/packed-composition-12-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

最多12*(1+3)=48次生成、48*(1+3)=192次HTTP尝试；硬超时与有限重试保持启用。
密态执行并发2，API并发10。不因网络可用就保证后续所有传输永不失败。

## 真实Agent执行结果

报告：`/home/lhy/poseidon-work/results/agent-batch-j0cqw9mk/report.json`。
SHA256：`0ab6ba6f2f292f872ef2f3d4bd8574324661d5d96b6f7e6ded9ce4f82521f429`。

- 12/12首轮通过parse/check/compile/execute/数值比较，0修复、0网络重试。
- 48组真实密态输入、2228个输出值；最大绝对误差`5.964687077586461e-10`，
  按输出元素数加权MAE `7.031210884224241e-11`。
- 比较原始模型reference，门限仍为`1e-5 + 1e-4*abs(reference)`。
- 12次官方DeepSeek `deepseek-flash/high`调用。usage回执：输入62501、输出118924、
  总计181425 tokens，0次缺失usage。未读取或估算人民币账单费用。
- 所有case均有原始API响应、候选源码、编译产物、密文元数据和解密数值证据。
- 只读独立审计`agent-lineage-audit-tzp3quj8/report.json`通过12/12，0次新API/密态执行，
  0个遗留密钥目录；重新验证原始模型和reference，而非只读取passed标记。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-j0cqw9mk/report.json
```

TLS探针`deepseek-tls-probe-yj0olnan`记录默认路由3/3连接超时、6478代理3/3通过
TLS证书校验并返回预期无凭据401。真实批次显式走6478，未关闭证书校验或改动系统代理。

付费批次自动清理7098095910字节临时密钥，合计本轮规则+Agent清理15172455007字节，
约14.13 GiB。原随机密钥不可恢复；可重跑实验生成新密钥。报告、源码、编译和数值证据保留。

最终173项回归全部通过、0跳过，包含本轮真实Agent证据审计与历史native、chunked、
packed/tensor/spatial结果；回归不产生新付费调用。
现有工作树修改保留，未commit/push/创建PR，也未切换分支。

本轮完成的是有界模型接口的组合扩展；不代表任意shape、全DSL构造组合、上游高层helper、
真实bootstrap或Poseidon GPU编译产物执行已经完成。目标继续保持active。
