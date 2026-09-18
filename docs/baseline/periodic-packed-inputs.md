# 可变周期输入：schema5 / periodic-packed-v1

本接口把输入元素数从schema4的5–16扩展为1–256。它不是新FHE算法，
也不改变SEAL安全参数；它显式利用现有SEAL_HEVM的周期编码行为。
旧schema1–4、四槽接口、默认waterline40均保留。

后续已增加保留多维shape的运算、广播、reshape和末维Linear，见
[多维张量语义](tensor-input-models.md)。下面的初始12例结果保持原样，不回写为新实验。

## 输入、输出与边界

- 单个加密逻辑输入，rank1–4，每维为正整数，总元素数1–256。
- 按C行主序展平，以0补到最小的P∈{4,8,16,32,64,128,256}。
- 一个密文的16384槽中重复该P元素向量。P不是多项式阶数N；N仍为32768。
- 权重公开固定，CPU float64 reference；固定四组测试输入不发送给模型服务。
- 允许flatten、add/subtract/multiply、negate、square、power2/4、Linear。
- 初次12例的多维输入先flatten；后续多维扩展见上面的独立文档。
  暂不把较小旧接口上的Conv/Pool/BatchNorm等支持外推到schema5。
- Linear单层输出/隐藏宽度最多16。scalar_neurons模式每个输出对应一个广播密文，
  取其slot0；packed_prefix模式返回一个密文，取逻辑长度范围内的连续槽，最多256值。
- 全零权重行使用可信客户端新鲜加密的zero_ct；不是decrypt-and-reencrypt。
- 当前新接口生成程序只开放标量表达式、别名/重绑定、+=/-=/*=、正向二次幂rotation。
  旧接口的native函数/对象数组等构造语法尚未与可变周期接口组合验证。

## 实际链路

原始schema5模型 → PyTorch/FX检查 → 独立Python reference与固定输入
→ 规则程序或真实Agent生成Hecate → AST检查 → 隔离Hecate tracing
→ Dacapo Earth/CKKS/HEVM/CST → 实际密钥检查 → SEAL CPU密态执行
→ 解密 → 与原始模型reference逐项差分。

生产代码位置：

- `scripts/baseline/packed_model.py`：输入算子/shape白名单。
- `packed_input_abi.py`：展平、补零、周期、密文/输出槽绑定规则。
- `fx_to_hecate.py`：独立规则对照；Linear先乘补零权重，再以1,2,...P/2旋转归约。
- `candidate_contract.py`：独立版本化请求、公开常量来源和语义规则。
- `seal_keys/packed_keys.cpp`、`packed_metadata.cpp`：独立可选密钥生成与实际文件验证，
  不替换历史四槽helper，也不修改Dacapo或SEAL库。
- `seal_artifact_gate.py`、`seal_cpu_golden.py`：显式新ABI分支；旧默认行为不变。
- `test_packed_model.py`、`test_packed_evidence.py`：静态、原始reference、真实产物及来源审计。

`run_model_batch.py`旧批量执行器共享四槽密钥，因此明确拒绝schema5；
使用`run_packed_model_goldens.py`、`run_candidate.py`或`run_agent_batch.py`的每例密钥路径。

## 已验证的确定性基线

真实FHE完整报告：
`/home/lhy/poseidon-work/results/packed-model-goldens-ibny_ihk/report.json`

SHA256：`ed7a8282f4c39a439ed9dabcf016465d02ca22bd0c7d759a6450b74eded003a1`。

12个正确程序全部通过，另2个错误归约反例均由数值门限拒绝。
正确程序共48组输入、1208个输出值，最大绝对误差`8.4390344712304e-10`。
这批生成源是`deterministic_packed_fx`，不是Agent，也不是手写golden。
先前两例smoke报告为`packed-model-goldens-y1rq1fh2`，不重复计入12例覆盖。

安全参数仍为N32768、14×60bit模数、SEAL tc128检查、16384槽；无bootstrap。
使用已有明确命名的`seal-cpu-eva-w45-v1`配置，默认配置未改变。
逐项门限保持`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

新增维度的离线检查覆盖19个长度边界×4种rank，共76种shape配置。
有限输入测试不是所有输入上的语义等价证明，也不意味着支持任意shape或全部DSL。

## 复现命令

在WSL源码目录`/mnt/d/Code Space/Poseidon`执行：

```bash
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py
```

12例真实Agent运行入口（付费，只有获得相应授权后执行）：

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/packed-input-12-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

凭据统一从本地`.env`读取，不写入报告、源码、命令或prompt。
每例最多3轮修复，每次生成最多3次网络重试，即12例最多48次生成/192次HTTP尝试。
384000是服务接口的输出上限，不是承诺实际消耗或无限预算。

本轮真实环境无凭据TLS探针：`deepseek-tls-probe-f2ell8bt/report.json`。
默认路由3/3握手超时，6478 CONNECT代理3/3通过证书校验并收到预期401。
短探针不保证长响应永不超时，不关闭证书验证、不写全局代理配置。

## 真实Agent付费结果与独立审计

批次：`/home/lhy/poseidon-work/results/agent-batch-h98vtawd/report.json`。
SHA256：`f532bc6060086e4681083acc3d5e276b014a1e7b54d2d44f1acaef95b4889f98`。

- 12/12首轮通过parse/check/compile/execute/数值比较；无修复、无网络重试。
- 12次真实API调用，输入token73288、输出token93125、总计166413，全部有usage回执。
  这里没有人民币账单数据，不能据此宣称一个已结算费用。
- 48组密态输入、1208个输出值，最大绝对误差`8.038668897203394e-10`，
  按输出值加权的MAE `5.962243597774562e-11`。
- 所有API调用使用官方DeepSeek `deepseek-flash/high`；API并发10，密态并发2。
- 测试模型包括17/31/32/63/64/127/128/255/256元素，覆盖rank1–4；
  Linear最大16输出，另有MLP、fan-out、residual、全零权重行以及256值仿射输出。
- 12例结束后自动清理`8815733471`字节临时密钥，全部密钥目录不存在；
  源码、请求/原始响应、IR/HEVM/CST、输入、reference、解密值和报告保留。
  完整规则基线另清理`10269118772`字节临时密钥。

静态回归和读盘证据审计共109项通过、0跳过。审计核对原始Agent响应→candidate源码
→trace payload→静态检查→产物哈希→实际密钥/level/scale→解密输出→独立原模型reference，
以及原始逻辑输入到padding数组的映射，不仅信任报告中的passed字段。

独立只读审计报告：`agent-lineage-audit-gj1iilvc/report.json`；12例覆盖、0次新API、
0次新密态执行、0个遗留密钥目录，且`all_goal_requirements_complete=false`。
可再次运行（不调用付费API）：

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-h98vtawd/report.json
```

付费前修正了新协议意外继承的“四槽周期/禁止重绑定”提示。原规则基线发生在提示修正前，
未调用LLM，数学规则和执行ABI未改变。历史审计按精确报告哈希和原提示检查静态语义；
没有改写旧请求，没有放宽真实Agent的当前提示验证。付费12例全部使用修正后的提示。

## 后续范围

仍需分别推进较大输入与复杂构造语法的组合、更多算子、更多模型族与结构留出测试。
所有本页密态结果的backend都是`upstream_SEAL_HEVM_CPU`，不是Poseidon GPU。
真实bootstrap、上游所有高层poly辅助函数以及全部DSL语义仍未完成。
