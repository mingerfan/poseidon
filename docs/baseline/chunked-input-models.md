# 较大逻辑输入：schema 4 / 分块输入 v1

后续独立接口已扩展到1–256元素，见[可变周期schema5](periodic-packed-inputs.md)。
本页继续保留schema4当时的限制、请求身份和实验记录，不回写历史结论。

## 解决的问题与当前边界

原 schema 2 每个用户输入只有4个元素；schema 3 支持2..4个独立用户输入，
但各输入仍为4个元素。这并不等于支持一个16元素模型。

新增 schema 4 接收**一个原始逻辑输入**：rank1..4，各维正整数，总元素5..16。
它经C-order展平后拆成2..4个连续4元素块，末块不足4元素时补零；
各块单独加密，并沿CKKS槽重复自身4元素周期。N、Q、SEAL安全检查不变。
这不是把原模型降采样或只取前4个元素，也不是缩小安全参数。

本版支持：显式flatten、向量加减乘/负号、square、power2/4、Linear、
由这些操作组成的MLP、fan-out和residual。中间Linear宽度最多8、最终输出1..4。
输入多维只描述同一个tensor的形状，不隐式加入batch语义。
不支持schema4跨块rotate、Conv/Pool、reshape或高层辅助函数，明确拒绝。
旧schema2/3和旧请求契约保持原样；不是完整DSL或无限尺寸支持。

## 模型、参考和生成接口

描述结构同schema2，但`schema:4`，`input_shape`可为`[16]`、`[3,4]`、`[2,2,2,2]`等。
例如`flatten(x) → Linear(16,8) → square → Linear(8,2)`；权重仍在`constants`中明确给出。
10个可复现模型见 `scripts/baseline/cases/chunked-input-10-manifest.json`。

- `chunked_model.py`：检查原始模型并产生内部schema3物理图；不是Agent。
- `chunked_input_abi.py`：定义版本化`model_input_binding`，检查块顺序、起止位置、补零。
- `run_model_batch.prepare_case`：用原始模型构造PyTorch CPU float64 reference，
  与原图独立Python算术交叉验证。reference不使用分块图或候选DSL的输出。
- `arrays.npz` 同时保留原始`logical_inputs`、真实传给密态执行的`inputs`和固定reference。
- FX转换器处理物理图；Agent请求保留原始schema4模型，附带绑定及公开常量。
- 新请求 `hecate-chunked-input-synthesis-v1` 使用已验证的标量运算语法；
  不能默默混入其他实验性native/构造语法标志。后续组合需要独立验证。

原始Linear的数学关系为：

`output[row] = sum(block_dot(input_chunk[k], weight_row_chunk[k])) + bias[row]`

各块dot都包含真实跨槽归约，之后跨密文求和，bias只加一次。
尾部补零列权重为0；全零权重行走已有可信客户端加密零接口，
不使用透明密文或解密回填。

## 确定性密态基线：已通过

`/home/lhy/poseidon-work/results/chunked-model-goldens-cs9wkfzq/report.json`

SHA256 `cad12cea82cb2af52df21c8ae14950d3f6c1d1de010a5b4e1fb81d4016b2ef8a`。

10/10规则生成程序通过：6种输入形状的Linear，16→8→2多项式MLP，
8元素fan-out、12元素residual、15元素含全零权重行的Linear。
另2个错误输入块反例在真实密态执行后均被数值比较拒绝。
正确程序40组输入、80个输出值；最大绝对误差`2.7587182538368893e-10`，
加权MAE `5.474099271205075e-11`。反例最大误差约0.125和0.1544。

真实链路为Hecate→Dacapo→HEVM/CST→upstream SEAL HEVM CPU。
显式使用已注册的`seal-cpu-eva-w45-v1`，不改变默认waterline40；
N32768、14×60bit、tc128，门限固定`1e-5 + 1e-4*abs(reference)`。
不含GPU或bootstrap。规则基线不计为真实Agent覆盖。
12次运行按原保留策略清理8,147,760,120字节可再生密钥，保留全部其他证据。

复现（无API）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 2300s python3 scripts/baseline/run_chunked_model_goldens.py
```

独立审计：`test_chunked_model_evidence.py`；逐一验证原模型、冻结哈希、
原始输入到物理输入的精确对应、请求/源码、HEVM参数、解密数组及错误反例。

## 真实Agent批次

首批 `/home/lhy/poseidon-work/results/agent-batch-uw6qdchz` 未收到完整候选。
实际worker忽略代理环境变量，默认透明路径的TLS连接停滞；最初curl响应来自
不同代理路径，不能用来证明API worker可达。
凭据为空的三轮配对探针确认：默认路径3/3在TLS connect阶段超时，
显式6478路径3/3完成证书验证并得到预期401；host Python及实际Nix Python结果一致。
证据：`deepseek-tls-probe-jyippwp9`、`deepseek-tls-probe-wu3wgtwv`。

通过PID、父进程和case路径核实后，仅终止本批10个停滞的HTTP子进程；
候选及批次父进程正常保存终态并清理全部密钥。
旧报告保留`transport_worker_failed`，这是诊断后主动停止的表现，
不是编译/密态/DSL数值失败。共12次HTTP尝试（含2次连接失败重试），
无完整候选、无usage；不能据此断言服务商没有计费。

新批 `/home/lhy/poseidon-work/results/agent-batch-zea03xuw` 通过`--failed-from`
和`--allow-config-change`链接旧报告，仅切换`--loopback-proxy-port 6478`。
没有覆盖旧失败、没有重跑成功案例（旧批无成功案例）。最终结果以终态报告及
`test_chunked_paid_evidence.py`审计为准；短探针不保证长响应永不失败。
官方DeepSeek `deepseek-flash/high`，API10/密态2，1200秒、384000输出token，
每例最多3轮修复、每次生成最多3次网络重试；10例最多40次生成/160次HTTP。
只发送合成公开模型、常量、layout/绑定、规则与受限反馈，不发送参考输出、
测试输入数组、规则答案、凭据或密钥。上限按新批计，旧批12次HTTP单独保留在谱系中。

### 终态结果与修复

新批最终10/10通过，首次7/10；14次API、0次网络重试，
49,392 prompt + 144,340 completion = **193,732 reported tokens**。
耗时252.48秒；40组成功输入、80个成功输出，另MLP错误候选执行4组，
故整个新批合计44组密态执行。最大绝对误差`5.362776056561103e-10`，
加权MAE `8.423602343540954e-11`，数值门限未改变。
新批report SHA256：`2a5c01df5e3ee59ee9da4a0eedb181fc78ae2fcda41d68b0cc608c2a1b0ed46a`。

失败分层明确保留：

- `[2,4]` Linear和`[3,4]` residual首次候选在tracing失败。
  原因是本次接入遗漏：规则允许公开常量左乘密文，但新任务没有进入
  Hecate Plain常量绑定分支，NumPy把运算变成object array，随后没有`.rotate`。
  这不是已证明的Agent数学错误。Agent各改写一次后通过。
- `Linear(16,8) → square → Linear(8,2)`首次296操作，超过既有256操作上限；
  第二次编译、密态执行成功，但错误交换第二输出行的`c49=-0.125`和`c50=0.25`，
  数值误差约0.30747。第三次由Agent交换回正确位置后通过。
  没有放宽操作数上限、容差、参考或安全参数。

批次结束后在`candidate_trace.py`补上新任务的Plain绑定，然后**原样重放**
那两个最初有效候选：请求和API响应字节均未修改，2/2真实密态通过，
最大绝对误差`5.073707010083695e-10`。
重放没有API调用，不能算新的Agent生成或把首次成功率追改为90%。
证据：`chunked-trace-replays-loh3_99e/report.json`，
SHA256 `22590899b7d0f3582840cad4f302874170ccb8763c77b29d5d6d23e9e889d378`。

加上旧路径批次12次，整个双路由谱系共26次HTTP尝试；193,732 tokens只包含
新批有usage的14次，旧12次缺少usage，货币费用未知。
这两个路由批次不是一个同配置的一次成功率实验；完整保留旧失败的分母与类别。

最终100项定向回归全部通过，无跳过：原始模型与物理图、各位置基向量、
provider边界、旧请求兼容、历史11项Agent证据、schema4真实Agent证据、
原始失败候选修复重放、实际HEVM配置及独立解密比较。
所有新批均已终态；规则批/默认失败批/6478付费批/两例重放共清理
23,085,320,340字节（约21.50GiB）可再生密钥，保留其他实验与错误证据。

再次启动付费测试应明确使用已验证的命令级`--loopback-proxy-port 6478`，
不要从curl读取的代理环境自动推断API worker路由；也不保证此路径永久可用。

## 后续仍需要完成

更大的逻辑输入、更多输出元素/中间宽度、跨块空间算子、更多shape与批次语义；
分块输入与已扩展native构造的组合；上游高层helper及真实bootstrap等剩余DSL语义。
这些是未完成范围，不因本版有限测试通过就算完成。
