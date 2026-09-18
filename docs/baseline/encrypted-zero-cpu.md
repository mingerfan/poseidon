# 自由权重边界：显式加密零输入

## 结论与用途

当前已通过真实 Hecate → Dacapo → HEVM/CST → SEAL CPU 的六个正确人工程序、
三个错误程序，以及七个模型的规则转换器对照。它解决的是用户自定义模型中的
全零 Linear 权重行、零卷积核和公开零乘数，不是新增 GPU 后端。

模型仍由用户提供静态图和公开权重；本轮不增加一个“必须选择的模型类别”。
这些 JSON 是回归测试，不是允许模型的枚举表。

## 为什么不能简单生成 x * 0

当前 SEAL 构建开启 `SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT`。密文乘全零明文会
产生透明密文并被拒绝；不能为跑通测试关闭这一检查，也不能用 `x-x` 来规避。

新接口由可信客户端使用现有公钥加密流程生成一个零密文，将它作为额外物理参数。
用户模型的数学输入、权重、明文 reference 和保存的输入数组均不改变：

```python
@hc.func("c,c")
def golden(x, zero_ct):
    return zero_ct
```

这是全零 Linear 的人工 golden。`x` 是模型输入，`zero_ct` 不是第二个模型输入，
而是客户端提供的辅助密文。该设计不需要解密中间结果，没有解密再加密 bootstrap。
四个逻辑输入时，物理签名为 `x, y, z, t, zero_ct`，共五个密文参数。

## 分层职责和不变量

- 模型描述：schema 2/3 不变，1..4 个逻辑输入，每个输入仍是当前约定的四元素布局。
- 规则前端：遇到全零权重行/公开零乘数时声明辅助输入；非零权重仍执行真正点积。
  scalar-neuron 布局中的混合零/非零逐元素系数逐项处理。
- Agent 接口：新的 `hecate-function-synthesis-v5` 请求，对应 `hecate-function-v4`
  AST 合约。旧版本签名和规则不变；模型不能自行重新定义零值、数据来源或位置。
- DSL/编译器：零仍是 ciphertext 类型，可以加偏置、与其他结果合并或继续乘法。
  Earth/CKKS 的 scale/level 调度仍由现有编译器负责。
- 执行器：每组输入重新调用现有 SEAL 公钥加密接口，检查辅助密文非透明、NTT 形式、
  两个多项式，以及实际 level/scale 与编译产物一致。
- 证据：保存四次密文指纹，检查没有复用相同密文。这是防止误复用的观察证据，
  **不是随机数生成器质量或形式化安全性的证明**。

不放宽 `tc128`、N=32768、14 个 60-bit key primes（13 个 data primes），不启用
bootstrap/upscale，不把明文结果作为密态结果。

## 实际证据

| 证据 | 结果 | 保存位置 |
|---|---|---|
| 六个人工 golden + 三个错误程序 | 6/6 通过，3/3 在数值比较阶段被拒绝 | `/home/lhy/poseidon-work/results/zero-golden-batch-w9vuz53i/report.json` |
| 七个规则转换器案例 | 7/7 真密态通过 | `/home/lhy/poseidon-work/results/fx-batch-nmqzon3l/report.json` |
| 混合 scalar-neuron mask 人工 golden | 通过 | `/home/lhy/poseidon-work/results/candidate-replay-_gv4ebdu/report.json` |

第一行报告 SHA256：`28a6332e9db74732819707deafa5acd75edf4b142f1a30c74e2f834b4f84b96a`。
该批次共 36 组输入、72 个输出值；正确程序最大绝对误差为 `1.2493903082596404e-08`。
门限仍为逐项 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
零 reference 的相对误差记录为不适用，不用除零产生的数值掩盖误差。

每个模型都有独立标量公式、图 reference 和 Torch float64 的交叉检查；输入覆盖
零、有符号固定值、固定随机种子和 ±1 边界。错误程序故意将应为零的输出替换为
真实模型输入；它们可以编译并实际运行，但必须在解密比较时失败。

可复现命令（从 WSL 源码根目录运行；均不调用付费 API）：

```bash
timeout -k 3s 2850s python3 scripts/baseline/run_zero_goldens.py
timeout -k 3s 330s python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/zero-scalar-mask.json \
  --golden-file scripts/baseline/golden_cases/encrypted_zero/mask.py --max-repairs 0
```

`test_zero_abi.py` 通过 `POSEIDON_ZERO_GOLDEN_REPORT` 和 `POSEIDON_ZERO_RULE_REPORT`
选择真实证据做审计；环境变量未设置时跳过实际结果检查，不能将 skip 报告成通过。
使用已经固定的 Nix/Python 环境运行这些测试。

## 边界及 Agent 门禁

这里尚不宣称所有代数消去（例如任意 `x-x`）都能自动转换为加密零，也不证明
任何权重幅值和乘法深度都满足给定误差预算。现有形状、公开常量范围和深度约束保留。

人工 golden 和规则转换器不是 Agent 生成成功。Agent 使用单独的七案例 manifest：
`scripts/baseline/cases/encrypted-zero-manifest.json`，结果独立记录，不能把上述
人工程序的成功率加到 Agent 分母。模型近似误差与 CKKS 误差也不合并；此批次没有
替换激活函数，数学模型不变。

临时密钥按每次运行清理，报告、输入、权重、DSL、IR、HEVM/CST 和解密输出保留。
原随机密钥删除后不可恢复，但可重新生成新密钥复现实验；不删除历史实验报告。

## 新 DeepSeek Flash 批次已通过

保存位置：`/home/lhy/poseidon-work/results/agent-batch-gp73w2a1/report.json`。
SHA256：`7a56d87021642885ef317550ab09299854b9f320a5cdd98f12a74816bda85c33`。

- DeepSeek `deepseek-flash`，high，输出上限 384000 tokens，单请求 1200s。
- API 并发上限 10（本批仅七例），本地密态执行并发 2，显式 CONNECT 代理 6478。
- 首次 parse/type/layout/compile/execution/numerical correctness 均为 **7/7**。
- 7 次付费调用，0 语义修复，0 传输重试；28 组输入、60 个输出值。
- 最大绝对误差 `2.221962614565029e-08`，门限未改变。
- 返回 usage 共 58279 tokens：prompt 10904、completion 47375；不据此臆测实际账单金额。
- 七个运行均移除临时密钥，共 4752860070 bytes，未删除实验结果。

Agent 的四输入 golden 自主使用 rotate(1/2/3)，scalar-mask 使用 rotate(-1/-2/-3)，
实际密钥检查和执行均通过。MLP 输出依赖中出现 SSA 别名；它计作一个被观察到的
结构特征，不增加模型族或算子种类，也不是新的数学能力。

无凭据 TLS 探测证据：`/home/lhy/poseidon-work/results/deepseek-tls-probe-a4pme4kv/report.json`。
三次 6478 路径均通过证书/主机名检查并收到预期 401；默认直连三次超时。
短探测和本次七例成功不能证明今后所有长连接永不失败。

规则/人工/Agent/旧队列的独立证据联合测试在固定环境通过；普通 WSL 全量回归为
395 项，313 通过、82 按条件跳过。第一次并行跑全量测试时，Nix 检查争用同一
30s launcher.lock 而失败；付费批次结束后串行重跑通过，没有改依赖或放宽检查。

最终跨队列审计：`/home/lhy/poseidon-work/results/dsl-grammar-audit-zspesvlv/report.json`，
SHA256：`1d1e118ad86c29b78774189cb91c98e996340dad9bf065c3712d7a27936e78ca`。
114 个成功案例、456 组输入、1304 个输出值；结构特征覆盖为 29/32，七个案例声明
辅助加密零。跨队列最大绝对误差 `4.636203883549981e-06`，不是本批七例的最大误差。
不同历史配置的成功证据可以合并做覆盖分析，不能合并成同一配置的首次成功率。
