# 自定义模型清单批量运行（2026-09-11）

## 入口与边界

`run_agent_batch.py --case-manifest <file>` 不再要求选择固定 48/96 例目录。
清单是数据，不是脚本：包含 1..96 个完整 schema-2/3 静态模型描述，
每个模型可直接提供公开权重、输入名称、合法 shape 和连接关系。
模型 id 必须唯一，不用 id 查找预置权重或模型实现。

格式：

```json
{
  "schema": 1,
  "cases": [
    {
      "schema": 2,
      "id": "my-affine",
      "input_shape": [4],
      "constants": {"weight": [0.2, -0.4, 0.1, 0.7], "bias": 0.125},
      "nodes": [
        {"id": "product", "op": "multiply", "inputs": ["x", "weight"]},
        {"id": "out", "op": "add", "inputs": ["product", "bias"]}
      ],
      "output": "out"
    }
  ]
}
```

这扩展的是批量输入接口，不会偷偷扩充底层支持范围；算子、shape、broadcast、
数值上界和资源限制仍由原模型检查器决定。不支持的图在读凭据或发起 API
调用之前拒绝整份清单，不静默删除不支持案例来提高成功率。拒绝重复 JSON
字段、重复 id、非有限数值、额外顶层字段和超过 1 MiB 的文件。

模型数不增加模型类别计数：自定义批次统一记作 custom_graph，不能把六个
自定义 id 当作六个新模型族。

## 只读计划

从 WSL 源码根执行；这个模式不读取 .env、不调用模型、不运行密态计算：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_agent_batch.py --plan \
  --case-manifest scripts/baseline/cases/semantic-gap-manifest.json
```

示例清单明确包含三输入、四输入、减法/取负/Linear 组合和三个负旋转图。
它们有已有的独立 reference、人工 golden、静态门禁和 CPU 密态验证基础。
Agent 可以选择语义等价的表达式；例如 rotate(-1) 可能由周期四下的 rotate(3)
实现。因此数值通过与观察到某个原生语法必须分开报告，不能强迫一种无必要
的语法以凑齐覆盖数字。

## 显式付费运行

以下命令会向 DeepSeek 发送公开模型结构与固定公开权重，产生 API 费用；
不发送测试输入、reference 答案或密钥。只有显式 --live 才调用服务。

```bash
timeout -k 5s 68000s python3 scripts/baseline/run_agent_batch.py --live \
  --case-manifest scripts/baseline/cases/semantic-gap-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 1200 --provider-retries 3 \
  --loopback-proxy-port 6478 --jobs 10
```

6478 是用户本轮确认的代理端口，不是永久自动发现值；本轮命令使用
127.0.0.1 CONNECT，并保留目标服务 TLS 证书及主机名校验，不修改全局代理。
原 7897 连接失败、默认路线 TLS 建连超时，而 6478 在普通 WSL Python 及
Nix Python 中都完成了无凭据 /models 请求（预期返回 401）。
短请求通过不保证所有长生成传输成功。TLS/证书完整性错误仍不自动重试。

参数仍有硬上限，不承诺无限预算或确保生成成功。每个模型最多一次初始
候选加三次反馈修复；单次生成中符合策略的传输失败最多三次重试。
两种重试独立计数。API/候选工作进程并发允许 1..10，默认 10；不足十例时
不会人为补齐调用。密钥生成、sandbox probe、tracing、编译和 SEAL 执行
通过 Linux flock 共用两个本地执行名额，API 等待不占名额。等待本地名额
最多 600 秒，超时记录基础设施失败，不无限等待。锁文件在 WSL 原生 cache
目录；内核在进程退出后释放锁，不用删除锁文件来解除占用。

批次报告分别记录 api_concurrency、native_execution_concurrency；单例记录
名额等待时间和获取次数。原有并发字段仍保留以兼容历史审计。
这些数字是并发上限，不是实际同时在线请求数，也不保证服务端不限流。

新默认服务为 DeepSeek、模型 deepseek-flash（V4.1-Flash）；官方公告指出旧
deepseek-v4-flash 已退役并暂时路由到新版本：
[DeepSeek 2026-09-10 公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)。
模型身份检查仍严格执行，不将新旧版本混为同一个实验配置。

## 冻结与结果审计

外层先验证清单、计算 SHA256，再将该摘要传给隔离入口重新核对。
执行前在批次目录写入精确字节快照 custom-manifest.json；report.json 记录
摘要和原始模型数量。工作进程从内存中已验证的模型描述生成独立 model.json，
不再在每次调用前从用户文件重新读入可变模型。

保留已有独立候选进程、四组输入、真实 Dacapo/SEAL CPU 执行、固定误差门限、
输入/权重/reference 不可变、模型反馈修复及密钥自动清理策略。
不添加 GPU 验收，不回退为规则答案或模拟执行。

`audit_agent_lineage.py` 现在能从哈希校验的自定义清单确定审计分母。
它仍验证父报告哈希、失败子集选择、模型描述和结果，不假设所有批次都是
96 例。旧 96 例完整链路审计回归保持通过。

失败重跑需要传入原清单和 --failed-from <完成报告>；只重跑失败项。
清单或权重不能悄悄改为另一组，再沿用原批次的成功结果。
--extended 与 --case-manifest 互斥；旧预置入口不变。

## 验证记录

- 新增 10 项测试：数据自由度、重复/非法图、文件变更、离线计划、
  在凭据加载前拒绝、参数冲突和自定义失败子批次审计。
- WSL 全量回归：360 项，287 通过，73 条件跳过，无失败。
- 固定 Nix 依赖环境：新测试与旧 96 例真实结果审计联合 16/16 通过。
- 初始测试夹具把 rows 方法覆盖成了列表；只修复测试夹具命名，
  未修改生产校验规则。
- 自动权限审核最初超时，随后重试成功；那不是 TLS 服务端错误。

本轮在线批次：/home/lhy/poseidon-work/results/agent-batch-8mofphci/report.json。
该旧模型批次终态为 completed_with_failures：六次 API 返回 HTTP 200，
但均被 response_model_mismatch 拒绝，未到候选编译或密态执行。
旧记录缺少可核实 usage，不能把它记成零费用；报告保留原样。
新版本补跑用同一清单、--failed-from 和 --allow-config-change 明确记录
模型/执行并发变化，不覆盖旧报告。**启动不等于通过**；以终态和后续
数值/源码覆盖审计为准。

## DeepSeek Flash 补跑结果（2026-09-11）

用户批准模型改为 deepseek-flash、API 并发上限 10。补跑报告：
`/home/lhy/poseidon-work/results/agent-batch-k1gloaa6/report.json`。
父报告保持原样；selection 记录模型从 deepseek-v4-flash 切换，以及 API
并发从 2 到 10、本地执行限流新设为 2。六例不是十例压力测试。

本批终态 passed，约 57.29 秒：6/6 首次生成、静态检查、编译、密态执行
及数值比较通过，无反馈修复，无传输重试，无失踪 usage。真实服务调用 6 次，
prompt tokens 6239、completion tokens 13136、total tokens 19375。
token 数包含服务所报告的相应使用量，不等于美元费用；未估算账单。

| 案例 | MAE | 最大绝对误差 | 最大非零 reference 相对误差 |
| --- | ---: | ---: | ---: |
| custom-triple-merge | 1.02730e-8 | 2.09274e-8 | 1.88305e-6 |
| custom-quad-merge | 4.35297e-9 | 1.48430e-8 | 5.27637e-8 |
| user-subneg-linear | 3.47132e-9 | 1.07601e-8 | 4.03499e-8 |
| rotate-minus1 | 1.39974e-6 | 4.49627e-6 | 1.12732e-5 |
| rotate-minus2 | 1.40196e-6 | 3.41146e-6 | 1.36458e-5 |
| rotate-minus3 | 1.35306e-6 | 4.63620e-6 | 3.76076e-5 |

每例四组输入，共 24 组、88 个解密输出。冻结门限仍为
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`；未改安全参数，
未用模拟 bootstrap。仅验证现有 Dacapo/SEAL CPU 路线，不声明 Poseidon
GPU 执行。密钥清理日志合计删除可再生临时密钥 3692183098 字节（约 3.44 GiB），
审计确认零 private-keys 目录保留；模型、候选、编译产物和解密证据保留。

独立重算与哈希审计：

- 数值：`/home/lhy/poseidon-work/results/agent-lineage-audit-fkqrrraf/report.json`，6/6。
- DSL：`/home/lhy/poseidon-work/results/dsl-grammar-audit-kv4phsf9/report.json`。
- 本批观察到 20/32 个受限 DSL 结构分区。与旧 96 例审计
  `dsl-grammar-audit-gugzv80z/report.json` 的集合并集为 26/32；新覆盖
  三/四输入、密文取负、rotate(-1/-2/-3)。这只是不同配置的覆盖证据并集，
  不是同配置 102 例成功率，也不是整个上游 Hecate DSL 的形式化正确性证明。
- 仍未观察到的六项：statement.alias、add.length1、multiply.length1、
  subtract.scalar、subtract.length1、subtract.length4。人工 golden 证据
  与 Agent 覆盖分开，等价表达式不强制改写以凑覆盖数字。

新回归：WSL 369 项，296 通过、73 条件跳过、零失败；固定 Nix 环境下
并发/模型身份/自定义清单/旧 96 例审计联合 25/25 通过。十进程离线测试
确认本地重计算阶段最多两并发；锁超时、异常释放、符号链接拒绝均通过。
新增测试最初误把 response_content 的返回元组当作字符串，修正断言取
内容字段后通过，未放宽生产校验。
