# Poseidon / DSL Agent development

## 当前主线：pt_agent（自 2026-10-20 起，优先于下文旧流程）

2026-10-20 起只有一个工作目录：/home/lhohy/Code/Poseidon（分支 lhy-agent-dsl）。原 Poseidon-generalization 和
Poseidon-resnet20 的内容已合并到这里，下文关于 lhy-generalization / lhy-resnet20 独立工作目录的规定不再适用。

开发 DSL Agent 时，只读两样东西：用户在对话中指定的那一份
DSL/DSL_agent/docs/pt_agent/specs/Rn-*.md，以及 DSL/DSL_agent/docs/pt_agent/STATUS.md。
不读 TASK_STATE.md、PARALLEL_DEVELOPMENT.md、DELIVERY_CONTRACT_CURRENT.json 和 docs/generic/r*.md，
这些文件保持原样，不改也不续写。PLAN.md 是路线图，不按它直接实现。

角色：只实现当前这份 spec，按里面写的文件、签名、算法和测试执行，不做设计决策，也不自行换方案。
spec 没写的一律不做；有疑问或报错，就停下来写进 STATUS.md 的"阻塞"一栏，等待新的 spec。
例外（自 2026-10-21）：本轮你自己新写的代码或测试，如果出现实现层面的错误（笔误、漏写参数或必填属性、
导入错误、变量名错误等），可以直接修正后继续，并在汇报的"与 spec 不一致之处"中逐条列出。修正时不得改动
spec 规定的签名、算法、阈值、期望值和断言含义，也不得修改之前轮次的测试。以下情况仍然必须停下：
spec 本身有矛盾；期望值与实测不符；需要做设计选择；修正需要改动 spec 未列出的文件。
兼容适配例外（自 2026-10-22）：spec 修改接口、新增必填字段或升级版本号时，旧测试和实验脚本中只为跟上这些变化而做的
机械修改可以直接做：同步版本字符串；给夹具或假对象补上新增字段的中性值；把写死的版本号或常量改为引用定义处。
做了就在汇报的"与 spec 不一致之处"中逐条列出（文件:行号、改了什么）。改变断言含义、期望数值或算法的修改，仍然必须停下。

范围：
- 新代码只放 DSL/DSL_agent/scripts/pt_agent/，测试放 pt_agent/tests/；文档只写 STATUS.md。
- scripts/baseline/ 下的现有模块只 import 复用，不修改；不往 baseline/benchmarks/ 复制快照。
- third_party/、src/、ckks-runtime、Dacapo 一律不改。B 线的 Dacapo 13 文件补丁只以
  DSL/DSL_agent/validation/dacapo-13file.patch 保存，未经批准不应用到子模块。

对 pt_agent 生效的新规则（替换下文对应的旧规则）：
- Agent 冻结前可以使用内部工具循环（parse/typecheck、MockExpr 明文仿真、CountExpr 代价）。
  真实编译、CKKS 和测试集结果永不回传给 Agent。
  数据划分（用户 2026-10-22 决定，同日修订）：CIFAR 测试集第 9000–9999 张为"留出验证集"，供 Agent 和精调在循环中使用；
  第 8000–8999 张为"确认集"，只在冻结时对候选做一次确认，其结果不回传给 Agent；
  第 0–7999 张为"报告测试集"，只用于最终报告，其结果永不回传、不用于任何选择。M1 第 5 条使用的 40 张加密测试图
  （索引 0–67）属于报告测试集。训练集中的图像只用于校准（calibration）和训练，不再作为可行性验证集。
- DeepSeek 调用总预算 300 次。每次调用在 $PT_RESULTS/registry.jsonl 记一行，预算用完就停。
- 可信 lowering 生成的 Hecate 代码可以在 hecate 前端里 trace；LLM 输出只解析，永不执行。
- 不写 rNNNN 文档、handoff 文件、批次冻结台账和 SHA 绑定。结果只写 $PT_RESULTS/registry.jsonl
  （字段：run_id, round, model, mode, git_rev, metrics, artifacts_dir, ts）和各次运行的目录。
  PT_RESULTS=/home/lhohy/pt_agent_results，在仓库之外。
- 过程记录不进 git：baseline/benchmarks/、docs/whole_models/、docs/coordination/ 已写入 .gitignore。
- 测试只跑本轮的：cd DSL/DSL_agent/scripts && python3 -m unittest discover -s pt_agent/tests -p 'test_r<n>_*.py'
- 本机内存只有 7 GB：单进程 ≤4 GiB，编译 1 个并发。超出就打包成 server_jobs/<name>/run.sh，不在本机硬跑。
  例外（用户 2026-10-20 批准）：pt_agent 的 EmitBackend 产出的程序可以用 toolchain 的可信运行器（R5d），
  不进沙箱，限制为内存 6 GiB、单文件 8 GiB、墙钟 1 小时、磁盘剩余 ≥10 GiB；LLM 输出照旧永不执行。
- 封存集模型只能在 tag pt-agent-freeze-v1 之后运行。
- 外层（系统 python3）负责读写 $PT_RESULTS 和 registry；内层（Nix --pure 环境）不读取 PT_RESULTS、registry 和 config.load，
  需要的路径一律由外层作为命令行参数传入。

继续有效的旧规则：不改库、不用 SEAL 回退、不用明文模拟替代 CKKS、不做中间解密再加密；
容差（数学 1e-12+1e-12|ref|，CKKS 1e-5+1e-4|ref|）不放宽；凭证只经现有加载器读取，不打印；
不碰网络配置；安装、下载、新权重、commit、push、PR 都要单独批准。

汇报：本轮完成或卡住时，按 STATUS.md 模板原地更新，然后在对话里回复"改动文件 + 测试输出 + 与 spec 不一致之处 + 未完成项"。

## Checkout and branch routing

The integration checkout is /home/lhohy/Code/Poseidon in ARM Ubuntu, on lhy-agent-dsl.
The retained Mac copy is historical and must not be developed or synchronized over Ubuntu.
Codex SSH project tasks execute in Ubuntu directly; verify pwd, uname, Git branch and running jobs first.

poseidon is upstream main. DSL is the future shared team branch and must not be created by this task.
Confirmed feature branch names: lhy-generalization and lhy-resnet20. They require separate Ubuntu working directories and the same committed integration baseline before concurrent development. Do not switch branches in an active shared checkout.

Agent source and tests: DSL/DSL_agent/scripts.
Agent documentation and handoff: DSL/DSL_agent/docs.
Other members' DSL packages: DSL/DSL_benchmark; preserve their original bytes.
Old scripts and docs/baseline paths are compatibility entry points.

Read DSL/DSL_agent/docs/coordination/TASK_STATE.md and PARALLEL_DEVELOPMENT.md before continuing.
A owns generic component/operator/layout work; B owns whole-model and backend certification workloads. Public interfaces and shared files follow the ownership matrix.

## Required boundaries

Preserve all existing changes and historical responses, source hashes, reports and program packages.
Do not modify Poseidon, CKKS Runtime or Dacapo libraries to solve Agent issues without a concrete separately approved proposal. Upstream integration is recorded separately from local backend implementation.
No SEAL fallback, plaintext simulation, intermediate decrypt/reencrypt bootstrap, reference changes or tolerance relaxation.
CKKS: 1e-5 + 1e-4*abs(reference). Independent mathematical difference: 1e-12 + 1e-12*abs(reference).
Never alter networking, proxy, DNS, routes, NAT, bridging or VM network configuration.
Credentials remain at the project root .env; only the approved Provider loader reads them. Do not print, transfer or commit credentials.

Plan tests already have user approval. Freeze each batch and enforce its finite budget; do not re-ask for unchanged test authorization.
Global evidence 48 GiB; reserve disk 4 GiB; per case/batch 4 GiB; compile total 2, link 1; API and encrypted execution initially serial across both branches. Preserve sandbox, network isolation, hashes and native limits.
Installations, downloads, new data/weights, backend/security parameter changes and resource expansion still require separate approval.
Local commits, pushes and PRs each require explicit approval. Do not follow older automatic commit/push guidance.
No blind stash pop, reset --hard, checkout --, clean or loss of unrelated files. Snapshot and retained stash are recovery assets.

Goals are thread-local. Start a new feature Goal only after its branch/work directory and common baseline are ready. Do not mark blocked or unverified milestones complete.

以最小范围、最少改动完成任务。从上下文推断意图，能继续做的事不要停下来问。把批准留在最后一步：交付可审查的结果，而非中途请求许可。

禁止字节级/哈希/SHA 校验，除非我明确要求完整性验证。禁止“门限/阈值”作为修辞或模糊限定。禁止“不是A而是B”句式和“不能/不等于”边界限定。禁用 delve、leverage、foster、it's worth noting、importantly、genuinely、“Bottom Line:”“In short:”“Question? Answer.”、自造连字符复合词。用平实主动语态，散文优先，列表只在信息真正并列时使用。

小改动不跑完整测试套件。验证到“足够支持结论”即停，失败或阻塞时说明证据与剩余项，不修复无关的既有失败。
