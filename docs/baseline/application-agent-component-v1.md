# 面向应用的模型到 DSL 组件 v1

## 定位与本轮边界

本组件在模型转换和部署准备阶段运行。宿主应用提交模型、目标后端、验收等级和硬预算，组件产生已验收 DSL 源码包或明确终态。Agent 不进入每次密态推理调用路径。
本轮没有建设 Web 服务、客户密文推理 API、线程安全 SDK、多 Agent 系统或新 Provider，也没有修复此前 26 项深多项式编译阻塞。

## 审查发现与最小改造

| 原实现差距 | 本轮处理 | 保留边界 |
|---|---|---|
| failure 有 retry_policy，但生成循环不消费它 | 在原 run_feedback_loop 增加可选控制回调；新应用入口按受控策略继续或终止 | 旧 CLI 不注入新控制器，原行为保留 |
| 编译/数值失败被统一描述，容易误解成编译器 bug | layer/code/owner/retry_policy/location/context；编译仍归 candidate_or_backend | 未可靠定位的位置为 null，不猜测行号 |
| 没有持久化任务句柄、取消和崩溃处理 | Application.submit/status/run/result/cancel/recover，任务锁、原子状态和调用预留账本 | 独立 Linux worker、主线程执行；不是线程安全 SDK |
| 相同请求再次生成和验收 | 验证级别、完整请求与依赖身份匹配后复用程序包，并重新校验完整性 | 身份漂移或包被修改时拒绝复用；不隐式跨平台转换 |
| 只有数值级成功 | 新增显式 compiled 等级与 v3 程序包；numerical 保持默认及原标准 | compiled 不声明密态或数值通过 |
| 上下文包含与当前图无关的附加指导 | application_policy 从图算子/构造需求选择现有哈希版本上下文；普通图不加额外 guidance，Chebyshev 选 explicit-v9，定向构造选 explicit-v8 | 不删除必需契约规则；未进行新的付费提示策略评测 |
| 应用直接接触实验路径/报告 | 应用返回任务 ID、结构化结果和程序包绑定；完整报告留在私有任务目录 | 后端仍通过 component_backend 接入旧单例运行器，WORK/Nix 暂未完全参数化 |

核心复用 component_contract.prepare_task/synthesize、run_feedback_loop、DeepSeekProvider、component_backend.qualify、CandidateValidationAdapter、统一审计与 candidate_bundle。
正常请求不读取 Benchmark 索引、批次编号、旧实验报告或人工探针。模型名称/family 不参与能力判断。
benchmark_graph 模块目前承载通用图 schema/shape 检查，并非读取 Benchmark 模型集；后续可单独迁移模块名，不应为重命名改变旧请求哈希。

## 宿主接口

在已配置的 Linux Nix/Python worker 内，把仓库 scripts/baseline 放入 Python 模块搜索路径：

    from application_component import Application
    from application_backend import SealBackend

    app = Application(host_owned_task_store, SealBackend())
    task_id = app.submit(
        model,
        backend="upstream_SEAL_HEVM_CPU",
        validation_level="numerical",
        options={},
        limits={
            "generations": 4,
            "http_attempts": 16,
            "wall_seconds": 3600,
            "api_timeout_seconds": 1200,
            "max_tokens": 384000,
        },
    )
    queued = app.status(task_id)
    # 宿主在一个独立 worker 的主线程调度执行：
    outcome = app.run(task_id, trusted_provider)
    outcome = app.result(task_id)
    # 另一个宿主进程可以查询或请求取消：
    app.cancel(task_id)

submit/status 不调用模型；run 是同步 worker 方法，宿主负责异步排队/进程调度。不要在 Web 请求线程里直接运行。
trusted_provider 由可信宿主构造，复用已配置 DeepSeekProvider/HTTPSTransport；已有 request/config 审批校验仍生效。组件不加载凭据、不创建网络传输，不因调用 run 自动授予付费权限。
无新规划调用、诊断模型、裁判模型、多候选搜索。生成与修复使用同一个 Provider 会话。
原 scripts/agent.py 与 scripts/agent_component.py v1 协议保留；旧 schema 1–5 继续经旧 CLI，应用 v1 接收 poseidon-model-graph-v1，不声称本轮统一了全部旧输入格式。

options 沿用 component_contract 的编译配置、helper、构造、packing 选项。能力描述 application_policy.describe() 直接读取现有 OPS、bounds 和契约清单；实际准入仍由现有 validator 判断，不维护模型模板白名单。
声明可准入不等于已经证明任意组合可编译/数值正确。

## 状态与返回

正常阶段：queued → running（preflight/generating/validating/repairing/packaging）→ 终态。

| 终态 | 含义 |
|---|---|
| succeeded | 已取得指定等级的程序包 |
| unsupported | 输入、目标后端或契约能力检查拒绝；不代表一般数学不可表达 |
| budget_exhausted | 生成或 HTTP 额度耗尽 |
| validation_failed | 验证拒绝、重复候选/重复失败或不适合自动修复 |
| service_error | 环境、Provider、完整性、交付或丢失 worker 等异常 |
| cancelled | 宿主请求取消 |
| timed_out | 执行时间预算到期 |

status/result 返回 task_id、status、phase、目标 validation_level、调用预留计数、结构化 failure、acceptance 和 package。只有 succeeded 且 acceptance 匹配时才能当作验收成功。
failure 不包含完整日志、环境路径、测试数组或 reference；位置无法可靠取得时为 null。完整验证报告和原始响应单独保存在宿主私有目录，不进入公开返回。
同一 layer/code/位置/必要上下文连续重复两次即停止；相同候选源码第二次出现不再编译。不会无限探索、自动增加额度或等人工分析后继续原请求。
编译/产物/数值失败可在额度内尝试修复，但不会自动宣布编译器 bug 或模型不可表达。review/never_automatic 路由直接返回终态，不创建人工等待工作流。

## 预算、传输、取消与恢复

Application 的 Controller 统一预留生成次数和每次 HTTP 尝试。DeepSeek 原适配器决定哪些传输错误允许重试，但每次真正尝试前必须获得 Controller 的预留；不能因 Provider 内部重试越过总额度。
单次 API 超时、tokens、Provider 会话额度必须不超过任务声明；最多 4 次生成、每次已有重试上限保持 3。现有 native 槽锁仍控制密态并发不超过 2。
账本预留发生在潜在网络调用之前；“已预留”不等于服务商已接收或实际计费。中断时保留未知结果，不承诺 exactly-once 或明确货币上限。
运行时间从 worker 的执行控制阶段计算；排队等待不计入 wall_seconds。主线程定时器对阻塞 Provider/后端调用检查超时和取消；原子提交结果之前再次检查取消。
cancel 通过持久化标记让运行 worker 退出。recover 在取得任务锁后把遗留 running 任务标记为 interrupted_outcome_unknown 的 service_error，保留原账本，绝不自动补发不确定调用。
恢复查询不等于断点续发。SIGKILL/断电后需要宿主进行恢复判定和既有密钥清理；本轮不保证完整进程树在任意故障下即时回收。

## 两种验收与程序包

compiled：静态检查、真实 Hecate tracing、编译、适用 HEVM/CST 检查及独立审计；不执行测试密文，不解密，不声明数值正确。
numerical：在上述基础上，四组约定输入真实 SEAL CPU 密态执行、解密，与独立 reference 逐项比较。冻结门限仍为 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。
数值级只是有限输入上的证据，不是形式化等价。旧入口默认仍为 numerical；不能用 compiled 满足数值级请求。

程序包继续包含 candidate.py、model.json、request.json、manifest.json；公开权重、布局和要求在 request 中，可信编译入口沿用 scripts/dsl_bundle.py。
编译级使用明确标记的 poseidon-dsl-bundle-v3；数值级保留现有 v2。依赖/编译配置身份和验收等级均参与核验。
package 返回来源 task_id 和 binding；目录为宿主 task_store / 来源 task_id / program。复用时来源可能是较早任务，reused=true。
manifest 本身不是外部签名，宿主必须保留返回的可信绑定。程序包是可重编译的 DSL 源码交付，不是已经实现的常驻部署推理服务。

重放仍使用现有入口：

    python3 -B scripts/dsl_bundle.py replay --bundle /absolute/program
    python3 -B scripts/dsl_bundle.py replay --bundle /absolute/program --execute

第一条只查完整性；第二条按包内明确等级重新验收，compiled 仍不做数值执行。已有 v1/v2 包的数值默认不变。
本轮用 scripted_replay 证明应用流程与真实后端，没有付费 Agent 生成。旧 bundle 的 agent_generated 严格来源判定未放宽；应用调用账本与程序验收分开记录，尚未完成新的 live 应用链路验收。

## 开发工具隔离与后续

benchmarks/tools、人工最小复现、覆盖统计、分片补跑和提示对比是开发工具。应用入口不导入这些脚本，也不会自动修改编译器、SDK、沙箱或系统依赖。
本轮最小边界已形成；后续优先把 component_backend 下面的 CLI 参数适配提取为独立 QualificationSession、注入完整 RunContext，并增加全局宿主队列配额与更多模型的接口回归。不要在做这些改动时重新搭建另一个 Agent。
跨进程作业调度、完整线程安全、外部服务鉴权、推理部署、所有 DSL 分区以及剩余深多项式失败仍是独立范围。
