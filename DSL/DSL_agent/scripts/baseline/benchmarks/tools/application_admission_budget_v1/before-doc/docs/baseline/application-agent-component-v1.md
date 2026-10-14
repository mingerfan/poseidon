# 面向应用的模型到 DSL 组件 v1

> 2026-09-28 后端更新：当前使用 Poseidon CPU。原组件阶段的 SEAL 报告保留历史身份，当前验收范围见 [迁移说明](poseidon-runtime-migration.md)。

## 定位与本轮边界

本组件在模型转换和部署准备阶段运行。宿主应用提交模型、目标后端、验收等级和硬预算，组件产生已验收 DSL 源码包或明确终态。Agent 不进入每次密态推理调用路径。
本轮没有建设 Web 服务、客户密文推理 API、线程安全 SDK、多 Agent 系统或新 Provider，也没有修复此前 26 项深多项式编译阻塞。

## 审查发现与最小改造

| 原实现差距 | 本轮处理 | 保留边界 |
|---|---|---|
| failure 有 retry_policy，但生成循环不消费它 | 在原 run_feedback_loop 增加可选控制回调；新应用入口按受控策略继续或终止 | 旧 CLI 默认 legacy-v1；显式 --feedback-policy precise-v2 使用与应用相同的决策 |
| 编译/数值失败被统一描述，容易误解成编译器 bug | layer/code/owner/retry_policy/location/context；编译仍归 candidate_or_backend | 未可靠定位的位置为 null，不猜测行号 |
| 没有持久化任务句柄、取消和崩溃处理 | Application.submit/status/run/result/cancel/recover，任务锁、原子状态和调用预留账本 | 独立 Linux worker、主线程执行；不是线程安全 SDK |
| 相同请求再次生成和验收 | 验证级别、完整请求与依赖身份匹配后复用程序包，并重新校验完整性 | 身份漂移或包被修改时拒绝复用；不隐式跨平台转换 |
| 只有数值级成功 | compiled / numerical 两个等级与 Poseidon v4 程序包；numerical 保持默认及原标准 | compiled 不声明密态或数值通过 |
| 上下文包含与当前图无关的附加指导 | application_policy 从图算子/构造需求选择现有哈希版本上下文；归约、广播、公共乘法及多项式等相关图选择 Poseidon explicit-v10；无相关特征的定向构造仍选 explicit-v8；显式版本原样保留 | 不删除必需契约规则；未进行新的付费提示策略评测 |
| 应用直接接触实验路径/报告 | 应用返回任务 ID、结构化结果和程序包绑定；完整报告留在私有任务目录 | 后端通过 QualificationSession 共用原验证适配器，WORK/Nix 仍由已绑定 worker 管理 |

核心复用 component_contract.prepare_task/synthesize、run_feedback_loop、DeepSeekProvider、component_backend.qualify、CandidateValidationAdapter、统一审计与 candidate_bundle。
正常请求不读取 Benchmark 索引、批次编号、旧实验报告或人工探针。模型名称/family 不参与能力判断。
benchmark_graph 模块目前承载通用图 schema/shape 检查，并非读取 Benchmark 模型集；后续可单独迁移模块名，不应为重命名改变旧请求哈希。

## 宿主接口

在已配置的 Linux Nix/Python worker 内，把仓库 scripts/baseline 放入 Python 模块搜索路径：

    from application_component import Application
    from application_backend import PoseidonBackend

    app = Application(host_owned_task_store, PoseidonBackend())
    task_id = app.submit(
        model,
        backend="poseidon_cpu",
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
原 scripts/agent.py 与 scripts/agent_component.py v1 协议保留；旧 schema 1–5 继续经原 CLI，应用接口接收 poseidon-model-graph-v1/v2 及已核验的导入目录。

options 沿用 component_contract 的编译配置、helper、构造、packing 选项。能力描述 application_policy.describe() 直接读取现有 OPS、bounds 和契约清单；实际准入仍由现有 validator 判断，不维护模型模板白名单。
声明可准入不等于已经证明任意组合可编译/数值正确。

## 模型来源提交

`app.submit_source(source, **requirements)` 将模型导入和 DSL 生成放在同一个任务中。宿主继续通过 `status`、`result`、`cancel` 查询和控制任务，由独立 worker 调用 `run`。worker 使用已有的固定 Nix/Python 环境。

| kind | 提交字段 | 接入方式 |
|---|---|---|
| graph | model：JSON 计算图 | 校验已规范化图，或分解高层图 |
| restricted_python | manifest：源码清单路径 | 静态解析列明的源码文件 |
| trusted_pytorch | manifest、trust_local_code: true | 隔离进程捕获明确选择的本地 PyTorch 模型 |

三种来源均使用 `format: "poseidon-model-source-request-v1"`。Python 清单格式和版本沿用[模型导入说明](operator-decomposition-and-python.md)。受限源码使用 legacy-v1；可信 PyTorch 可选择已有资源配置，各算子的形状与属性要求保持。

~~~python
task_id = app.submit_source(
    {
        "format": "poseidon-model-source-request-v1",
        "kind": "trusted_pytorch",
        "manifest": "/absolute/path/model/manifest.json",
        "trust_local_code": True,
    },
    resource_profile="general-tensors-v1",
    validation_level="numerical",
    options={},
)
outcome = app.run(task_id, trusted_provider)
~~~

JSON 来源用 `kind: "graph"` 和 `model` 字段；受限源码用 `kind: "restricted_python"` 和 `manifest` 字段。`resource_profile` 是来源要求，编译、布局和生成选项仍放在 `options` 中。旧 `submit(model)`、`submit_import(directory)` 和 CLI 保留。

提交时冻结列明的源码及 NPY 权重，执行时使用任务快照。来源导入、生成、修复和验证共用任务预算与取消状态；导入耗时从同一 `wall_seconds` 中扣除。新增阶段为 `intake_queued` 和 `model_import`，导入完成后进入原有生成循环。

来源记录使用 `poseidon-model-source-origin-v1`，绑定文件内容、规范化模型、导入参考和控制层身份。程序包 v5/v6 在 `artifacts/model-source.json` 保存该记录；生成来源和后端验收分别记录。旧包没有来源记录时按原规则检查。
复用身份由模型、来源文件、权重、导入参考和控制层版本决定；时间预算与调用额度保存在各自任务快照中。交付仍校验原任务和完整来源绑定。

最新来源接口已完成组件兼容测试和三类来源的实际导入检查。生产组件的当前编译、密态执行及程序包复用范围在后端验收报告中单列。

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
新任务默认使用 `precise-v2`：相同候选源码第二次出现即停止；只有具备稳定位置与已核验约束的同一失败连续出现两次，才按重复失败停止。粗粒度编译或数值诊断继续服从原修复预算。旧任务缺少策略字段时沿用 `legacy-v1`。不会无限探索、自动增加额度或等人工分析后继续原请求。
编译/产物/数值失败可在额度内尝试修复，但不会自动宣布编译器 bug 或模型不可表达。review/never_automatic 路由直接返回终态，不创建人工等待工作流。

## 预算、传输、取消与恢复

Application 的 Controller 统一预留生成次数和每次 HTTP 尝试。DeepSeek 原适配器决定哪些传输错误允许重试，但每次真正尝试前必须获得 Controller 的预留；不能因 Provider 内部重试越过总额度。
单次 API 超时、tokens、Provider 会话额度必须不超过任务声明；最多 4 次生成、每次已有重试上限保持 3。现有 native 槽锁仍控制密态并发不超过 2。
账本预留发生在潜在网络调用之前；“已预留”不等于服务商已接收或实际计费。中断时保留未知结果，不承诺 exactly-once 或明确货币上限。
运行时间从 worker 的执行控制阶段计算；排队等待不计入 wall_seconds。主线程定时器对阻塞 Provider/后端调用检查超时和取消；原子提交结果之前再次检查取消。
cancel 通过持久化标记让运行 worker 退出。recover 在取得任务锁后把遗留 running 任务标记为 interrupted_outcome_unknown 的 service_error，保留原账本，绝不自动补发不确定调用。
恢复查询不等于断点续发。SIGKILL/断电后需要宿主进行恢复判定和既有密钥清理；本轮不保证完整进程树在任意故障下即时回收。

## 两种验收与程序包

compiled：静态检查、真实 Hecate tracing、编译、RuntimePlan、OperatorSpec 和常量 bundle 检查及独立审计；不执行测试密文，不解密，不声明数值正确。
numerical：在上述基础上，四组约定输入真实 Poseidon CPU 密态执行、解密，与独立 reference 逐项比较。冻结门限仍为 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。
数值级只是有限输入上的证据，不是形式化等价。旧入口默认仍为 numerical；不能用 compiled 满足数值级请求。

程序包继续包含 candidate.py、model.json、request.json、manifest.json；公开权重、布局和要求在 request 中，可信编译入口沿用 scripts/dsl_bundle.py。
CLI 原有 Poseidon v4 程序包保持原规则。新应用交付 v5；权重描述符请求交付 v6。manifest 明确标记 compiled 或 numerical，依赖、编译配置、分阶段身份及验收等级参与核验。
package 返回来源 task_id 和 binding；目录为宿主 task_store / 来源 task_id / program。复用时来源可能是较早任务，reused=true。

复用和结果交付核对原任务绑定、完整请求、模型来源、验收等级与依赖身份。缓存索引用于查找，程序包来源必须与当前请求一致。绑定异常返回 `service_error` 和 `cached_*` 错误码，层为 `delivery`，宿主处理。

worker 的取消与超时控制覆盖预检、缓存检查、生成、验证和交付。[实现与回归](agent-cache-delivery-acceptance-r968.md)。
manifest 本身不是外部签名，宿主必须保留返回的可信绑定。程序包包含可复用 RuntimePlan 与 DSL 源码；常驻推理服务是独立模块。

重放仍使用现有入口：

    python3 -B scripts/dsl_bundle.py replay --bundle /absolute/program
    python3 -B scripts/dsl_bundle.py replay --bundle /absolute/program --execute

第一条只查完整性；第二条按包内明确等级直接重载产物验收，compiled 仍不做数值执行。旧 SEAL 包保留完整性检查，执行入口要求新的 Poseidon v4 包。
本轮用 scripted_replay 证明应用流程与真实后端，没有付费 Agent 生成。旧 bundle 的 agent_generated 严格来源判定未放宽；应用调用账本与程序验收分开记录，尚未完成新的 live 应用链路验收。

## 开发工具隔离与后续

benchmarks/tools、人工最小复现、覆盖统计、分片补跑和提示对比是开发工具。应用入口不导入这些脚本，也不会自动修改编译器、SDK、沙箱或系统依赖。
本轮最小边界已形成；当前已提取 QualificationSession，应用与旧 CLI 共用验证适配器。后续继续分离生成来源核验、数据读取和阶段身份，再完善宿主队列与模型接口回归。不要在做这些改动时重新搭建另一个 Agent。
跨进程作业调度、完整线程安全、外部服务鉴权、推理部署、所有 DSL 分区以及剩余深多项式失败仍是独立范围。

## Poseidon 上下文

`explicit-v12` 在原有生成规则上增加逻辑轴、分组归约、形状变换与输入分块的索引关系。它按图算子选择规则，作为 `options={"generation_guidance": "explicit-v12"}` 显式提交；旧版本和默认选择保持。结构化反馈复用相同规则。当前已完成相关控制测试、真实编译、Poseidon CPU 数值比较与程序包复用；新生成质量另行评测。[实现与验收](agent-tensor-axis-guidance-r1332.md)。可选示例按完整公开请求预算裁剪，权重描述符与必需规则保持。[预算修复](agent-tensor-axis-context-budget-r1334.md)。

explicit-v10 按公开图特征准备表达规则，区别于旧 SEAL 契约。规则进入请求哈希及修复反馈，不读取人工修复程序或历史实验报告。90 项相关测试、两种应用验收等级与程序包复用通过；尚无本轮真实 Agent 提示策略评测。详见 [r241](poseidon-application-context-r241.md)。


## 能力预检与执行

`application_capabilities.describe()` 返回版本化的算子、分解、布局、资源和后端约束；`assess_model()` 和 `assess_request()` 返回准入结果、精确资源、保守下界和未知项。预检与编译、密态、Agent 生成各有独立状态。高层算子图先经现有导入入口分解，再提交应用任务。

新任务使用结构化反馈 `poseidon-diagnostic-v2`。公开反馈仅包含登记的错误码、位置、约束与现有受控摘要。无法可靠定位的编译错误按候选或后端问题返回，预算内继续修复。

新应用交付 v5/v6 程序包，分别绑定生成、编译、执行和验证身份。程序包重载核对编译、执行和验证身份。源码分析采用保守的整个模块依赖闭包，共用的请求与契约代码仍可能同时影响多个身份。语法分析缓存按当前内容哈希索引，文件、动态加载授权及实际工具身份每次重新检查。

`app.execute(task_id, samples)` 在已配置的独立 worker 中运行已完成任务的程序包：

~~~python
outputs = app.execute(task_id, [
    {"x": [0.1, -0.2, 0.3, -0.4]},
    {"x": [0.0, 0.25, -0.25, 0.5]},
])
~~~

每组输入为与模型签名一致的名称／数组字典，共 1–16 组。整数按 float64 精确可表示范围检查，布尔数据与非有限数拒绝。执行直接加载 RuntimePlan 与公开常量，返回命名输出、执行记录和原程序包的验收等级。

`status: "executed"` 表示这批新输入已完成密态执行。新输入的独立 reference 比较由调用方另行提供；返回中的 `numerically_validated_on_supplied_inputs` 为 false，原程序包的四组验收范围保存在 `package_validation`。应用精度单独返回认证状态。

执行前根据程序包绑定的原图检查声明的近似区间，使用原图的近似计算语义。输入按原签名验证并复制后再打包；返回的 `input_domain_check` 包含状态、原图哈希和检查数量。状态为 `passed` 或 `not_declared`，只描述区间检查范围。

越界返回 `status: "validation_failed"`，错误码 `approximation_domain_rejected`。公开诊断包含首个违规节点的索引、id、算子、样本序号和声明区间；该输入的加密及执行调用数为零。CLI 返回同一 JSON，退出码为 1。原已完成任务的状态和程序包保持可复用。

区间预检、调用方新输入的数值比较和应用精度分别记录。当前实现与验收范围见 [r706](agent-execution-domain-acceptance-r706.md)。

CLI 输入文件格式为：

~~~json
{"format":"poseidon-local-inputs-v1","samples":[{"x":[0.1,-0.2,0.3,-0.4]}]}
~~~

~~~bash
python3 -B scripts/dsl_bundle.py execute --bundle program --inputs inputs.json
~~~

旧 v1–v4 程序包保留原格式与身份规则。新本地执行入口当前接收统一图的 Poseidon CPU 程序包。宿主任务目录由 Application 构造参数指定，后端工作目录仍使用现有 POSEIDON_WORK_ROOT／平台配置。


## 已导入模型与输出结构

`app.submit_import(directory, **requirements)` 接收离线导入目录，检查模型、原图、分解、参考报告及来源清单，再交给已有生成流程。接口沿用任务状态、预算和取消机制，返回同一类程序包。

也可使用 `app.submit(model, model_provenance=provenance)` 提交已检查的来源数据。来源只保存在可信任务与程序包中，生成模型接收的请求保持不变。缓存键额外绑定来源哈希，来源不同的请求分别交付。

PyTorch v2–v5 的 Tensor、tuple/list、字典结构保存在来源包中。v4／v5 同时支持直接标量；新输入执行保持已有命名数组字段，另返回 `output_structure` 和 `structured_outputs`。计算验收仍针对每个张量的逐项结果。类型策略及清单格式见[参数化输入](operator-decomposition-and-python.md#参数化-pytorch-模型)。

## 权重描述符与生成来源

声明式图权重包导入后可显式提交：

```python
task_id = app.submit_import(
    imported_directory,
    options={"chunk_period": 4, "public_weight_policy": "descriptors-v1"},
)
result = app.run(task_id, trusted_provider)
```

生成模型收到公开图和权重描述符，可信 tracing 使用已检查的原数值。模型中的逻辑权重名与 DSL 已绑定常量名分开；可用名称由 public_constants 指定。描述符形状、原值哈希和常量来源全部参与请求绑定。完整可信数据在任务与 v6 程序包中保留。

真实生成成功后，result.generation 和包内生成记录绑定最终返回源码、原始响应、受控修复反馈、HTTP 尝试及 token 用量。资格验收中的脚本重放保留自己的来源标签。缓存复用保留原生成来源，而本次 generations/http_attempts 为零。

Controller 在潜在 HTTP 发送前预留次数，每个检查点保存脱敏 provider-metrics.json，任务结束保存 loop.json。另存 host-owned `provider-transport.json`，保留白名单中的流错误类别、阶段、计数及重试决定。它与可移植的生成来源记录分开，供宿主诊断服务异常使用。token 用量采用服务商返回的计数；费用未知时为 null，未确认尝试保留未知状态。传输环境诊断和凭据进入不了可移植生成记录。历史缺少 usage 的记录继续核验，缺失值保持未知。

六个参数化 PyTorch 模型和八个随机权重模型已完成真实 Agent 生成、Poseidon CPU 数值验收和程序包重载，14 项通过；来源控制测试另用合成 transport。见[生成验收](agent-generalization-live-acceptance-r626.md)。

当前实现仍用保守的模块依赖闭包，部分导入代码变化会使来源缓存失效。可选布局准备及固定工作／请求上下文已接入；底层路径注入和更细的身份拆分继续推进。


## 通用输入验收

参数化 PyTorch、权重描述符与程序包执行的真实生成验收见 [r626](agent-generalization-live-acceptance-r626.md)。14 项通过，三个原程序包完成新输入及重复执行；生成、编译验收和 caller 提供输入的数值结论分别记录。

## 布局准备

在已配置的独立 worker 主线程中，宿主可先准备布局，再提交原应用组件：

~~~python
from application_planning.layout import plan_model, submit_planned

decision = plan_model(
    model,
    options={"public_weight_policy": "descriptors-v1"},
    limits={"wall_seconds": 120, "candidate_limit": 8},
)
if decision["status"] == "prepared":
    submission = submit_planned(
        app, model, decision,
        validation_level="numerical",
        limits={
            "generations": 4, "http_attempts": 16, "wall_seconds": 3600,
            "api_timeout_seconds": 1200, "max_tokens": 384000,
        },
    )
    task_id = submission["task_id"]
    state = app.status(task_id)
    outcome = app.run(task_id, trusted_provider)
else:
    failure = decision["failure"]
~~~

输入为规范化后的统一图 v1/v2，PyTorch 来源先使用现有导入工具。来源元数据沿用 `model_provenance` 参数。

计划策略为 `bound-layouts-v1`。候选来自现有周期和 packing 注册表；rank、shape、输入输出块数、公开常量及编译配置均经原准入路径检查。按物理输入输出声明数、公开常量条目、元素数及周期排序，记录重复和拒绝项。此排序是确定性初始选择规则，实际延迟与内存需要后端验收。

用户显式周期（含 None）、packing、helper 或定向构造要求保留。只有未固定的选项参与候选选择。默认 120 秒／8 项，硬上限 180 秒／8 项；准备阶段超时返回 `budget_exhausted`，工作线程或已有定时器冲突返回 `service_error`。

`check_plan()` 检查模型、源码闭包、候选清单、选定请求及资源绑定。宿主应保留返回的可信计划绑定；哈希用于完整性核验。计划记录精确的声明数和 float64 公共数据规模，乘法深度、中间密文、峰值内存和执行时间仍标记 unknown。各编译、密态与生成验收状态继续独立报告。

`submit_planned()` 显式调用现有 `Application.submit()`，任务预算、取消、Provider 及程序包交付仍由原控制层负责。选定请求和布局进入原程序包绑定，计划与排序记录由宿主保存。既有 `submit()` 和 CLI 默认行为保持；可选布局准备入口使用 `scripts/plan_model.py`。

[静态及真实后端结果](agent-layout-acceptance-r646.md)

## 固定工作上下文

宿主可在已配置的 Nix/Python worker 中显式创建上下文：

~~~python
from application_execution.context import WorkerContext
from application_execution.backend import application

worker = WorkerContext.capture(host_owned_task_store)
app = application(worker)
task_id = app.submit_import(
    imported_directory,
    options={"chunk_period": 4, "public_weight_policy": "descriptors-v1"},
)
queued = app.status(task_id)
outcome = app.run(task_id, trusted_provider)
outputs = app.execute(task_id, samples)
~~~

这里的 app 仍是原 Application，沿用状态、预算、Provider、取消、修复和程序包接口。直接构造 Application 的旧用法继续保留。

新任务目录采用 `poseidon-worker-context-v2`，策略为 `staged-sdk-v1`，绑定实际平台、Python/NumPy/Torch、SDK、动态库、工作路径及编译／执行／验证身份。Provider 和生成控制模块的更新不改变这三项身份时，原程序包继续可用。

每次验收另行绑定完整请求、布局、编译配置和验收等级。待生成任务仍核对完整生成身份；编译器、运行库、参数、验证规则或环境改变时拒绝复用。

已有 `poseidon-worker-context-v1` 目录继续使用原策略。宿主也可显式调用 `WorkerContext.capture(task_store, identity_policy="legacy-v1")` 创建旧策略；策略升级使用独立目录，原记录保留。

相同请求的缓存匹配从已绑定任务元数据计算，比较模型、权重、来源、验收等级及三项后端身份。历史 `reuse_key` 字段保持原样，交付时继续核验原任务、包绑定和产物完整性。

worker-context.json 和 request-contexts/ 保存在宿主任务目录，文件权限为 0600；记录包含配置身份和哈希。Provider 使用原有受控生成上下文，程序包保持既有 v5/v6 格式和编译、执行、验证身份规则。

当前采用一套 SDK 对应一个 worker，底层运行器仍使用绑定的全局路径，编译与执行证据仍写入已声明的 WORK/results。更换路径、依赖或实现后，使用新的上下文与任务目录；旧记录保留。原程序包继续通过已有加载器核验和执行。

[上下文验收](agent-context-acceptance-r655.md) · [阶段身份与程序包复用](agent-staged-worker-reuse-r1329.md)


## 资源预检

宿主可在选定布局后增加资源检查，再提交同一个 Application：

~~~python
from application_planning.layout import plan_model
from application_planning.resources import assess_plan, submit_planned

decision = plan_model(model, options={"public_weight_policy": "descriptors-v1"})
if decision["status"] == "prepared":
    resources = assess_plan(decision, model, sample_count=4)
    if resources["status"] == "assessed":
        submission = submit_planned(
            app, model, decision, resources,
            validation_level="numerical",
        )
        task_id = submission["task_id"]
    else:
        failure = resources["failure"]
else:
    failure = decision["failure"]
~~~

参数和源码随分析绑定。结果包括精确输入／辅助零／输出声明、公开 float64 数组数据、逻辑图规模和输入密文系数存储下界。当前每个输入密文为 6.5 MiB；输入与可信辅助零在运行前同时存在。输出声明可能共享对象，单独记录。

`memory_budget_bytes` 是预检下界筛选预算，范围受原生沙箱上限约束。它不更改 SDK 的运行限制。`assessed` 表示已完成分析，`execution_readiness` 保持 `unknown`；峰值内存、评估密钥和实际中间值由后续编译／运行验证。明显不足的预算返回 `blocked` 和结构化资源失败，提交封装拒绝进入原 Application。

`check_analysis()` 复核请求、布局、参数、源码及预算，防止改写数值后重新计算哈希。分析身份由宿主保存；旧请求哈希和程序包格式保持。CLI：`scripts/analyze_resources.py`，`--layout-plan` 复用计划，`--memory-mib` 指定预检预算，`--samples` 为 1–16。

[本轮实现与真实后端结果](agent-resource-acceptance-r665.md)

### 编译程序包的资源分析

`application_planning.compiled_resources` 接收已有 Poseidon CPU 程序包，复核依赖、入口、RuntimePlan、OperatorSpec、bundle 和输入输出绑定，再计算当前 Runtime 保留的多项式系数数据及公开槽向量存储下界。CLI 为 `scripts/analyze_program.py`。

~~~python
from application_planning.compiled_resources import analyze_program, require_storage_precheck
from program_execution import execute_program

analysis = analyze_program(bundle)
require_storage_precheck(analysis, bundle)
result = execute_program(bundle, samples)
~~~

报告记录输入与中间密文、明文、输出对象别名、旋转步数、重线性化需求以及输出 level。每组样本依次执行，存储下界按单组计算。当前 Agent 使用的 `ckks-runtime-agent` 在一次执行内保留已定义值；参考目录 `ckks-runtime` 中的释放逻辑不属于该执行身份。

`memory_budget_bytes`（CLI 的 `--memory-mib`）仅筛选存储下界，运行时沿用原生 4 GiB 地址空间限制。下界超预算时返回 `blocked` 和首次越界定义的位置；适配器拒绝执行。`assessed` 只表示分析通过，`execution_readiness` 为 `unknown`。评估密钥大小、分配容量、工作区、峰值内存和耗时仍是未知项。

分析绑定当前源文件、参数、程序包及读取的产物快照；重载时重新计算事实，拒绝陈旧或伪造报告。该接口由宿主显式调用，旧请求、CLI、程序包格式及默认 Application 流程保持。

[离线测试与真实重载结果](agent-compiled-resources-acceptance-r828.md)


## 逐节点近似来源

可信 PyTorch v5 来源包包含节点／模块／默认近似配置和确定性的选择记录。应用仍通过 `submit_import`、`run` 和 `execute` 使用原接口；模型、原始图、分解和来源元数据共同参与身份校验。

选择器的冲突、缺失匹配和配置篡改在导入层返回诊断。生成端接收已选近似的公开图，执行与数值验收继续使用冻结计算 reference；原模型应用精度单独标记。[格式与示例](operator-decomposition-and-python.md#逐节点与模块近似)。

## 身份分析缓存

身份计算可以复用按文件名和源码内容哈希索引的不可变语法事实，最多 1,024 条，语法事实大小预算 2 MiB。每次请求仍重新读取、哈希并检查源码与依赖文件；完整身份或后端核验结果不使用该缓存。

缓存调整不改变身份格式、请求接口或程序包格式。源码变化照常产生新绑定，旧记录不改写。[性能与安全回归](agent-identity-cache-acceptance-r730.md)。

## 归一化规则表达

Poseidon CPU 的当前容量规则 lowering 可识别两个逻辑元素的中心化图，复用方差并重排平方反馈中的乘法。判断依据为形状、图依赖和已绑定常量，Provider 源码保持原样。

请求、近似配置、reference、参数和应用接口保持。其他形状沿用原 lowering；计算深度记录是诊断信息，真实编译和数值验收继续决定成功。六个代表例、原失败模型及程序包复用见 [r746](agent-normalization-lowering-acceptance-r746.md)。

## 静态索引来源包

`submit_import` 接受可信 PyTorch v6 来源包，校验静态索引计划、原图节点、整数状态及来源哈希。v6 容量图默认选择 `capacity-public-tensors-v9`；显式旧 packing 和旧格式维持原行为。任务、预算、修复和返回接口保持。

重复输出的规则 baseline 采用受绑定加密零的等价表达，只处理可信规则源码。Provider 生成程序沿用静态、编译、产物与数值验收。[输入范围与程序包测试](agent-pytorch-index-acceptance-r761.md)。


## 源码依赖目录

`application-source-catalog-v1.json` 统一保存生成、编译、执行和验证入口及动态加载声明。目录与实际源码参与身份绑定；路径、字段、重复项和动态调用格式均在加载时校验。

身份分析继续使用整个模块的保守闭包。当前资格验收直接使用共用会话，公开反馈使用 Provider 无关检查器；共享模型语义仍影响源码闭包，编译、执行与验证身份尚未完全独立；当前三阶段已移除Provider与旧实验运行器依赖。开发工具 `benchmarks/tools/audit_application_dependencies.py` 可查看真实依赖路径。

[目录检查及三个 Poseidon CPU 回归](agent-identity-catalog-acceptance-r766.md)。


## 资格验收会话

`component_backend.qualify` 直接验收固定请求与候选，不构造 CLI 参数。
`QualificationSession` 与原 CLI 共用沙箱预检、原生执行槽锁、`CandidateValidationAdapter` 和冻结检查。
应用端在原 Controller 预算内生成与修复，会话只验收当前候选。

`public_feedback_contract` 处理公开反馈字段和边界，Provider 保留原兼容包装。
返回结果、compiled／numerical 等级、任务接口和程序包格式保持。
[组件、旧 CLI 与程序包验收](agent-qualification-acceptance-r776.md)。

## 生成来源与数据读取边界

程序包通过 `generation_receipt.verify` 核验固定请求、返回源码、配置、usage和调用账本。
该接口使用数据契约；生成控制层仍用 `application_generation.capture` 保存真实返回记录。
原记录格式、公开反馈、预算和Provider别名接口保持。

`strict_json_files` 是应用、程序包与开发工具共用的受限JSON读取和原子报告写入入口。
程序包的数据核验可在未导入Provider、生成适配器和Benchmark运行器时进行；源码身份仍采用保守闭包。
旧来源报告若与当前捕获身份不同，保留原记录并返回拒绝，不改写旧绑定。
[来源、兼容与Poseidon CPU验收](agent-receipt-acceptance-r789.md)。

## Poseidon 沙箱清单

`poseidon_sandbox_policy`声明当前候选可见的只读文件，`process_resource_limits`提供共用进程限制。
资格验收、CLI和程序包实施身份绑定此清单；挂载字节或依赖变化产生新身份。
缺失、重复、路径非法或文件间接跳转均在身份检查中拒绝。

资源限制和bubblewrap规则保持原值；人工helper探针保留为历史开发工具。
可信PyTorch导入直接使用公共限制；应用仍保守绑定共享模型语义。
[真实helper、native、程序包和CLI回归](agent-sandbox-profile-acceptance-r797.md)。

## 可信 PyTorch 捕获依赖

捕获入口直接使用`process_resource_limits`，其实际字节参与捕获冻结、来源报告和重载检查。
v1–v6的输入格式保持；旧报告源码身份不符时明确拒绝，保留原包并重新捕获。

当前编译、执行和验证的源码身份不再绑定Provider、生成适配器、旧沙箱或实验运行器。
共享模型语义仍采用整个模块的保守闭包。三个现有模型的真实捕获、CPU数值、重载及新输入复用见[验收记录](agent-capture-boundary-acceptance-r805.md)。

## JSON 矩阵模型导入

`scripts/prepare_model.py --graph ... --resource-profile ... --write --output ...` 接收 `poseidon-operator-graph-v2`，随后沿用 `app.submit_import(directory, options=...)`。目录包括原图、规范化图、分解与独立参考记录；程序包核验显式版本、矩阵规划及实现身份。容量图默认采用 v9 路由，显式布局保留。

提交、状态、取消、预算、Provider、验证等级及程序包接口保持。矩阵点积标量使用 `[1]` 张量，命名输出按该签名返回。[用法](operator-decomposition-and-python.md#json-矩阵计算图)。


## 结构化输出适配器

`TaskOutputAdapter(app)` 为已有任务绑定宿主选择的 Tensor、tuple/list、字符串键字典及静态 None 结构。宿主通过 `bind` 获取收据，通过 `result` 重载核验，再通过 `execute` 调用原程序包执行接口。任务、请求哈希、预算、取消及 compiled/numerical 等级沿用原接口；收据单独绑定输出结构和任务/包身份。

用法见[结构化输出接口](agent-output-delivery-v1.md)。当前完成 53 项契约测试、24 项组件测试及 96 组明文结构检查；新适配器的真实密态与 Agent 验收单列。


### 执行路径的资源预检

首次数值资格验收、`app.execute()` 和 CLI replay 共享编译产物预检。
固定参数与真实 RuntimePlan 给出保留系数和常量槽位的存储下界；超过现有
4 GiB 地址空间限制时，返回 `compiled_retained_payload_exceeds_budget`。

反馈按原请求预算进入有界修复和重复失败检测。程序包新输入执行返回
`validation_failed / resource_gate`；预检文件或依据变化返回 `service_error / integrity`。
成功的下界预检继续执行验收，峰值内存和密钥空间保留为未知项。

[控制与 Poseidon 回归](agent-resource-gelu-progress-r1305.md)记录本批来源和绑定。


## 命令工作进程与构造反馈

`scripts/agent_assessment.py` 每次命令通过独立 Linux supervisor 管理所属工作进程。结束或超时后清理整个所属进程树，返回 `process_cleanup.confirmed`、`status` 和绑定收据。后台子进程未正常结束时，返回 `assessment_entry_background_worker`；清理无法确认时，返回 `assessment_entry_cleanup_unconfirmed`。该机制管理单次命令的进程；显式 v2 验收计划提供跨命令总期限，CLI advance 监控期限与取消并清理所属进程。

公开常量、声明函数、可信 helper、`zero_ct` 及 `hc/np/object/range` 是保留绑定。覆盖这些名称的 native 构造返回 `native_reserved_binding`、源码位置和使用新局部变量的提示，进入原有预算内修复。诊断字段继续经过公开白名单检查。

[控制、真实命令与 Poseidon 重放结果](agent-supervision-binding-feedback-r1336.md)。

## 构造成本与旋转反馈

统一图提交可以显式设置 `options={"generation_guidance": "explicit-v13"}`，CLI 对应 `--unified-guidance explicit-v13`。原默认及 v1–12 保持。

生成上下文从布局给出实际旋转步数、保留绑定和展开限制。native helper 每次调用会重复计入其全部公开常量注册成本；复用中间结果可以减少重复调用。静态工作量失败时，已有 AST 分析重算成本，并将经过原拒绝诊断校验的调用位置和计数送入同一个预算内修复流程。

构造成本不代表乘法深度、密文操作数或内存。验收规则和资源限值保持，成功仍由真实编译及所选验证等级决定。[兼容测试与三个 Poseidon CPU 回归](agent-construction-guidance-r1337.md)。

## 数据集验收总期限

`ApplicationAssessment.submit(..., total_wall_seconds=240)` 和 CLI submit 的
`--total-wall-seconds 240` 选择跨命令总期限，范围 1–14,400 秒。后续 advance
沿用原期限和调用额度。Linux 启动身份、CLOCK_BOOTTIME 与墙上时钟共同校验，
重启或无法确认时钟时返回服务异常。

CLI advance 在运行中监控取消和期限，清理所属 worker 及其子进程；未确认的执行
保留已用额度和不确定状态。直接同步 Python API 在执行边界响应取消和期限。
已完成结果可重复重载；默认 v1 接口、模型请求和程序包格式保持。

[控制兼容、真实进程和程序包验收](agent-assessment-total-budget-r1338.md)。

## 静态张量拆分来源

可信 PyTorch v17 在现有来源捕获服务内接入 chunk/unbind；
submit_source → prepare_source → 原生成和资格验收接口保持。
原图、静态端口、共享视图、规范化图及实现身份随来源包核验。
源清单和程序包使用显式 v17，旧版本及其哈希规则保持。
应用正常请求沿用固定预算和受控反馈，不依赖开发验收脚本。

六个规则程序完成真实 Poseidon CPU 资格验收和程序包新输入执行。
[使用方式](operator-decomposition-and-python.md#pytorch-v17-静态-chunk-与-unbind) ·
[阶段记录](agent-pytorch-partitions-acceptance-r1339.md)。
