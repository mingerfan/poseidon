# Agent 组件接口 v1 与 DSL 程序包 v2

应用集成的新生命周期入口见 [面向应用的模型到 DSL 组件](application-agent-component-v1.md)。下文 v1 worker 和旧数值默认保持兼容。

本页说明 ARM Ubuntu checkout 中的组件接入边界。它复用现有 Agent、校验器和 SEAL CPU 执行链，不改变 Dacapo、CKKS 算法或安全参数。旧 scripts/agent.py、schema 1–5 和旧请求规则继续保留。

## 接入方式

宿主框架首先使用独立 Python worker。不要从业务进程直接调用 run_candidate.inside()：该函数管理环境、线程、测试密钥和实验目录，仍不是线程安全的库接口。

```bash
# ARM Ubuntu；x86 Ubuntu 使用 x86_64-linux。work root 仍由 POSEIDON_WORK_ROOT 指定。
POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/agent_component.py --job /absolute/job.json
```

job 是不超过 256 KiB 的普通 JSON 文件，必须使用精确字段；不支持任意命令、插件路径、凭据或 live action。stdout 返回一个 JSON 终态，执行日志写入 stderr。付费生成没有因接入此 worker 自动获准。

最小只读 job：

```json
{"format":"poseidon-agent-component-v1","action":"capabilities"}
```

| action | 必需附加字段 | 结果含义 |
|---|---|---|
| capabilities | 无 | 声明的图边界、契约和 helper profile；不是执行覆盖率 |
| prepare | model、options | 完整公开 request；尚未执行 |
| validate | request、candidate | 程序合法性与定向构造静态证据分别返回 |
| qualify | request、candidate | 使用既有隔离环境执行、解密及独立数值审计 |

model 是统一图 poseidon-model-graph-v1 的对象，candidate 仍是 schema/request_id/hecate_source。旧 schema 1–5 暂通过原 CLI 使用，未声称新 worker 已统一接入这些旧输入。

options 仅接受 compiler_configuration、construction_profile、construction、helper_profile、helper_exercise、chunk_period、generation_guidance、capability_composition。默认编译配置为 seal-cpu-eva-w45-v1，显式 null 保留无配置扩展的旧请求行为。构造和 helper 的互斥规则继续由现有契约校验，不自动拼接规则文本。

## 核心与验收职责

scripts/baseline/component_contract.py 提供 prepare_task、reconstruct_request、validate_program、synthesize、capabilities。导入该模块不修改进程环境，也不导入 Torch。宿主自行注入可信 provider、evaluator 和记录回调；这不授权使用任意动态插件。

synthesize 沿用原有最多三轮修复循环。evaluator 的成功必须同时具有 encrypted_execution=true 和 numerically_validated=true；仅静态通过不能结束为成功。失败只转发通过 public_feedback 白名单的受控诊断。完整验收报告中的本地路径、参考数组等不进入 provider 历史。原始响应与完整验收报告由宿主分别留存。

validation_adapter.CandidateValidationAdapter 从单例运行器中提取候选验收逻辑；旧 CLI 与 component_backend.qualify 共用它。ValidationContext 保存可信宿主的请求、冻结文件、测试密钥路径、沙箱调用与完整性回调，不进入模型提示词。

component_backend.qualify 负责四组固定输入、独立 reference、AST、bubblewrap、Hecate tracing、Earth/CKKS、HEVM/CST、安全参数检查、SEAL 密态计算与解密比较。它使用临时测试密钥；目前不是接收客户密文的部署推理 API。用户私钥不属于生成核心接口。

数值门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。

返回状态与指标应分别使用：

- semantic_validation=accepted：静态程序合法性检查接受。其他状态均不能当作合法性通过，失败原因见 failure。
- construction_coverage=static_witness_obtained：获得定向静态/有限干预证据；尚不代表真实密态通过。
- construction_coverage=not_verified：定向证据未获得；可能是候选不满足，也可能是有限检查器证据不足。
- status=static_accepted_not_executed：没有执行。
- encrypted_execution=true：实际密态执行完成；数值失败时仍可为 true。必须同时检查 status=passed 且 numerically_validated=true 才是完整验收通过。observed_stages 另列各阶段及完整性是否被拒绝。
- new_agent_generation=false：本次是已有候选验收，不能计为新的 Agent 生成成功。

failure 含 layer/code/owner/retry_policy/diagnostic。编译/产物拒绝保守归为 candidate_or_backend，不自动断言编译器 bug。尚未获证实的构造贡献不自动推断成非法 DSL。生成端的诊断与宿主私有证据分开。

## 导出与精确重放

```bash
POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/dsl_bundle.py export \
  --evidence /absolute/passed-evidence --output /absolute/new-bundle

POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/dsl_bundle.py replay \
  --bundle /absolute/new-bundle

POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/dsl_bundle.py replay \
  --bundle /absolute/new-bundle --execute
```

不带 --execute 只查完整性。v2 包包含 model.json、candidate.py、完整 request.json 和 manifest.json，保存 public/native/helper/chunk/定向要求、编译配置、可信依赖、源文件和二进制身份。目录可移动；已有输出目录不覆盖。v1 的原生程序重放路径保留。

若输入来自高层算子分解或受限多文件 Python，导出时增加 --model-import /absolute/import-directory。包内 model-provenance.json 保存原始图、分解映射、显式近似配置、参考报告及受限 Python 来源。重放重新解析来源并执行独立数学差分。它不会导入上传的任意 Python。

近似数学函数与原函数的误差仍独立于 CKKS 数值误差；provenance 中 approximation_accuracy_certified=false 不能被解释为近似精度已经认证。没有提供来源目录时明确记为 not_supplied。

已付费 Agent 的历史证据可用 --live-approval /absolute/exact-config.json 进行来源审计；该参数不发起 API 请求。它核对保留响应、请求身份、源码、反馈及调用账本，不能将人工替换的源码记为 Agent 原始生成。内部证据完整性不是服务商签名或账单证明。本轮新增 live 导出路径只完成合成证据测试，没有新付费端到端验收。

精确重放要求匹配的 SDK、依赖和验收实现。换架构/换 SDK 必须显式重新验收；当前没有自动跨平台重绑定功能。manifest 的自哈希也不是外部可信签名。

## 生命周期、资源与边界

每个 qualify 在独立 worker 中运行，复用已准备好的 Nix 环境，候选继续各自沙箱化。同一 work root 的原 native 槽锁保持上限 2；不能把不同 work root 当成共享全局锁。没有重建 SDK，也没有新增安装。

已验证 pinned worker 的 SIGTERM 取消与部分/完整临时密钥清理。部分密钥只接受活跃 worker 持有的目录 inode/device 身份；离线回收仍要求完整安全参数元数据。未知文件、软链接及共享硬链接继续拒绝清理。SIGKILL、主机断电以及外层 SSH/Nix 启动器的端到端取消没有得到本轮保证。

新 API 是组件集成的边界层：还没有持久化 JobHandle、跨进程恢复服务、完整 RunContext 注入或部署 runtime。预算和付费批准继续由可信宿主管理，不能承诺 provider exactly-once。

## 明确选择能力组合

默认不合并不兼容契约。新选项 capability_composition="native-bn-directed-v1" 仅开放真实上游 BN 与以下 native 定向构造的组合：

- unified-native-call-scalar_cipher：本地 helper 的密文参数调用。
- unified-native-copy：对象数组复制。
- unified-scalar：标量增强赋值。

prepare job 的 options 示例（model 仍是独立统一图）：

```json
{
  "helper_profile": "upstream-poly-bn-silu-v2",
  "helper_exercise": ["HE_BN0"],
  "construction": "unified-native-copy",
  "capability_composition": "native-bn-directed-v1"
}
```

旧 CLI 使用相同组合名及既有参数 --unified-helpers、--unified-helper-exercise、--unified-exercise。只读请求准备推荐使用组件 prepare action；这些参数本身不授权付费调用。组件 prepare/validate/qualify 和程序包导出重放均保留该选项。未选择它的旧请求哈希不变。

这个组合要求 native 契约、明确 BN 定向任务、不分块；实际调用只能使用已绑定的 HE_BN helper。共享 helper profile 中的 SiLU 不因本选项开放。两类构造分别验证输出贡献、真实 frontend 调用与数值结果，使用其中一类不能冒领另一类的覆盖。

已验范围为每种构造三个不同拓扑：BN、BN→square、BN+输入。既有 public 公开循环与分块组合另有三个回归上下文。不能由此宣称任意 helper/public/chunk 组合均可用。

## DSL 检查状态与后续

public witness v13 修复 values/items、映射写入、切片写入、排序 key、starred 和列表推导的类型保持干预，并补记映射下标写入事件。native witness v3 能在绑定的 BN 数学探针下独立检查 native 构造。有限干预没有找到贡献证据仍然拒绝，不放宽为通过。

结构化诊断区分 required_construct_not_executed（没有观察到要求的事件）与 construction_evidence_unresolved（有限探针未得到见证，需要复核候选与检查器）。后者不能直接解释为语义非法或对所有输入均无影响。

401 分区、107 上游 API 分类继续保持历史评分绑定。当前逐项复查与真实执行结果见 [验证适配器验收](validation-adapter-acceptance-r171.md)，不将局部回归改写为全量覆盖率。33 个编译相关模型和真实 bootstrap 缺口未改。

类型说明勘误：旧 public-v1 规则文字只列出容器返回，但当前实现也允许满足输出绑定的单 Expr 返回。为保持旧请求哈希，未悄悄改旧规则字符串。下一版契约需统一 PublicScalar/PublicTensor、Plain、Expr、对象数组与返回形态，并分别补正反例。

后续仍包括完整 RunContext/JobHandle、旧 batch 的宿主生命周期接入、尚未证实的贡献构造、类型边界及更多三上下文能力组合。本轮不提供客户密文部署接口，也不宣称完整 SDK、任意 Python 或 Poseidon GPU 支持。
