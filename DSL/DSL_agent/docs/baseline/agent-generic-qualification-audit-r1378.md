# 通用接入与能力资格审查

## 本轮结果

唯一开发目录为 ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e480`。本轮新增独立开发工具和文档，生产 Agent、Dacapo、CKKS Runtime、Poseidon、参数、reference、误差门限及网络配置保持。

[机器可读索引](agent-generic-qualification-index-r1378.json)绑定 `b75a090803d2a519ca436c3917e17b37bd39f8555b55c28a324a3c3b3e6cbbaa`，当前 SDK 为 `8ca41ede84d20c654fdc7885658e343d1d263cfc0885fd1a96ad259c54e38f8a`。工具读取 AST、JSON 和文件字节，不导入模型，不启动捕获、编译、密态执行或 Provider。

| 检查 | 结果 | 适用范围 |
|---|---:|---|
| 新索引控制测试 | 26 passed，failed/errors/skipped 0 | 防止报告/产物/候选篡改、错用后端、数值误判、历史身份被提升 |
| 当前阶段来源与选定证据文件 | 2,215 个文件哈希 | 已冻结阶段来源和本索引选定的历史证据 |
| 高层注册表 | 34 项，与现有参考集一致 | 登记、参考证据；不代表全部组合可执行 |
| 独立分解参考记录 | 110 个不同结构；65 份来源文件匹配，45 个工具生成上下文 | 每项至少三个上下文；本轮未重新执行参考 |
| 历史 Poseidon CPU | 15 个参数分区，每区 3 例，共 45 例 | 规则候选、原始身份；45 模型/请求，40 份不同 DSL |
| 历史产物留存 | 1,045 个产物哈希匹配 | 原 tracing、IR、RuntimePlan、bundle、执行产物；未重新运行产物检查器 |
| 历史逐项数值复核 | 180 组、992 值；最大绝对误差 5.0009183362e-7 | 重查原报告的解密值和 reference；未重算独立 reference |
| 历史解码/reference文件关联 | 45/45 例、992 值精确一致；14 控制通过 | execution outputs→selectors→报告/decrypted；冻结 NPZ→reference；未重新执行 |
| 当前组件控制测试 | 88 passed，failed/errors/skipped 0 | Fake Provider/Backend；只读实际 SDK 身份；不计为真实 Agent 或密态验收 |
| 新编译／密态／API／HTTP | 0／0／0／0 | 当前身份尚无本批新后端验收 |

索引没有把参考、规则后端和真实 Agent 数量相加。45 项旧执行记录缺少现代 task ID 和四阶段身份，这些字段保留为 null；索引派生哈希只用于追溯。原记录和原始响应保持。

本索引核验冻结任务中声明的源码及配置字节；它不是对所有安装库的独立重新验收。r1376 的只读 plan 另行检查实际后端身份并通过。应用资格字段保持 not_run，历史证据没有提升为当前 SDK 通过。

补充记录 [历史数值文件关联](agent-historical-numeric-link-r1379.json) 绑定 r1378 及原报告、请求、模型、固定数组。原索引未收录的 execution.json 和 decrypted.npy 仅按两个固定文件名读取，并在新记录中单列本次 SHA；没有补写旧索引。该关联检查证明留存文件相符，没有重新运行模型或独立 reference。14 项控制的实际日志为 numeric-link-controls-final.log；[日志绑定](../../scripts/baseline/benchmarks/tools/generic_qualification_index_v1/numeric-link-log-proof.json)独立核验其 SHA 和 14 个测试名，原 9 项日志保留。

[88 项组件控制结果](../../scripts/baseline/benchmarks/tools/generic_qualification_index_v1/application-controls-final.json)包含提交、取消、有界修复、诊断、程序包交付及阶段复用。第一次运行已有 78 项通过，另 10 项因测试启动器未应用已有 Nix 进程内库路径而未启动；应用现有锁定 loader 后只补跑这 10 项，全部通过。最初错误日志和计数修正均保留，未安装依赖、修改全局环境或重跑已通过 78 项。

r1377 为初版快照，保留其原绑定及原工具副本。r1378 收紧了当前资格字段和工具/测试绑定，26 项控制通过；两份快照不能混用工具身份。当前开发索引并未接入应用能力接口，也不是新的发布能力声明。

当前 24 个基础图算子、34 个高层算子已登记。逻辑最高配置 `general-tensors-v1` 为 1–4 输入/输出、rank 1–32、输入/输出各 4096 元素、中间张量 65536 元素、原图/展开图 1024/4096 节点、公开权重 100 万元素。物理密文、公开向量、路由、深度和内存另行检查，逻辑上限不是运行保证。

## 常见调用形式

“已登记”表示当前条件内存在接入映射。历史后端结果仅适用于原案例、表达和依赖身份；本表不是通用模型白名单。

| 调用形式 | 当前接入 | 主要条件或缺口 |
|---|---|---|
| Sequential/ModuleList，Linear→square→Linear，公开循环 | 已登记 | 静态模块列表；历史三种写法有密态证据 |
| Depthwise Conv2d→BatchNorm→AdaptiveAvgPool→Linear | 已登记 | rank4、公开 running statistics、显式 padding；历史三种写法通过 |
| torch.add/Tensor.add(alpha) 与残差 | 已登记 | 公开有限 alpha、合法广播；历史数值通过 |
| interpolate/Upsample→concat→Conv | 缺映射 | resize 接入尚未登记；本轮未执行 |
| ConvTranspose2d/conv_transpose2d | 缺映射 | 转置卷积接入尚未登记；本轮未执行 |
| shape 查询→reshape/squeeze→Linear | 已登记 | 静态维度与元素数匹配；有历史真实 Agent 结果 |
| LayerNorm 模块／函数／手写分解 | 有条件 | 显式 rsqrt 近似；原函数应用精度另计 |
| SDPA／QK→Softmax→V | 有条件 | 显式近似、公开 mask；原 v11 三个规则表达编译失败，不能由映射推断通过 |
| MHA＋mask/padding＋残差 | 有条件 | 静态尺寸、公开 mask、显式近似；低阶规则有历史通过，cross 的后续 Provider 失败不等于不可表达 |
| forward(batch_dict)、整数 input_ids、可选 forward 参数 | 缺签名 | 当前只接受 1–4 个有序 float64 Tensor；缺少输入树和参数绑定 |
| 公开 buffer.float/double/to(dtype) | 有条件 | 显式 v24，认证公开状态；161 控制/33 捕获通过，后端待验 |
| 字典/list/tuple 含 Tensor 与 None | 已登记 | 最多 4 个 Tensor 叶，登记版本；named 原案例在旧 e2 身份补验通过 |

来源包目前为同目录平铺 1–8 个 Python 文件、合计 64 KiB，工厂入口 `module.factory`，工厂参数为 JSON 数据，权重为完整且认证的 NPY 状态包。嵌套包、相对导入、输入树和可选参数是正常模型使用习惯中的主要缺口。v24 类型转换只处理公开状态；输入仍为 float64。

具体每个算子的 rank、轴、广播、参数和近似条件，以及模型/候选/报告映射，见索引 `operators`、`normal_source_patterns`、`reference` 和 `historical_cpu`。

## 共同问题与实施顺序

| 顺序 | 改动 | 完成检查 | 后端边界 |
|---|---|---|---|
| 1 | 接入结构化诊断、请求级产物记账，接通数据型资格清单 | 旧请求/哈希兼容；输入与环境错误分开；预算跨修复累计；受控反馈无私有数据 | Agent 层 |
| 2 | 输入树/公开参数绑定、来源包组织、常用等价写法与明确 dtype 规范化 | PyTorch 与等价图参考一致；分类器/解码器/Transformer 的普通调用形式共同验收 | 不执行任意上传 Python |
| 3 | 各算子参数范围与组合、布局成本和通用 routing | 参考、真实编译、密态、Agent 分层登记；未知内存/深度不记可执行 | 不能用 Python 估计替代 Runtime 生命周期改动 |
| 4 | 逐节点近似与应用精度、现有 bootstrap 接口 | 原函数、近似、分解、CKKS 四层误差；bootstrap 独立验收 | 参数/算法/Runtime 核心修改另批 |
| 5 | 多模型族、未见组合、整模型及容量 | ResNet20 是一个整体验收子集；MLP/RMSNorm/Attention 和普通模型接入分别登记 | CUDA 必须有 NVIDIA 实机 |

先完成共同控制层，再合并处理输入形式和算子组合，减少逐个小写法追加版本。能力资格清单由发布流程生成数据；正常应用请求读取明确版本的数据，不扫描实验报告或依赖人工修复脚本。开发索引用于准备该清单，目前尚未接入应用资格接口。

## 最小控制层接口草案

本节为设计，尚未修改生产接口。r1376 已冻结，结算前不追加字段或重签任务。

保留旧 `limits` 的五项预算、旧 DSL request 与哈希。新增显式可选参数：

~~~python
Application.submit(model, ..., admission_policy="legacy-v1", operational_limits=None)
Application.submit_source(source, ..., admission_policy="legacy-v1", operational_limits=None)
~~~

新策略选择 `structured-v1`；新运行预算示例：

~~~json
{
  "format": "poseidon-operational-limits-v1",
  "max_new_artifact_bytes": 536870912,
  "min_free_disk_bytes": 4294967296
}
~~~

省略新增参数沿用旧行为。新预算进入可信任务/来源快照绑定，不进入模型、Provider 请求、DSL request 或已验证程序包复用键。改变预算本身不要求重新生成。未知版本、额外字段、负数、非有限值和 bool 冒充整数拒绝。

接入诊断采用独立版本，不改写已冻结的 `poseidon-diagnostic-v2`：

~~~json
{
  "format": "poseidon-admission-diagnostic-v1",
  "layer": "input",
  "code": "model_constraint_rejected",
  "reason_code": "input_element_capacity",
  "location": {"kind": "input", "index": 0},
  "context": {"observed": 4097, "limit": 4096},
  "owner": "request",
  "retry_policy": "never_automatic"
}
~~~

只使用可信检查器能证明的原因与公开约束。不从任意异常文本猜测原因；无法细分时返回受控通用码。依赖缺失归 `service_error`，不归模型不支持；数据篡改归完整性错误。接入失败发生在生成前，不发送 Provider 修复反馈。

产物记账由同一 Controller 管理，覆盖本任务来源、响应、全部修复轮次、资格目录及交付包。目录先登记所属关系，拒绝链接逃逸，重叠目录只计一次；失败记录保留，预算不因修复或转入生成阶段重置。

首版采用保留文件的逻辑字节数和观测高水位，并记录 `enforcement=monitored-v1`、`hard_filesystem_quota=false`。目录监测只能在检测后停止；现有单文件限制不等于累计写入 quota。记录真实观测超额，保留现场，不扩大额度或删证据。

| 条件 | 终态 |
|---|---|
| 模型/能力/调用约束拒绝 | unsupported |
| 本任务产物预算耗尽 | budget_exhausted |
| 磁盘余量或依赖不足 | service_error |
| 目录/任务绑定被篡改 | service_error，完整性原因 |
| 取消或时间截止 | 原 cancelled/timed_out |

这些事件停止请求，不请求模型修复，也不外发路径、磁盘或环境信息。旧注入后端不支持资源记账时，新请求须在生成前明确拒绝，不能忽略预算。

接线位置为 `application_component.py` 的提交/Controller/交付、`application_model_source.py` 的来源与规范化接入、`qualification_session.py` 的产物目录登记、`validation_adapter.py` 的阶段边界，以及显式采用运行预算的 `program_execution.py`。所需控制测试包括旧哈希兼容、精确拒绝、未知原因脱敏、环境错误、跨修复累计、磁盘不足、链接/记账篡改、预算改变后的缓存复用、真实超额和取消竞争。代表性真实后端与程序包验收另行冻结范围。

## 未完成与下一步

- 生产应用已具备提交/状态/取消/有界修复/交付/包复用路径；本轮选取 88 项组件控制通过，没有重跑全部 Benchmark。
- r1374 的 161 控制与 33 捕获属于当前 SDK；r1373、r1366 等真实后端/Agent 结果属于各自历史身份。
- r1376 的 36 份来源配置、12 个去重请求待批准；本轮核验未启动且绑定保持。下一条只读命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_dtype_offline_v1/run.py --plan
~~~

- 原生 bootstrap ratio128 试验：5 次阶段检查通过、1 次失败、6 次未运行；第二次有符号刷新 15 值超门限。尚未向 Agent 开放 bootstrap。
- 原尺寸 ResNet stem 的卷积后通道均值有旧规则密态证据；这不是完整残差块或 ResNet20。全测试集精度、整网密态和真实 Agent 整网生成仍未完成。
- 缺少 NVIDIA 实机，CUDA 执行未验收。当前 runner 缺少同密钥跨调用密文交接，不能用分段解密调用代替整网密态执行。

本轮没有付费调用、安装、下载、网络修改、后端/参数修改、提交或推送。工具最初不识别注册表的静态字典展开；修复并收紧资格与工具绑定后完成 26 项控制。解码补充工具原先要求两个未入旧索引的留存文件必须存在索引，改为固定名称的新补充绑定后通过 14 项控制。原错误、原控制结果及工具副本保留。数值门限保持 `1e-5 + 1e-4*abs(reference)`。
