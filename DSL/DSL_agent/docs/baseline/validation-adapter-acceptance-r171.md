# 验证适配器、构造检查与能力组合验收 r171

结论：候选验证逻辑已从单例运行器中提取，旧 CLI 和组件 qualify 共用真实验收适配器；修复已确认的构造见证缺陷，并以显式配置开放 BN helper 与三类 native 构造组合。本次没有修改编译器，没有新 Agent 生成或付费调用。

## 具体变更

- validation_adapter.py：CandidateValidationAdapter / ValidationContext，统一响应检查、AST、隔离 tracing、编译、产物门禁、SEAL 密态计算、解密、数值检查、完整性和受控反馈。保留原冻结参考与安全参数。
- component_backend.py：执行事实与验收结论分开。数值失败仍报告 encrypted_execution=true、numerically_validated=false；产物门禁拒绝则没有密态执行。observed_stages 保存 tracing/编译/执行/比较阶段，完整性拒绝不被报告为可信执行。
- public witness v13：修复 values/items、字典下标写入与 update、切片写入、排序 key、starred zip、列表推导的类型保持干预；补记映射写入事件。缺少事件与有限探针没有见证分别报告，未放宽贡献要求。
- native witness v3：定向构造探针可使用绑定的上游 BN 数学求值器；真实 helper 调用与 native 构造各自验证，不能相互替代。
- capability_composition=native-bn-directed-v1：只开放 unchunked BN 与 unified-native-call-scalar_cipher、unified-native-copy、unified-scalar。必须显式选择并指定 helper exercise；SiLU/public/chunk 等不自动混用。
- 旧 schema、默认行为、2030 份完整 request 及其哈希保持兼容。程序包绑定保存新组合配置。

适配器是可信宿主的验证层，不是线程安全 SDK；完整 RunContext、持久化 JobHandle、客户密文部署 API 仍未实现。

## 验收结果

| 检查 | 结果 |
|---|---|
| 最终相关单测 | 146 passed / 0 failed / 0 errors / 6 skipped |
| 新 BN/native 组合 | 9/9 真实密态通过；每类三个不同拓扑 |
| 既有公开循环＋分块组合 | 3/3 真实密态通过 |
| 修复检查器后的原 Agent 源码重放 | 6/9 通过，3/9 保留失败；没有改原响应 |
| 三份独立人工修复 fixture | 3/3 真实密态通过，不计 Agent 成功 |
| 旧 Linear CLI self-test | 通过，包含预期响应/静态/数值故障 |
| 组合程序包导出与重放 | 通过，最终版本重新编译及解密比较 |
| 验证适配器执行中 SIGTERM | cancelled、退出130、临时密钥清理通过 |
| 最终状态接口回归 | 产物拒绝、已执行但数值失败、完整成功均正确区分 |
| 编译器/运行时保护 | 93 个源码及二进制哈希不变 |
| 旧请求 | 2030/2030 精确重建不变 |

6 项 skipped 依赖其他特定历史证据目录，不计 pass。21 项矩阵是 18 passed / 3 failed；人工修复、重放与接口回归另列，不能相加成唯一模型数。每项成功都实际经过 Hecate、Earth/CKKS、HEVM/CST、SEAL、解密与四组输入逐项比较。通过项最大绝对误差约 1.6046e-7，冻结门限仍为 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。

新组合的三个拓扑为 BN、BN→square、BN+输入。此次没有验证所有 BN 变体或所有 helper/public/chunk 笛卡尔组合，不能宣称当前全部 DSL 分区密态覆盖完成。

## 原 Agent 三项失败及已验证修复方向

| 原案例 | 原失败层与证据 | 人工修复及结果 |
|---|---|---|
| construct_083_2 | artifact_gate：MulCP 的公开操作数全零，有透明密文风险；明文算术与 reference 一致 | 删除数学上为零的公开项，保留门禁；密态误差约7.43e-9 |
| construct_116_0 | numerical_comparison：最大误差约2.10975；候选精确算术与 reference 也相差2.10975，而密态与该算术仅差约7.09e-9 | 修复 concat 的周期8旋转方向、mean 的单槽选取与广播、长度6输出的重排；误差约1.42e-8 |
| construct_117_2 | numerical_comparison：最大误差0.125；精确算术也差0.125，密态与算术仅差约5.12e-11 | 周期8下将第二个长度3输入右移3等价为左移5；误差约4.42e-11 |

已确认：这两项数值失败来自候选布局算术，不是定向构造本身导致 CKKS 精度降低。透明密文项由安全门禁准确拒绝，不通过关闭门禁或修改编译器处理。

三个原 request、原响应和原失败保持不变。人工修复源码与元数据位于 scripts/baseline/cases/validation-adapter-r167，明确 manual_repair=true、new_agent_generation=false。后续若要让 Agent 自行修复，需另有精确付费批次授权。

## 尚未解决的项目

原86份候选的本次分母保持完整：6份原程序密态通过、3份后续层失败、21份贡献证据未获验证、56份类型/契约拒绝。逐项原诊断和修改方向在同名 JSON 的 retained_cases 中。

21份贡献项里，6份缺少要求的可信事件；其他例子包括：

- continue 在循环末尾，不改变计算；同名局部 counter 不是可信事件计数器。
- nonlocal 只写不读；sorted 的结果做对称求和，顺序不影响输出。
- fresh.USub 没有区分原对象与新存储，overlap 使用普通赋值而非要求的重叠增强赋值。
- 对 arr[0] 的标量取负代替数组取负；数组全是 Expr 却要求 public_cells。
- mixed array 只在局部创建，没有从 helper 返回。

这些不能通过放松检查计为成功。有限探针无见证也不是所有输入上的无影响证明。56项类型/契约拒绝须分别处理：密文当公开量、ragged array、编码形状、禁止的调用等继续拒绝；Plain/object dispatch 和尚未开放的组合需要新版契约、类型规则及正反例，不能一概归为模型错误或编译器缺陷。

33个既有编译容量/level/scale模型未修改、未重跑，模型列表仍在 compiler-blocked-models-r159。真实 bootstrap 后端缺口保持阻塞。

## 证据绑定与复现

[完整报告](validation-adapter-acceptance-r171.json) · [401分区及逐项静态视图](component-capability-r170.json) · [组件接口及新组合用法](agent-component-v1.md)

原始证据在 /home/lhohy/poseidon-work/platforms/aarch64-linux/results/validation-adapter-r167。

- acceptance-final：21项真实验收矩阵、旧CLI兼容与2030请求检查。
- followup-final：三份人工修复、独立算术定位、组合包重放与执行中取消。
- api-final：最终146/6单测、三种阶段状态的真实回归、最终版本程序包重放。
- static-r170.json：86项原响应静态复查；旧报告保持原绑定。

矩阵与人工修复使用当时源码哈希。最后只修改 component_backend.py 的结果元数据与两个对应测试；实际验证适配器、构造规则、参考、沙箱和编译链未再变化。api-final 对受影响接口重新执行，不改写旧绑定。最终报告记录这两个文件差异。

开发中保留了一次 dict.update 见证测试失败及修复后的日志；另有单测完成后的 JSON 序列化故障，已从完整日志生成准确摘要。它们不计为通过执行。最终测试和独立报告均成功。

当前唯一开发目录 /home/lhohy/Code/Poseidon，分支 lhy-agent-dsl，HEAD 3493c905。全部既有修改保留；本轮未 commit/push/PR，未安装/下载，Mac历史checkout未改。检查的28个最终验收目录中临时私钥均已清理，原始响应、IR、失败诊断和解密结果保留。

下一条只读命令：

    cat docs/baseline/validation-adapter-acceptance-r171.md

建议下一步先统一新版类型/返回契约及其反例，再按三个上下文逐个开放更多组合；宿主接入继续使用独立 worker，不直接并发调用 run_candidate.inside。
