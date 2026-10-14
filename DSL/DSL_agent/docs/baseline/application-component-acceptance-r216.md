# 面向应用的 Agent 架构改造验收 r216

已建立第一版应用组件边界：**提交任务 → 查询状态 → 取得带验收等级的 DSL 程序包，或明确失败**。
生成与自动修复复用现有组件和同一个 Provider 会话，没有新建另一套 Agent。
完整接口、架构差距与限制见 application-agent-component-v1.md。

## 已确认实现

- 新 application_policy / application_component / application_backend 提供能力描述、确定性上下文选择、持久化任务、预算账本、取消/超时、保守恢复与包复用。
- 原 run_feedback_loop 增加可选控制回调，旧 CLI 不启用时保持原语义；retry_policy 在应用入口实际控制是否继续修复。
- 检测重复源码和重复失败；Provider 每次 HTTP 尝试先预留任务额度；不中断后自动重发未知调用。
- 原 ValidationContext、qualify、审计器和程序包增加显式 compiled 等级，默认 numerical 不变。
- 应用返回不包含私有 reference、输入数组、环境路径或凭据。开发探针、Benchmark、历史批次不进入应用请求路径。
- 模型、权重、完整要求、验收等级与依赖身份相同的任务可复用已验证包，复用前校验完整性。

## 验收结果

| 类别 | 结果 |
|---|---|
| 相关自动测试 | 249 passed，0 failed，0 skipped |
| 旧请求精确重建 | 2208 / 2208 |
| 编译级 MLP 程序包 | 静态、真实 tracing、编译、产物审计通过；未执行密态/解密 |
| 编译级包重新加载 | 重新编译与产物审计通过；不能自动升级成数值通过 |
| 数值级 MLP 程序包 | 真实 Linear→square→Linear、SEAL 密态、解密和四组逐项数值比较通过 |
| 数值级 MLP 误差 | 8 个输出值，max abs 9.0762288657e-10 |
| 数值级包重新加载 | 独立重编译和密态验收通过，max abs 3.91863517329e-09 |
| 两等级重复提交 | 均直接复用，额外生成次数 0 |
| 旧 Linear self-test | 既有故障→修复→密态数值成功链路通过 |
| 新真实 Agent 评测 | 未运行；付费调用 0 |

249 是组件及兼容测试项数，不是 249 个密态模型。应用生成路径使用 scripted_replay 测试夹具；不能记为新 Agent 成功。
数值门限始终为 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。密态验收只证明约定有限输入，不是形式化证明。

## 保留的限制与推断

- 当前宿主仍需调度独立 Linux/Nix worker，在主线程调用 run；不支持把该方法直接当线程安全 SDK。
- 后端下面暂保留单例 CLI 兼容桥和 WORK 环境；下一步可提取 QualificationSession/RunContext。应用任务本身不依赖任何实验批次编号、历史报告或人工修复脚本。
- 程序包是带依赖身份的可重编译 DSL 源码包，不是完整常驻密态推理服务。
- 本轮没有评估新的付费提示策略收益，也没有付费应用链路验收；旧 Agent 来源证明条件未放宽。
- 取消、超时、工作进程丢失保留预算；未知调用不自动重发。未宣称任意 SIGKILL/断电下的进程树及密钥即时清理保证。
- ARM 实际执行；x86 配置保留，但本轮未在 x86 执行。
- 之前 26 项失败维持原状态，本次架构验收不能替代它们的修复。

## 变更与证据

修改限定在 Agent 组件、验证适配、Provider 预算钩子、包接口、相关测试与文档。
93 项 Dacapo/SEAL/HEVM 受保护源码及二进制哈希不变；固定模型、reference、安全参数、数值门限不变。
没有安装、大型下载、付费调用、commit、push 或 PR；原分支与本地修改保留。

首次编译级集成测试暴露审计分支的 period 初始化遗漏，已修复；失败证据 acceptance-initial.json/log 保留。
r213、r214、r215 为不同代码绑定的开发验收；本报告只以最终 r215 的验收和测试作为当前结论，未覆盖旧结果。

机器可读报告：docs/baseline/application-component-acceptance-r216.json。
原始证据：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/application-component-r212。
当前绑定：9a872cd25b70d4c3bbc3ee21b762aa058d307e0d7b3d5116a8d8c035a51793d1。

复核命令：

    cd /home/lhohy/Code/Poseidon
    python3 -B scripts/baseline/benchmarks/tools/audit_application_r216.py --verify

下一项最小工程任务：继续抽离底层 QualificationSession 和显式 RunContext，使用已建立的应用接口与相同验收标准，不启动全量 Benchmark 或新增付费测试。
