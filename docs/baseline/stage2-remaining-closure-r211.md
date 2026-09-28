# 剩余失败集中诊断与修复 r211

## 结论

真实 Agent 未通过数仍为 **26**，本轮新增 Agent 通过为 0，付费调用为 0。没有把人工表达计入 Agent 成功。

这 26 项现均有合法人工表达，分别通过 16 组样本与独立数学/PyTorch reference 的明文差分，然后在真实 Hecate tracing 后被固定 Dacapo 编译器拒绝。它们尚无本轮密态成功证据。
这是当前表达及固定配置的实测阻塞，**不是证明所有等价表达均不可能执行**。原 Agent 终态仍为 16 编译、7 静态、1 Provider、2 调度中断，未回写历史记录。

## 本轮实际修改

- validation_adapter.py 接入有界、只读的 Earth 编译失败解释器 earth_failure_diagnostic.py。依据校验过的固定 profile 和固定版本类型推导条件，不改变接受判定或参数。
- 结构化诊断保存在本地 attempt 报告；对模型仍使用原有受限 diagnostic 字符串，不扩大 Provider 字段白名单。
- 修复人工诊断构造器 packed_graph_diagnostic.py：补零 concat 后的逻辑张量可能超过槽周期；negate 保持逻辑单元映射，避免把尾部折回槽 0。不支持的超容量物化明确拒绝。此构造器不是 Agent 输出改写器。
- 增加精确有理数 Chebyshev→幂基转换与 Estrin/分配乘法人工探针。保留全部系数；浮点字面量转换由双参考和冻结门限检查。未推广为生产默认算法。

## 已确认的编译约束

固定源码 third_party/dacapo/lib/Dialect/Earth/IR/EarthDialect.cpp:314 附近的 MulOp 推导要求操作数 level/shape 对齐，且左操作数满足：

consumed_level × rescalingFactor + scale <= bootstrapLevelUpperBound × rescalingFactor

这里是 **Earth 编译器的类型条件**，不是重新引入 60×level 的 SEAL 容量检查。SEAL artifact gate 仍使用真实模数乘积。
Earth level 是已消耗层数；HEVM level 是剩余数据模数数量；二者不能混用。
典型实测 12×60+90=810 > 13×60=780，另有 13×60+60=840 > 780。没有绕过条件或更改 scale 元数据强行执行。

上游 poly/Func.py 的 HE_ReLU 在前两段多项式之后调用 bootstrap；HE_Max/HE_MaxPad 也调用 bootstrap。
当前 SEAL HEVM 没有真实 bootstrap，因此不能把高层 helper 的 bootstrap 路径视为已恢复。此事实不等于所有纯多项式等价实现均不可能。

## 验收

| 范围 | 结果 |
|---|---|
| Python 相关测试（包括隔离/传输 mock 测试） | 201 passed / 0 failed / 0 skipped |
| 旧请求精确重建 | 2208 / 2208 |
| 人工原模型表达静态检查、双参考差分 | 26 / 26，每例 16 组 |
| 上述表达真实 frontend tracing | 26 / 26 |
| 上述表达真实编译 | 0 / 26；全部编译累计 scale 条件拒绝 |
| 上述 26 项密态执行/解密 | 未执行，不能算 pass |
| Linear self-test | 真实编译、SEAL 密态、解密及数值比较通过 |
| mlp4x4x2 golden | 真实 Linear→square→Linear 密态链路通过 |
| 旧 SiLU Agent 源码原样回放 | 密态数值通过；不是新 Agent 生成 |
| 新付费请求 | 0 |

冻结门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。

24 个最终人工编译阻塞证据来自 r206/r207；另两个布局修复后的证据来自 r210，各自源码绑定保留。
Linear/MLP/SiLU 回归属于 r208 绑定；最终 r210 仅修正编译失败反馈字段，另以两次真实编译失败验收反馈集成。未把不同绑定伪称同一批。

93 个受保护的编译器/SEAL/HEVM 源码和二进制哈希保持不变；未安装、下载、修改安全参数、model、reference，未 commit/push/PR。ARM 执行；x86 本轮未执行。

## 剩余原任务与逐例证据

| 任务 | 历史 Agent 终态 | 新人工表达 | 编译累计值 / 上限 |
|---|---|---|---|
| free_bench_helper_0032 | compiler | estrin | 810 / 780 |
| free_bench_helper_0033 | harness_interrupted | estrin | 810 / 780 |
| free_bench_helper_0034 | harness_interrupted | estrin | 810 / 780 |
| free_bench_helper_0035 | compiler | estrin | 810 / 780 |
| free_bench_helper_0036 | compiler | estrin | 840 / 780 |
| free_bench_helper_0037 | compiler | balanced-chebyshev | 810 / 780 |
| free_bench_helper_0038 | compiler | estrin | 840 / 780 |
| free_bench_helper_0039 | static_check | balanced-chebyshev | 810 / 780 |
| free_bench_helper_0040 | static_check | estrin | 810 / 780 |
| free_bench_helper_0041 | compiler | estrin | 810 / 780 |
| free_bench_helper_0042 | static_check | estrin | 810 / 780 |
| free_bench_helper_0043 | provider | estrin | 810 / 780 |
| free_bench_helper_0044 | static_check | estrin | 810 / 780 |
| free_bench_helper_0045 | static_check | estrin | 810 / 780 |
| free_bench_helper_0046 | static_check | estrin | 810 / 780 |
| free_bench_helper_0047 | compiler | estrin | 810 / 780 |
| free_bench_helper_0104 | compiler | weighted-estrin | 840 / 780 |
| free_bench_helper_0105 | static_check | estrin | 840 / 780 |
| free_bench_helper_0106 | compiler | estrin | 840 / 780 |
| free_bench_helper_0107 | compiler | estrin | 840 / 780 |
| free_bench_helper_0108 | compiler | estrin | 840 / 780 |
| free_bench_helper_0109 | compiler | estrin | 840 / 780 |
| free_bench_helper_0110 | compiler | estrin | 840 / 780 |
| free_bench_helper_0111 | compiler | estrin | 840 / 780 |
| free_supplement_semantic_context_331124fd5981bd48a0c2971f | compiler | estrin | 795 / 780 |
| free_supplement_semantic_context_89458ebf15d877d81d6b8b2d | compiler | estrin | 795 / 780 |

## 保留的本轮失误与修正

- r207 的 0037/0039、r208 稳定基表达明文不一致，原因是诊断构造器过早物化超周期 concat；原证据保留，修复后 r210 各 16 组通过，最大明文误差约 6.66e-16。
- r209 新增公共结构化反馈字段触发 unapproved_feedback_fields；这是本轮适配器接线错误，非 Agent 失败。已恢复原白名单，结构化数据只落本地；r210 通过真实失败反馈流。
- r210 首次探针启动误用 work root，Nix 离线预检拒绝，尚未编译。保留 stable.log；正确路径使用 stable-retry.log，无安装或付费调用。
- 上述失败不计 pass，不覆盖旧报告或请求。

## 下一步及权限边界

优先审查/修复编译器 scale/level 调度，仍固定安全参数、数学模型、reference 和误差门限。
若仅靠调度仍无可行解，应继续保留后端阻塞；不删多项式项，不降低近似次数，不用解密再加密冒充 bootstrap。
此前用户明确要求“不修改编译器”；本轮已达到该边界。修改 Dacapo 调度需另行授权。没有离线密态成功依据前，不追加这 26 项付费重跑。
阶段三的接口/分解机制可独立开发，但第二阶段这些项目不能宣布通过。

复核命令（ARM Ubuntu checkout）：

    cd /home/lhohy/Code/Poseidon
    python3 -B scripts/baseline/benchmarks/tools/audit_remaining_failures_r211.py --verify

机器可读总账：docs/baseline/stage2-remaining-closure-r211.json；含逐例原模型绑定、源码哈希、编译日志、历史终态和本轮手工证据。
