# 公开常量算术 Poseidon CPU 离线验收 r1385

当前方案未启动，候选均为固定规则生成，API／HTTP 为 0。

范围：公开加、减、乘、除、取负各 3 个新上下文，加 f64_to_f32_c0、f64_roundtrip_c2、compat_v23_named 三个原回归，共 18 项。模型、公开权重、规则 DSL、来源、请求、当前 SDK 和后端均已冻结。

每项检查真实 Hecate tracing／编译、RuntimePlan／OperatorSpec／常量 bundle、Poseidon CPU 密态执行、解密和四组输入逐项比较，以及程序包导出与重载。add_c0、multiply_c2、negate_c1 再检查三组新输入、一组重复及缓存复用。

| 边界 | 上限 |
|---|---|
| 编译／密态调用 | 18／24 |
| 输入组次 | 84 |
| 完整窗口／固定 worker | 45／40 分钟 |
| 首次任务 | 180 秒 |
| 编译／链接／密态并发 | 2／1／1 |
| native 内存／CPU | 4 GiB／150 秒 |
| 新增产物／全局／余量 | 512 MiB／32 GiB／至少 4 GiB |
| API、安装、下载 | 0 |

误差门限为 `1e-5 + 1e-4*abs(reference)`。现有网络、后端库、参数、模型和 reference 保持。无关候选验证失败保留并继续独立案例；完整性、环境、全局资源或时间故障停止相关批次。额度结束保留现场，剩余案例列为未运行。

旧 r1382 的 3 项／9 次密态额度已结算；本方案使用新的源码绑定，需要本批授权。真实 Agent 评测另计。

[精确案例、哈希及预算](agent-public-arithmetic-offline-proposal-r1385.json)。输出位置：
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-arithmetic-cpu-v25-r1385`。

只读复核，在 ARM checkout 执行：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=$HOME/poseidon-work \\
  python3 -B scripts/baseline/benchmarks/tools/public_arithmetic_expression_v1/run_pinned.py verify_offline.py
~~~
