# 应用控制层 Poseidon CPU 离线回归

本方案尚未批准，执行器默认仅核验计划；只有匹配该绑定的独立授权记录才能启动。

| 案例 | 验收内容 |
|---|---|
| f64_to_f32_c0 | 公开 float64 状态转 float32 后的运算 |
| f64_roundtrip_c2 | float64→float32→float64 与原算子组合 |
| compat_v23_named | 旧来源命名输出及嵌套 None 恢复 |

三个案例复用 r1376 已冻结的源码清单、模型、公开权重、规则 DSL 和独立 reference；旧结果保持。先以当前控制层重新捕获来源，再核对完整请求和原数学图相同。候选逐字节重放，真实 Agent 新生成为 0。

新增检查覆盖接入诊断、来源到生成的累计产物账本、真实 tracing/编译、RuntimePlan/OperatorSpec/bundle、Poseidon CPU 四组逐项数值验收、程序包重载、每包三组新输入和一组重复输入，以及缓存命中时无生成/编译。单独执行调用各自记录产物窗口。新身份以新目录保存，拒绝覆盖旧结果。

- 最多 3 次编译、9 次密态调用；不重试失败。
- 固定 worker 最长 10 分钟，含 Nix 启动总窗口最长 12 分钟；初次任务 180 秒。
- 编译≤2、链接≤1、密态串行，native 内存4 GiB／CPU150秒保持。
- 新产物≤512 MiB、全局≤32 GiB、可用磁盘≥4 GiB。
- 请求保留峰值监控128 MiB；每次新输入执行64 MiB。该监控是检查点计量，非硬配额。
- API/HTTP、安装、下载、网络、后端库和参数修改均为0。
- 数值门限为 1e-5+1e-4*abs(reference)。
- 失败/超限保留现场和已执行调用，未运行项不记通过。

SDK：4469dc8dea04cdee31c354f728a830d23f447d702836d2d8e8a057c9014e9c86

方案绑定：ed7765a63eb36d64030a8b2f25994ae36d0ac320f095e022a6f0b0c4203f0a5e

只读计划入口，在固定 ARM Nix 环境中：
~~~bash
python3 -B scripts/baseline/benchmarks/tools/application_admission_budget_v1/offline_cpu.py
~~~

获得匹配的授权后才使用 --run-approved。旧20分钟/36项授权已结算，此方案是新源代码身份下的三个代表回归。
