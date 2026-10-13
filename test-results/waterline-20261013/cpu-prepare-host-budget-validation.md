# CPU Prepare Host 预算与 HostStaged 压力测试

`poseidon_gpu_api_test.cpp` 新增真实 CPU Prepare Host 预算负/正边界，并保留统一执行器、真实 Encode/H2D/Ready/Release。测试遵循现有协议的 Host external input 与 Host final output；先完成输入 H2D，再保留六份普通 Host Encode 结果，在 CPU Prepare/AddCP 后释放它们，最后 D2H。这使 CPU Prepare 消费者成为预算峰值，避免 IO workspace 覆盖待测的边界。没有绕过 Verifier 或新增 Device IO 协议。

设 `bytes = N * Q * sizeof(GpuWord)`。CPU Prepare 原 Host bound 为 `2*bytes + 40*N`，包含同时存活的 pinned upload 后为 `3*bytes + 40*N`。压力图在消费者前持有 `16*bytes` Host resident，负例预算取 resident + 原 bound + bytes/2；正例精确取 resident + 完整 bound。负例必须定位消费者 instruction #8、且 `cpu_prepare_calls` 增量为零；正例检查全部 dense real slots 与全部 imaginary slots，结束后动态 foreground/plaintext credit 与实际 pinned live 均为零。

本地 RTX 4060、Release、architecture 89 结果：

| 参数 | 原 Prepare bound | 完整 Prepare bound | resident | 负例预算 | 正例预算/实际 Host peak | 槽数 | 最大实虚误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N4096/Q7/scale30 | 393216 | 507904 | 1835008 | 2285568 | 2342912 | 2048 | 3.672e-5 |
| N65536/Q32/scale40 | 19398656 | 27787264 | 134217728 | 157810688 | 162004992 | 32768 | 9.127e-7 |

两种参数的负例均在 instruction #8 拒绝，CPU Prepare 调用增量为零；正例动态 front/plain/pinned 均归零。完整默认 GPU API suite 20/20 通过，见 `cpu-prepare-host-bound-full-api.log`。源、库与 binary SHA-256 见 `cpu-prepare-host-bound-config.json`。旧 Device IO 测试构造被现有 IO-2 规则拒绝的日志保留为 `cpu-prepare-host-bound-n4096-q7.log`；该构造未进入正式测试。

`POSEIDON_WATERLINE_HOST_STAGED` 严格接受 0/1，unset 等同于 0。1 给五轮跨卡环通信设置 HostStaged，shared pinned credit 只允许两个 full-context communication reservations 同时存在。共享 Host、pinned、raw budget、账本峰值和实际 pinned 计数均打印。test subclass 经公共 route selector 确认 HostStaged，再调用原 `communicate_async`，累计实际 posts，并检查至少一条通信在返回时确有 pinned allocation。全部四卡时，每个 CPU/GPU/Auto backend 必须完成 20 条 HostStaged posts，并通过四份完整槽数值与不均衡进度检查。没有维护另一套执行路径。

本机只有一张 GPU，env0 与 env1 的 N65536/Q32 兼容运行均通过；env0 输出 `mode=Auto`，env1 输出 `mode=HostStaged`，两者 `expected_posts=0`。这只能验证模式和共享预算的单卡兼容，不证明跨卡 HostStaged 已执行。真正全部 GPU 的验收日志由 188Server 单独产生。

本地复现：

```bash
cmake --build build-waterline-qwen-agent --target poseidon_runtime_gpu_api_tests -j2
build-waterline-qwen-agent/bin/poseidon_runtime_gpu_api_tests
POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
POSEIDON_WATERLINE_HOST_STAGED=1 \
build-waterline-qwen-agent/bin/poseidon_runtime_gpu_api_tests
```

188Server 在已有长 SSH 保活连接存活期间，完成最新源同步与构建后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
POSEIDON_WATERLINE_HOST_STAGED=1 \
/home/xuming/poseidon-waterline-20261013/build/bin/poseidon_runtime_gpu_api_tests \
  > /home/xuming/poseidon-waterline-20261013/logs/final-waterline-host-staged.log 2>&1
```

单卡对照使用 `CUDA_VISIBLE_DEVICES=1` 和 `POSEIDON_WATERLINE_HOST_STAGED=0`。完整默认 suite 已由 19 项增加至 20 项；重复记录 helper 的 `--expected-passes` 现默认 20，已有 19 项库快照的日志与报告保留。
