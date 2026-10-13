# Waterline 实现与 188Server 验证

状态核对至2026-10-13 20:36（北京时间）。当前已推送版本为Poseidon `69217e3e`、ckks-runtime `f41b930`、DaCapo `ad35c87`。188最新单卡物理GPU1、全部四卡默认通信、全部四卡强制HostStaged的真实压力均通过CPU/GPU/Auto三模式。最终完整Qwen3-0.6B BSGS Auto正在全部物理0/1/2/3运行，完整输出和独立数值验收尚未完成。

当前已通过编译器 10 项回归、运行时 8 项回归与本地 GPU API 20 项回归。Waterline 使用后期 Prepare 改写、完整消费者组准入、每设备水位及共享 Host/pinned/raw 账本，并复用前台计算和通信执行函数。没有自动批末 Fence。HEFT 源码没有改动。

本地真实 CKKS 生产路径已执行 CPU-only、GPU-only、Auto，N=65536、Q=32 个 30-bit 素数、scale=2^40，32768 个实数槽。图有五次 AddCP，使用真实加密输入与完整 dense 明文，并按全部槽位与明文算术参考比较。三种模式 max absolute complex error 均为 9.21178e-7，容差预设 1e-4；结束时计费 Device 明文和 pinned live 均为零。Auto 在这一单卡图中选择 CPU，不把该结果当作混合路径已覆盖。

对应日志：`compiler-regressions.log`、`core-tests-commit.log`、`core-waterline.log`、`local-waterline-dense-current.log`。清理引用依据见 `cleanup.tsv`。临时失败和调试构建日志未作为通过证据。

```sh
cmake -S third_party/ckks-runtime -B build-waterline-core \
  -DCKKS_RUNTIME_BUILD_TESTS=ON -DCKKS_RUNTIME_BUILD_EXPERIMENTS=OFF
cmake --build build-waterline-core -j6
ctest --test-dir build-waterline-core --output-on-failure --timeout 60
ctest --test-dir third_party/ckks-runtime/third_party/dacapo/build/nix --output-on-failure
cmake --build build-runtime-gpu-api-release --target poseidon_runtime_gpu_api_tests -j6
POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 OMP_NUM_THREADS=1 \
build-runtime-gpu-api-release/bin/poseidon_runtime_gpu_api_tests
```

188压力已通过，完整Qwen3-0.6B验收仍未完成。本地旧Qwen25 24层DSL配置与目标不符，不能复用作完整目标模型。官方目标revision `c1899de289a04d12100db370d81485cdf75e47ca` 为28层、hidden size1024、head dim128、16 Q heads、8 KV heads、词表151936、Q/K norm、共享embedding/LM head。固定revision权重SHA256为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，311张量已核对；权重未提交到Git。客户端查embedding后加密tokens `[9707,11]`、positions `[0,1]`；目标为2输入、261输出（完整151936 logits、28层全部两位置KV）。

GPU Encode 接受实数槽及上下文模数范围内的有限 FP64 系数；宽整数按 significand/exponent 作 RNS reduction。常数 scale 40/60/80/120 与 CPU 每个 RNS word 一致。配置 reserve 已核对实际参数表、单份 owning key backing、编码器表和持久 workspace 上界；RMM 有硬上限，并分别报告实际活跃分配和保留池空间。这里不声称完整模型加速。Native Boot 的 Waterline workspace 查询暂未接通，普通显式 Native Boot 能力保留。CPU decrypt/re-encrypt Boot 的安全与数值语义必须与原生 Boot 分开报告。

后续资源阶段的有效日志为 `core-tests-validation-hook.log`（7/7）、`key-backing.log`（Q=3/5/7共享key且全槽与CPU一致）、`gpu-api-h2d-final.log`（19/19）、`h2d-timing.log`、`gpu-encode-timing.log`、`local-waterline-metrics.log`。后者是真实 managed-pool N=65536/Q32/scale40 图，三模式最大误差8.64e-7，实际持久表约1.15GB，RMM活跃峰值约1.19GB，保留池1.75GiB、硬cap3520MiB；结束pinned live和计费Device明文为0。时间指标是本机本次测试，不能据此推断V100或完整模型性能。

`qwen-runner-metrics-smoke.log` 仅为2输入/2输出小图，证明多输出入口、固定容差与事件计量接通，不是完整Qwen。旧N65536单卡/四卡容量拒绝和旧BSGS编译日志保留。最新N16384最终BSGS四卡图见 `qwen3-bsgs-final-all-compile.log`（exit0、29分55.87秒、RSS137225352KiB）；binary1182542616 B、bundle61298262016 B、values9692017、instructions19364300。编译和导出完整模型不等于CKKS数值通过，eager对象估计报告不能替代Waterline实际峰值。

2026-10-13 后续 stream 阶段已通过真实验证：所有计算 kernel 与 D2D/zero helper 使用实际执行 stream，跨设备回收暂用各设备合法 PTDS，并恢复调用者当前设备。`stream-ready-validation.md` 包含 AddCP c1 的确定性正/负对照；当前 helper 在 PTDS 阻塞时仍全部 28672 words 正确，旧 helper 全部错误。最新单卡 CUDA 库 `stream-reclaim-api-repeat-status.json` 连续 10×19 通过，旧数值失败和无法取 stack 的 hang 日志保留；不为旧 hang 虚构独立根因。

188Server最新真实小图在当前版本使用N65536/Q32×30-bit/scale40，五轮各卡独立AddCP和ring，比较全部32768个complex槽。表中三列顺序均为CPU/GPU/Auto：

| 配置与日志 | 最大绝对误差 | forward ms | 进程time-v / RSS峰值 |
| --- | --- | --- | --- |
| 物理1，`188-final-waterline-single.log` | 均8.96891e-7 | 125.398 / 35.5416 / 115.524 | 34.29s / 3605908KiB |
| 全部0/1/2/3，`188-final-waterline-all.log` | 均9.3044e-7 | 564.622 / 217.861 / 356.733 | 47.96s / 4006584KiB |
| 全部0/1/2/3强制HostStaged，`188-final-waterline-host-staged.log` | 均9.71521e-7 | 1026.95 / 759.282 / 796.408 | 49.39s / 4005672KiB |

三份日志均exit0，结束各卡remaining front/plain和pinned live归零。四卡每模式四个输出、三张peer卡真实完成均早于延迟GPU0；默认/HostStaged的Auto均新增1 CPU+3 GPU Prepare。强制HostStaged每后端实际post20次，20次post时观察到pinned；Host/pinned/raw预算318767104 / 50331648 / 1048576 B，账本峰值268435456 / 50331648 / 1048576 B。实际pinned allocation进程累计峰值33554432 B，区别于账本承诺。各卡RMM reserved/cap均3690987520 B、活跃分配累计峰值最高1208922248 B；RMM live保留了持久资源，不能要求归零。API/event计时与time-v墙钟分别报告，不能把不同图或连续累计counter直接相减为模型加速。

较早 `188-peer-completion-*` 对应 `b551ee2f`，其中GPU0已有其他作业的20280MiB outside快照单独记录并保留；它不是当前进程或整个run峰值。最新日志的initial outside和RMM仍分别报告，未将其他用户占用归给模型。

完整模型N16384旧矩形对角线四卡编译见 `qwen3-n16384-all-compile.log`，旧单卡容量拒绝见 `qwen3-n16384-single-capacity.log`。旧四卡Waterline基线峰值约4.68/6.22/9.37/11.55GB，Prepare1417857、组9726229；这些是旧图编译基线，不是最终运行实际峰值。首次runner启动的生产binary scanner漏Prepare tag5已修复，正式file reader及直接编译往返通过，保留 `qwen3-n16384-binary-scan-failure.log`。第二次OperatorSpec说明字段不合schema，仅重绑定合法spec摘要，metadata外1356009044 B body SHA256仍为 `9fcdbfcb46bab88f92a3d90c3cfef4e947bf470632e03885e81e3801a8c9f709`。旧GPU-only PID22555已由主执行代理核对own cmdline后SIGTERM，以优先最终BSGS/Host修复/100µs版验收，见 `qwen3-full-gpu-n16384-v3-superseded.json`；未完成全部输出验收，不是timeout/deadlock诊断，time-v同时出现signal15和exit0不能算完整模型通过。

最新最终BSGS单卡容量检查见 `qwen3-bsgs-final-single-capacity.log` / `188-final-single-capacity-summary.json`：instruction #5388456 front26212827136、plain/workspace3145728、capacity26843545600、waterline16642998272 B，exit1、7分41.38秒、RSS129037504KiB。这只说明当前预算/基线配置不足，不证明全部单卡配置不可行；最终四卡图已通过编译。

CPU Host/pinned资源修复已通过：CPU Prepare的uint64 Host plaintext和uint32 pinned上传源同时存活，二者都计入Host，pinned另有子预算。degree8真实编译CPU/Auto Host480拒绝、512通过，GPU Host64仍通过；修复前误接收的反例保留为 `compiler-cpu-pinned-before-fix.log`。本地N4096/Q7及N65536/Q32真实CUDA边界见 `cpu-prepare-host-budget-validation.md`、`cpu-prepare-host-bound-full-api.log` 及legal/mode0/mode1日志。最新188三份压力日志均在消费者#8提交前拒绝Host157810688 B（CPU Prepare calls新增0），Host162004992 B足量全32768实虚槽通过，误差单卡/默认四卡/HostStaged依次9.18525e-7 / 9.2084e-7 / 9.17894e-7，动态front/plain/pinned归零。全四卡强制HostStaged已按上述实际post/pinned/数值验收完成。

`scripts/qwen3_dsl/validate_decoded.py`核对fixture、signature、参考和runner报告绑定，分别汇总完整151936有效logits与全部KV的最大误差、RMS、cosine、top1，检查所有虚部和零尾部，固定atol=0.001/rtol=0.0001。`checker-test.log`的20项使用明确标注的模拟decoded，只证明映射/拒绝，不是模型CKKS结果。当前独立NumPy exact/approximate参考logit max3.787683944267428e-9、RMS7.210505850742908e-10、top1均358。N16384/Q40/P8、图scale120及Host秘密钥辅助decrypt-reencrypt refresh使用security level none，不声称标准安全FHE或原生Boot。

公开 `build-188.sh` 已在188执行：`188-configure-reproduce.log` configure/generate完成，`188-build-reproduce.log` 正式API和Qwen runner target完成、time-v exit0（该次为增量复现0.61s、RSS15744KiB，不是全量冷构建）。hash见 `188-latest-build.status`。复现时先保持一条独立长SSH，再运行：

```sh
POSEIDON_188_WORKSPACE=/home/xuming/poseidon-waterline-20261013 bash build-188.sh
POSEIDON_188_WORKSPACE=/home/xuming/poseidon-waterline-20261013 bash run-188-acceptance.sh
```

脚本位于本目录，部署到服务器时保持上述workspace结构和离线依赖；验收脚本run label目录必须是新目录。`run-188-acceptance.sh` 已完成前三项压力，最终Auto PID30187正在 `logs/qwen3-bsgs-auto-final` 实际前向。完整模型覆盖截至20:36：CPU-only未启动；GPU-only旧矩形图被superseded且最终BSGS GPU-only未启动；最终BSGS Auto正在全部四卡运行，尚无decoded/report/exit-status及独立验收结果。小图三模式不能补足完整模型三模式覆盖。

服务器大型artifact保存在 `/home/xuming/poseidon-waterline-20261013`；远程期间始终保留独立SSH长连接，tmux不替代保活。`monitor_process.py` 按PID采样driver内存及RSS，区间采样不是连续峰值，晚启动披露首个timestamp。runner记录main到首个/全部输出、setup/init/forward及保存；外部time-v计进程冷启动。CSV duration只是API提交时间，trace字节数不表示完成百分比。仍需完整模型实际输出、独立数值报告、实际后端覆盖和最终资源/等待/收尾汇总。

执行器无进展轮询由1ms缩至100µs，仍按原Ready/完整组/顺序检查；没有新调度路径或配置。`core-short-poll-tests.log`核心8/8，`core-short-poll-gpu-*`完整20/20及N65536/Q32 mode0/1各一次通过，源码/库hash保持一致。它可能增加CPU查询开销并改变Auto比例；本机host sleep探针与回归耗时不能外推188或完整模型加速。最终计划使用含Host/pinned修复的新编译器，产物目录为服务器`qwen3-plans-bsgs-final/gpu4`。
