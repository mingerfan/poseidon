# GPU raw pinned 复用对照

生产缓存回归已由 `gpu-raw-cache-*` 与 188 pressure 日志覆盖。此处是同一小型诊断二进制对每份成本的分解，不是完整模型吞吐测试。两种模式均 N16384/Q40/P8/scale120，10 次 CPU Prepare 和 10 次 GPU Prepare，各检查全部 8192 槽实部及虚部。

188 选择物理 GPU1、pool 最大 1536MiB/初始 64MiB。开始前旧完整模型 PID30187 已不存在，父 agent 的多卡压力测试已经结束。baseline UTC 13:17:48–13:17:58，`--cache` UTC 13:19:09–13:19:19，两次都通过。

| GPU Prepare，10 次 | baseline | `--cache` |
|---|---:|---:|
| pinned 分配次数 | 10 | 1 |
| drain 前 pinned 释放次数 | 10 | 0 |
| drain 前 pinned live 字节 | 0 | 65536 |
| 最终 drain 后 pinned live | 0 | 0 |
| API 提交均值 ms | 1.025306 | 0.219797 |
| 完成回收均值 ms | 0.417778 | 0.012027 |
| GPU Encode event 均值 ms | 0.272096 | 0.270313 |
| raw H2D event 均值 ms | 0.016569 | 0.012924 |

缓存模式固定 Host 和 pinned 各 65536 字节；动态 GPU Prepare Host 65536 字节、pinned 为 0。计数包含 10 次中的第一次懒分配，预热完成后先 drain，不排除首次缓存成本。CPU Prepare 两种模式都是 10 次分配、10 次释放，提交均值分别 27.600 / 28.227ms。缓存没有改变完整 GPU 编码的数学事件时间，主要减少 Host 分配、释放与提交开销。

本地 RTX4060 的同二进制对照也通过：GPU 提交 0.304064→0.094974ms，回收 0.109477→0.001922ms，GPU event 0.412003→0.411715ms；同样分配 10→1、drain 后 0。它是单卡兼容和组件验证，不作为 188 多卡或模型性能证据。

`--cache` 通过 public `validate_waterline_resources` 启用生产缓存契约，再以 public API 分别提交完整 CPU/GPU Prepare。它不执行完整 Waterline 图，不另建执行器。启用 Waterline 会更换实际前台 stream，故缓存模式不使用 caller-stream 的外部 event 测量 AddCP/Rotate；这些已 Ready 普通算子的对照保留在 baseline 及原诊断结果中。

复现：在已构建 GPU API test target 的 build 上运行 `build-and-run-gpu-standalone.py hotpath-components-test.cpp <build> --output-prefix <new-prefix>`；环境为 `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1`，随后用同一生成的 binary 加 `--cache`。188 的实际完整编译链接命令在 `hotpath-cache-188-current-baseline.commands.json`，运行命令、scope、UTC 和断言结果在 `hotpath-cache-188-current-status.json`，source/binary/library SHA256 在 `hotpath-cache-188-current-provenance.txt`。remote 使用独立 basename `hotpath-components-cache-test`，原未缓存诊断和 Nsight binary 均未覆盖。

该对照不能给出完整 Qwen 的端到端加速率；模型仍需完成真实新计划执行与数值验收。
