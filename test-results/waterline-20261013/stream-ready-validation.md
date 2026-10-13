# CUDA stream 与 Ready 验证

本记录是本地 RTX 4060、Release、CUDA architecture 89 的真实 CUDA 小图验证，不能代替 188Server 或完整 Qwen 的结果。设备 UUID、驱动及被测生产源码 SHA-256 见 `stream-ready-validation-config.json`。测试源仅通过公共 Poseidon API 调用计算与 Ready 等待。

`addcp-ptds-ready-test.cpp` 先预热 AddCP，并将待复用输出的 c1 写成零；在 caller per-thread default stream (PTDS) 放入一个不调用 CUDA 的 80 ms host callback，然后提交 Waterline foreground AddCP。消费者在 Ready 完成后，通过另一个 nonblocking stream 将 c1 拷入预分配 pinned buffer，逐 word 核对原密文 c1，并确认此时 PTDS callback 仍阻塞。这样覆盖了只在实际执行 stream 发布 Ready、却将未修改 c1 的 D2D copy 提交到 PTDS 的竞态。

当前实现 5 轮通过；每轮检查全部 28,672 words，c1 差异为零。Ready 与 snapshot 在 0.030–0.033 ms 完成，PTDS 的 80 ms gate 仍阻塞，结果见 `addcp-ptds-ready-test.log`。

`build-addcp-ptds-negative-control.py` 将同一生产 CUDA source 复制到 build 目录，仅把该 helper 的 `cudaMemcpyAsync(..., gpu_execution_stream())` 改回旧 `cudaMemcpy(...)`，再单独编译此 object、链接同一 public test。它不修改工作区生产源码或当前库。第一轮 Ready 提前完成时，28,672 个 c1 words 全部不一致，见 `addcp-ptds-negative-control-report.json` 与 `.log`；这是预期失败，证明测试能够稳定捕获该缺陷。

`consumer-wait-test.cpp` 的六个延迟依赖案例重新链接当前库后全部通过：Negate 加同一 foreground stream 后续算子、AddCP、AddCP reuse、Rotate、Relinearize、Rescale。每个案例检查全部 2,048 real/imaginary slots，最大误差 3.013e-7；异步提交 0.018–0.157 ms，CUDA dependency interval 为 80.150–80.284 ms。已 Ready 输入不重复计数，同一实际 stream 不增加等待 interval。见 `consumer-wait-stream-complete-report.json` 与 `.log`。

之前 `consumer-wait-regression-diagnostic-01.log` 的全零数值失败、`-02.log` 的 130 秒未完成和 `consumer-wait-regression-hang-stack.log` 的 ptrace 拒绝均保留。新结果不覆盖旧失败；上述正负对照直接证明 AddCP c1 的 stream/Ready 缺陷及修复。旧 hang 没有可用 stack，不能据新重跑通过独立断言其根因；当前完整 API suite 的重复结果另见 `stream-complete-api-repeat-status.json`。

architecture 89 完整 API suite 连续运行 10 次，每次 19/19，通过总计 190 项；每轮 12.98–14.99 秒，没有数值失败或 timeout。日志为 `stream-complete-api-repeat-00.log` 至 `-09.log`。这覆盖 146 个 kernels 和 8 个 D2D/memset helpers 的执行 stream 修复，使用本次开始时的 API/GPU 库快照；配置 JSON 同时记录库 SHA-256。之后其他实现修改需要使用新 prefix 追加验证。

随后接入最新 completion reclamation 的 PTDS guard / caller device 恢复，以及 keyswitch helper 中剩余的 null-stream copy 修复，重新构建并追加了 `stream-reclaim-*` 结果：单次完整 19/19；六个 80 ms 依赖案例全部通过，最大误差 7.368e-7、CUDA 等待 80.051–80.293 ms；PTDS 正控 5 轮 c1 全部一致，Ready/snapshot 0.027–0.036 ms 完成。最新库再连续运行 10 次完整 suite，每次 19/19，总计 190 项，没有失败或 timeout，每轮 12.99–14.89 秒。配置与库 hash 见 `stream-reclaim-validation-config.json`，重复状态见 `stream-reclaim-api-repeat-status.json`。这些结果证明最新改动的本地单卡兼容性；跨卡 reclamation 的正确性仍由 188Server 全部 GPU 实测独立验收。

从仓库根目录复现：

```bash
cmake --build build-waterline-qwen-agent --target poseidon_runtime_gpu_api_tests -j2
python3 test-results/waterline-20261013/build-and-run-gpu-standalone.py \
  test-results/waterline-20261013/addcp-ptds-ready-test.cpp build-waterline-qwen-agent
python3 test-results/waterline-20261013/build-addcp-ptds-negative-control.py \
  --build-dir build-waterline-qwen-agent
python3 test-results/waterline-20261013/build-consumer-wait-test.py \
  --build-dir build-waterline-qwen-agent
build-waterline-qwen-agent/bin/consumer-wait-test \
  test-results/waterline-20261013/consumer-wait-stream-complete-report.json
python3 test-results/waterline-20261013/repeat-current-gpu-api-tests.py \
  --binary build-waterline-qwen-agent/bin/poseidon_runtime_gpu_api_tests \
  --prefix test-results/waterline-20261013/stream-complete-api-repeat-new
```

最后一条命令拒绝复用已有 prefix，以保留此前失败和运行记录。188Server 使用该测试时遵循其单卡物理 GPU 1、或全部 GPU 的要求；本记录只运行本机唯一 GPU。

新增 `--output-prefix` 可保存 standalone 的新结果而保留既有正负对照，例如：

```bash
python3 test-results/waterline-20261013/build-and-run-gpu-standalone.py \
  test-results/waterline-20261013/addcp-ptds-ready-test.cpp build-waterline-qwen-agent \
  --output-prefix test-results/waterline-20261013/stream-reclaim-ptds-new
```
