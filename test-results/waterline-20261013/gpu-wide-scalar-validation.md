# uint64 GPU scalar 验证

`GpuEvaluator::multiply_scalar` 原签名已是 uint64，但拒绝大于 GpuWord 的输入，随后 handler/launcher/kernel 都接收32位 scalar。现在全链路保持 uint64，在 kernel 用现有 `floor(2^64/q)` Barrett 常数先将完整 scalar 除实际 q 取余，再将32位 residue 与原系数相乘。没有增加 kernel、stream、缓存或分配路径，原 uint32 输入使用相同模乘过程。

同时修复原地调用：旧 generic destination helper 会在 `destination==source` 时先分配并覆盖源，再建立 source view。scalar 每个 thread 只读写同一个位置，故原地调用直接保留源 storage，无需临时 ciphertext；独立输出仍使用现有分配函数。

本地真实 GPU 测试 `src/poseidon/tests/runtime_api/poseidon_gpu_scalar_test.cpp`，N4096/Q7（实际30位 primes）/scale40，pool 最大512MiB、初始64MiB。8个 scalar 分别为 0、1、17、UINT32_MAX、2^32、2^34+123、2^63+129、UINT64_MAX，每个有原地和独立输出两种，共16组。

- 974848 个 RNS 字全部等于独立 CPU 128位 `(original_word * uint64_scalar) % actual_q`，包括独立 stream 探针。
- 每组全部2048复杂槽解密通过；将实际解密值除 scalar 与原 ciphertext 的解密值比较，最大误差1.04083e-16。输入及输出始终保持原 scale，没有用被 retag 的 CPU Rescale 结果当参考。
- caller PTDS 由不调用 CUDA 的 callback 阻塞；另一条显式 nonblocking stream 完成原地宽 scalar 与 D2H，gate仍未释放时全字数值通过。
- 最大 scalar 为18446744073709551615；uint32的0/1/17/MAX能力同时覆盖。

直接 `GpuUploader` 的 packed Host copies 使用其 CXX CUDA default stream；它们的 default 与 `.cu --default-stream=per-thread` 不同。测试在准备输入阶段显式等 legacy 和 PTDS，再发另一 stream 上的 scalar。初版等待不完整的失败日志保留为 `gpu-wide-scalar-local*.log`，不能把其中的上传 race 报告为 uint64 Barrett 数学失败。

隔离负控只在 build 内复制 evaluator，恢复旧 alias 重分配块；新 uint64 kernel、正确的两条 default stream 输入 Ready 等待与数值 oracle 全部保留。负控再次在 scalar17 出现真实 RNS mismatch，exit1，独立证明 alias 问题；没有改变生产源码。负控 patch、实际 compile/link 命令及日志分别为 `gpu-wide-scalar-alias-negative-control.patch`、`.commands.json`、`.log`。可复制生产 evaluator 到命令指定的隔离目录后应用该 patch 重现。

正控实际命令见 `gpu-wide-scalar-local-ready-both-defaults.commands.json`，完整日志同前缀 `.log`，参数、source/binary/library SHA256 与失败保留清单见 `gpu-wide-scalar-local-status.json`。重现调用已有 standalone builder：

```sh
cmake --build build-waterline-qwen-agent --target poseidon_runtime_gpu_api -j2
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python3 \
  test-results/waterline-20261013/build-and-run-gpu-standalone.py \
  src/poseidon/tests/runtime_api/poseidon_gpu_scalar_test.cpp \
  build-waterline-qwen-agent --output-prefix <new-prefix>
```

188 单卡对应 `CUDA_VISIBLE_DEVICES=1` 并使用远端已重建的 build，不能把本地 arch89 binary 复制给 V100运行。本证据只验证该数学原语；新的 API scale 对齐、实际Q50计划与全部 GPU/完整模型验收由相应测试另行证明。
