# Rescale 数值修复

旧 API 按实际 Q prime 连续 Rescale 后，直接把 scale 改为 `2^target`，没有调整密文系数。独立 q_proj 图测得约 1.090078 倍的幅值，完整模型的旧运行因此不能作为数值验收；PID30187 已按用户要求停止，没有产生完整输出。

CPU/GPU 现在共同按实际 prime 选择原生 Rescale 次数，用正 uint64 整数乘密文系数完成 scale 对齐，再以 ModSwitch 消耗剩余物理层。目标 scale 必须低于实际剩余模数空间，原生 scale 至少保留 `2^40`，整数对齐相对舍入误差不超过 `1e-9`；不满足时显式报错。这些条件不是通用的噪声或幅值证明，也不涉及安全参数证明。

编译器 d5ed984 为每次逻辑 scale 降低额外保留至少30个 nominal modulus bits，120-bit 逻辑步长对应5个30-bit物理层。HEFT 源码没有修改。GPU scalar 全链路保留 uint64，并修复原地 scalar 调用覆盖输入 storage 的问题。

本地验证：

- CPU API 9/9：N16384/Q50、输入 scale230/240 到120、计划5/6层，全部8192复杂槽；最大误差约 `1.3e-11`。另保留50-bit CPU prime 的单层近1对齐，以及低精度显式拒绝检查。见 `cpu-rescale-normalization-tests.log`。
- GPU API 20/20：独立普通/多层 Rescale、MulCC/Relinearize/Rotate 与全部槽的稠密运算参考；见 `gpu-normalized-rescale-api.log`。
- 正式 CTest `poseidon_runtime_gpu_scalar_tests`：16组原地/独立输出，974848 RNS words 与独立128-bit模乘逐字相同，2048复杂槽及 nondefault stream 检查通过；见 `gpu-wide-scalar-ctest.log`。
- 隔离 q_proj 合成图：N16384/Q50/P8，1025 Prepare、1025 Rescale、无 Boot，全部8192复杂槽，maxabs `9.909881804e-10`，虚部/tail也通过；运行2.932秒。见 `q50-q-proj-corrected-report.json` 与 `.log`。这是线性图回归，不是完整预训练模型验收。
- 统一执行器 core 8/8、编译器10/10。Auto 只在 CPU lane 可用时给新组分配CPU；已准入组继续推进。

188Server 的新二进制、Q50资源边界及完整模型执行仍需单独验证。完整模型容差保持 atol=0.001、rtol=0.0001，未因修复放宽。
