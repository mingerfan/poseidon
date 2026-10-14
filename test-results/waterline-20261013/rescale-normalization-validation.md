# Rescale 数值修复

旧 API 按实际 Q prime 连续 Rescale 后，直接把 scale 改为 `2^target`，没有调整密文系数。独立 q_proj 图测得约 1.090078 倍的幅值，完整模型的旧运行因此不能作为数值验收；PID30187 已按用户要求停止，没有产生完整输出。

CPU/GPU 共同按实际 prime 选择原生 Rescale 次数。设输入 scale 为 Δ，目标为 T，选定原生除模的 prime 乘积为 P，取正 uint64 整数 `m=round(TP/Δ)`；先将全部输入 RNS 系数乘 m，并把中间 scale 同步为 mΔ，再原生除模，实际 scale 为 mΔ/P。最后对齐目标 T 的相对幅值误差为 `m/(TP/Δ)-1`，必须不超过 `1e-9`。剩余物理层使用 ModSwitch。

修复保留原 lazy rescale：生产图仍是 `--waterline=40`，逻辑160→40、物理drop5。其整数校正约2^30，中间scale约190，再除五个30-bit prime到约40。原先先除再乘会先降到约10，放大舍入误差；此前 API 为避免该损失而拒绝完整Q50计划，不能据此把原逻辑40认定为配置错误。把waterline改成120会改变逻辑图和Boot布局，额外候选编译已停止且未完成，不能作为等价修复。

目标 scale 须至少 `2^40` 且低于实际剩余模数空间，中间scale须有限且为正，整数对齐相对误差须在上述界内；不满足时显式报错。因整数取最近值，实际除模后的scale可能比目标低约1e-9相对量，检查不错误地要求双精度log2严格>=40。这些条件不是通用噪声、幅值或安全参数证明。

编译器 d5ed984 为每次逻辑 scale 降低额外保留至少30个 nominal modulus bits，120-bit 逻辑步长对应5个30-bit物理层。HEFT 源码没有修改。GPU scalar 全链路保留 uint64，并修复原地 scalar 调用覆盖输入 storage 的问题。

早期后乘版的本地验证（保留原始证据，不代表scale40修复）：

- CPU API 9/9：N16384/Q50、输入 scale230/240 到120、计划5/6层，全部8192复杂槽；最大误差约 `1.3e-11`。另保留50-bit CPU prime 的单层近1对齐，以及低精度显式拒绝检查。见 `cpu-rescale-normalization-tests.log`。
- GPU API 20/20：独立普通/多层 Rescale、MulCC/Relinearize/Rotate 与全部槽的稠密运算参考；见 `gpu-normalized-rescale-api.log`。
- 正式 CTest `poseidon_runtime_gpu_scalar_tests`：16组原地/独立输出，974848 RNS words 与独立128-bit模乘逐字相同，2048复杂槽及 nondefault stream 检查通过；见 `gpu-wide-scalar-ctest.log`。
- 隔离 q_proj 合成图：N16384/Q50/P8，1025 Prepare、1025 Rescale、无 Boot，全部8192复杂槽，maxabs `9.909881804e-10`，虚部/tail也通过；运行2.932秒。见 `q50-q-proj-corrected-report.json` 与 `.log`。这是线性图回归，不是完整预训练模型验收。
- 统一执行器 core 8/8、编译器10/10。Auto 只在 CPU lane 可用时给新组分配CPU；已准入组继续推进。

预乘修复新增 CPU API 9/9 正控：scale40、drop5/6，54次全2048复数槽独立dense oracle，包含两层CP/CC交叉及AddCC/AddCP，最坏maxabs=5.0261e-8；见 `cpu-rescale-prescale-tests.log`。原target80/120、50-bit模数与target20拒绝回归通过。GPU与组合图、188Server当前修复版及完整模型另行记录；完整模型尚未验收。完整模型容差保持 atol=0.001、rtol=0.0001，未因修复放宽。

预乘最终版本地组合正控见 [脚本](lazy-rescale-composite-regression.py)、[结果](lazy-rescale-composite-regression.json)、[日志](lazy-rescale-composite-regression.log)：生产BSGS线性、平方/全槽求和、polynomial/mask、RMS和深层polynomial共6输出，每后端检查49152槽。逻辑40的CPU/GPU/Auto最大误差分别4.44192e-5 / 8.73261e-5 / 5.26052e-5；120约2.15e-10。原定atol=.001、rtol=.0001未改；两者均5次Host refresh，但41/36次Rescale、部分最终level/scale不同，因此不把120称为原lazy40图的等价配置。实际Q对应的77条normalizer检查全部通过，相对误差最大4.7473e-10。脚本、header、compiler与runner hash绑定已核对。旧后乘API的scale40失败见 [负控日志](rescale-postmultiply-composite-negative-control.log)。

188新预乘版通过CPU9/9、GPU20/20、scalar16组、Prepare真实分配失败清理，以及单卡/全部四卡默认通信/全部四卡HostStaged压力；[源码与binary绑定](188-prescale-build.status)、[监督退出0](188-prescale-exit-status)。同一小图40/120六路也在物理GPU1通过，[40 GPU报告](188-prescale-composite-scale40-gpu-report.json)。使用公开 [组合运行脚本](run-lazy-rescale-composite-188.sh) 可复用生成的两个小图目录，生成目录及日志label必须新建。

组合图复现：`python test-results/waterline-20261013/lazy-rescale-composite-regression.py --output build-lazy-rescale-fresh`。将其 `scale40/`、`scale120/` 中的plan、spec、fixture及bundle复制到188的输入目录后，设置 `POSEIDON_COMPOSITE_INPUT_DIR` 和新的 `POSEIDON_COMPOSITE_LABEL` 执行上述脚本。单独保存 `latest-build.status`；不复制大模型权重或大模型plan到仓库。
