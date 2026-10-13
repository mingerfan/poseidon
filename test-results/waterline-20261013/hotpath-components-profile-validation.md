# 小型热路径诊断

此处只衡量 public API 的合成小程序，不是完整模型吞吐测试。188Server 上程序与旧完整 Auto 模型 PID30187 同时运行，选择物理 GPU1，池硬上限 1536MiB、初始 64MiB；启动前要求至少 2GiB 空闲。参数 N16384/Q40/P8/scale120，每个组成部分 10 次。实际模数列表、资源查询与所有样本见 JSON。

未插桩运行 UTC 12:28:08–12:28:20，通过全部 8192 槽实部和虚部数值检查，最大误差不超过 2.22e-16。池内实际分配峰 610188392 字节、reserved 800MiB，结束 pinned live 为 0。

| 组成部分 | 均值 ms | 说明 |
|---|---:|---|
| CPU 完整 Q40 Encode | 26.025 | Host 数学及结果构造 |
| CPU Prepare 提交 | 29.166 | 完整 CPU 编码和上传提交 |
| CPU Prepare 编码计数 | 27.386 | API 内 CPU encoder wall 计数 |
| GPU Prepare 提交 | 1.083 | 包括每次 pinned 分配与提交 |
| GPU Prepare Ready 等待 | 0.220 | 提交返回后另测 |
| GPU Prepare 回收 | 0.437 | 包括每次 pinned 释放 |
| GPU Encode event | 0.272 | 实际 encoder stream；含排队及 launch 间隙 |
| GPU raw H2D event | 0.018 | 实际上传 stream |
| AddCP 提交 / event | 0.019 / 0.041 | 已 Ready 输入 |
| Rotate 提交 / event | 0.174 / 1.181 | 已 Ready 输入 |
| Host secret refresh API | 101.822 | 输入已经 Ready，不含等待输入 |

GPU event、提交与 Ready 等待可能重叠，不能把这些时间相加。该诊断没有重现完整模型中每份 Prepare 的 4ms，因此不能把 4ms 全部归因于 GPU 数学。它证明了每份 GPU 准备仍有约 1ms 提交和约 0.4ms 回收的固定成本，而实际 GPU Encode event 约 0.27ms。

Ready Host 输入的 secret refresh 分段均值：decrypt Q8 2.017ms、decode Q8 9.869ms、encode Q40 26.903ms、encrypt Q40 67.000ms。分段和约 106.79ms，与直接 API 的约 101.82ms 同量级，说明此处主要耗时是 CPU 数学，而不是输入依赖等待；不同循环样本不可当作精确分账。

Nsight Systems 仅启动本诊断程序，UTC 12:33:11–12:33:29，`--trace=cuda --sample=none --cpuctxsw=none`。46 次 `cudaMallocHost` 累计 2.456s，46 次 `cudaFreeHost` 累计 0.919s，合计为这个诊断的 CUDA API 累计调用时间的 70.2%。中位数分别 1.095ms / 0.504ms，最大值 633.449ms / 378.016ms；累计与均值被少数长调用显著拉高。它不是完整模型 wall time 的 70.2%，也不是可宣称的端到端加速比例。

在自己 stream 先排入 50ms Host callback，再分配 64KiB pinned，分配返回仅 0.996ms；该结果没有证明全 context 同步。callback 不调用 CUDA。并发任务或驱动造成长尾的具体机制尚未由此探针确定。

可复现 profile/stats 命令、UTC、边界与原始 source SHA256 见 `hotpath-components-188-profile-scope.json`；编译链接命令见 `hotpath-components-188.commands.json`，原 binary/library SHA256 见 `hotpath-components-188-provenance.txt`。原 source 保留为 `hotpath-components-before-cache-test.cpp`，新的 `hotpath-components-test.cpp --cache` 属于后续缓存对照，不能替代此次源快照。CUDA API/kernel CSV 与小日志已取回；大的 `.nsys-rep` 和 `.sqlite` 保留在服务器原诊断目录。
