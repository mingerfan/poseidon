# 冻结v3完整12层encoder实测（历史scope）

后续v6真实12层中位76.417356609s，1warmup＋3测、同原12层权重，
但不同binary/window/material；详见[新v6结果](../full_inference_v6/GPU_RESULTS.md)。
本文件与v3证据包保持历史数值，不将新packing或新结果混入。

T128/H768/12×64/F3072，72组独立dense矩阵，12层真实SSA串接；不含embedding/pooler/classifier。
1warmup＋3完整测，online秒 `[119.297476332,119.299492793,119.172574332]`，
**中位119.297476332s**，Runtime wall中位121.843493260s。非单层外推。

每run实际348次native Boot/324prepared Linear调用；正式1044Boot、含warmup1392。
3CT/98,304槽有限，最后L12/Q13/scale55.370070863338654，但max参考误差29704.08564，
**精度未通过**。tracked显存峰值39.695GiB、Host RSS47.649GiB。没有逐层HE误差/全层GPU timer。
Cerium尚无12层实测，不能给百分比。09:22:01UTC已退出、GPU归零，09:51:51UTC复核。

原始报告：`test-results/bert-a800-20261115/full12-v3/`。该目录被Git忽略，额外复制小型
证据包到本目录的`v3-full12-evidence.tar.gz`（约2.8MB），SHA256：
`6e69534c674949539051af79e84c8fa65e4d091ad9ecde6bceaa26966c979051`。
源码/ratio256 profile冻结包为`source-snapshot.tar.gz`，私有runner SHA：
`658d6d860bad4997eebc8d3420501419a0b73ef8216a157776ef1b85369cff38`。

远端完整资产：`/root/private_data/bert-dsl-20261115/full12-v3/`；此处证据包不包含
340MB dense binary或整个巨计划，其SHA与可再生成的冻结source/RNG环境已归档。

活跃v4候选不是本次结果；跨NumPy/libc环境相同seed也不保证float32逐bit相同，
后续固定输入/权重比较见[绑定说明](../WEIGHT_BINDING.md)。没有后续GPU任务排队。
