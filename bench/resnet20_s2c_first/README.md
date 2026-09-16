# ResNet20 S2C-first 独立应用

这是本工程当前默认的单卡 ResNet20 密态推理入口。性能模式执行一条连续密文链：

```text
27 x prepared encrypted input patches (Q32) -> stem (Q31) -> ReLU (Q9)
  -> 9 BasicBlocks
  -> 18 x (S2C-first degree-59 bootstrap + ReLU)
  -> global average pool -> FC -> encrypted logits
```

输入 patch 直接在 Q32、scale `2^40` 编码和加密；Stem 的一次明文乘法与
rescale 自然得到 Q31、scale `2^40`。入口不再先生成 Q50 patch，也不存在原来的
Stem 后 `Q49 -> Q31` 显式降层。全局参数仍保留 Q50/P25，因为 Bootstrap 需要
`Q2 -> Q50 -> Q46 -> Q31`。

计时前一次性完成输入、旋转键、S2C/C2S 矩阵、EvalMod 常量、卷积/ReLU/head
明文常量和 workspace 的准备及上传。正式 online 区间只设一个全网 CUPTI
correlation 和一个首尾计时区间，从 prepared encrypted input 连续计算到 encrypted
logits；区间内没有逐阶段 `cudaDeviceSynchronize()`、H2D/D2H、中间重加密或运行时
换入换出。计时和 CUPTI 采集完全结束后，入口下载并解密一次最终 logits，用于报告
预测类别和最大 logit 误差；这次 D2H/observer 不计入 online latency。

入口与旧应用的 Bootstrap scheduler 隔离：构建时不编译或链接旧应用的
`gpu_ckks_runtime.cpp`、`gpu_resnet20_inference.cpp`、`gpu_relu.cpp` 和 `main.cpp`，
也不调用 `GpuCkksRuntime::bootstrap()`。旧的逐阶段同步性能实现已从本入口删除，
`--unsafe-performance-only` 现在只对应上述全常驻连续实现。

当前仍复用外部原型中与 scheduler 无关的模型资产读取器和无状态 multiplexed
tensor/Conv 布局算子，以保持权重、packing、卷积和 Option-A shortcut 语义不变；
不复用旧 Bootstrap 或旧网络调度。完全自包含仍是后续独立迁移任务。

## 构建和运行

```bash
bash bench/resnet20_s2c_first/build.sh /tmp/poseidon_resnet20_s2c_first

# 默认性能路径：全材料常驻后的单次连续 9-block/18-bootstrap 推理
CUDA_VISIBLE_DEVICES=0 /tmp/poseidon_resnet20_s2c_first \
  --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0

# 严格验收性能日志
python3 bench/resnet20_s2c_first/summarize_performance.py /path/to/performance.log

# 只检查参数、拓扑和数据契约，不分配 GPU
/tmp/poseidon_resnet20_s2c_first --metadata-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0

# 带 observer 的完整正确性回归；不用于性能计时
CUDA_VISIBLE_DEVICES=0 /tmp/poseidon_resnet20_s2c_first \
  --unsafe-accuracy-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0
```

`--unsafe-*` 表示 Q50/P25、`dnum=2` 尚未获得安全参数批准，不能作为生产安全
配置。两条 GPU 路径均使用随机非零秘密密钥，不允许零密钥或中间结果重加密。

## 当前完整实测（2026-09-16，Q32 输入与稀疏参数层）

V100-SXM2-32GB 上已完成一次 9/9 blocks、18/18 bootstraps 的完整 warmup 和一次
正式连续 online 推理。应用侧所有材料始终常驻，正式区间内 H2D/D2H 为 0、无逐阶段
同步、无中间解密/重加密、无运行时换入换出。30 GiB RMM 池在准备和 warmup 完成后
仍有 `1470.375 MiB` CUDA 可用显存。

应用 KeySwitch 保持 fixed `dnum=2` 和分层 P：evaluation-key payload 为
`1832 MiB`。每个 application `GpuParameterData` 现在只上传实际使用的 q-only 层，
共 19 层（6 个 rotation 层、13 个 ReLU relinearization 层）；七组主要 NTT 表的
预检大小为 `862.75 MiB`。这只是删除不可达的参数表，不改变计算图、密钥、模数链
或算子结果。

| 口径 | 延迟 |
|---|---:|
| 单次应用 wall | `7218.232339 ms` |
| CUDA event | `7227.673828 ms` |
| CUPTI kernel/D2D/memset 活动区间并集 | `7058.020744 ms` |
| host enqueue | `7203.875640 ms` |

| 类别 | GPU activity 时间 | 全网 GPU 占比 |
|---|---:|---:|
| Conv+BN | `702.325819 ms` | `9.9507%` |
| ReLU | `1180.926338 ms` | `16.7317%` |
| Bootstrap | `5132.198665 ms` | `72.7144%` |
| Shortcut | `29.649602 ms` | `0.4201%` |
| Pool+FC | `12.414824 ms` | `0.1759%` |
| Residual/copy | `0.505496 ms` | `0.0072%` |
| **合计** | **`7058.020744 ms`** | **`100.0000%`** |

计时结束后只解密最终 Q4 logits。GPU 与独立明文参考都预测类别 3，最大 logit
误差为 `0.0124877523`，正确性验收通过。机器可读结果见
`performance_20260916_q32_sparse_parameters.json`；原始日志在 `/tmp`，属于临时
运行产物，JSON 中记录了 SHA-256。

## 历史性能基线（2026-09-15，Q50 Stem 输入）

以下结果产生于输入 patch 仍从 Q50 开始、且应用 rotation 尚未切换到当前分层 P
实现时，仅保留作优化前基线，不能当作当前 Q32 Stem 版本的正式性能结果。

V100-SXM2-32GB 的首轮完整实测：

| 口径 | 延迟 |
|---|---:|
| 单次应用 wall（首 kernel 提交到最终全设备同步返回） | `8162.723379 ms` |
| CUDA event（默认 stream 首尾） | `8164.140137 ms` |
| CUPTI kernel/D2D/memset 活动区间并集 | `8071.055126 ms` |
| host 从进入图到提交完最后一个算子的跨度 | `8127.841425 ms` |

CUPTI 验收 `108241` 条活动，H2D/D2H 为 `0`。CUPTI activity 并集比 wall 少约
`91.67 ms`，这是 GPU activity
之间的 launch/API 间隙。当前后端在若干密文复制/NTT 路径中仍使用同步 D2D
`cudaMemcpy` API，所以 host enqueue 接近 wall；它是本次连续应用的真实行为，已
包含在 `8162.723379 ms` 中。

连续采集期间 CUPTI 只启停一次，各算子边界仅切换 external correlation ID，不做
阶段同步。14 个底层标签确定性归并为以下 6 大类，六项合计
`8071.055126 ms`，与全网 GPU activity 并集一致：

| 类别 | GPU activity 时间（kernel+D2D+memset） | 全网 GPU 占比 |
|---|---:|---:|
| Conv+BN | `1423.060480 ms` | `17.6317%` |
| ReLU | `1430.068546 ms` | `17.7185%` |
| Bootstrap | `5130.018323 ms` | `63.5607%` |
| Shortcut | `56.063438 ms` | `0.6946%` |
| Pool+FC | `31.299733 ms` | `0.3878%` |
| Residual/copy | `0.544606 ms` | `0.0067%` |
| **合计** | **`8071.055126 ms`** | **`100.0000%`** |

其中 stem Conv+BN 归入 `Conv+BN`，encrypted head 归入 `Pool+FC`；residual
clone/add 和 stem level-drop 归入 `Residual/copy`。BN 已折叠进卷积明文权重。

Bootstrap 的 `5130.018323 ms` 进一步分为：S2C `348.863515 ms`、prepare
`0.185311 ms`、ModRaise `8.213223 ms`、C2S `1973.734041 ms`、两路 EvalMod
合计 `2746.897622 ms`、实部重组 `52.124611 ms`。其中 EvalMod 占 Bootstrap
`53.5456%`，C2S 占 `38.4742%`。

从进程启动到日志最终写完约 `467 s`，其中约 7 分 39 秒用于参数表、密钥、
矩阵和明文常量的生成/上传以及一次不计时的 warmup。这些是 offline preparation，
不属于单次 prepared online latency。

显存核算的静态 payload（不含 GPU 参数表）为 `21324.5 MiB`。另用
`14216.5 MiB` 常驻占位叠加真实 bootstrap 做过保守压力测试，结束时仍有
`4146.375 MiB` 可用；正式路径使用固定 30 GiB RMM 池，warmup 和 online run
均无换入换出。机器可读结果见 `performance_20260915.json`。

## 正确性回归（保留但非默认）

2026-09-14 的独立 observer 入口完整跑通 9/9 blocks、18/18 bootstraps：预测为
正确类别 3，最大 logit 误差 `0.01224267227883669`，中间重加密为 0。机器可读
清单见 `acceptance_20260914.json`，日志 SHA-256 为
`44129ecada743bd9414aa075bb0101c88df5ed64c2f845ccf216d10c77b2ddac`。

正确性回归中的逐边界同步、解密和参考检查只用于诊断，不能作为应用 latency。
其日志可用 `summarize_run.py` 验收。
