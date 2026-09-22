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

以下假设 GPU 核心的 CMake build 目录已配置；更新 evaluator/handler 后先重编译
核心对象，再链接应用，避免复用旧的 `.o` 文件。

```bash
cmake --build src/poseidon/tests/bootstrapping/build \
  --target test_gpu_bootstrap_modraise -j2
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

## 上一轮完整实测（2026-09-18，Q32、C2S full-baby 与 ReLU Q-prefix views）

V100-SXM2-32GB 上已完成一次 9/9 blocks、18/18 bootstraps 的完整 warmup 和一次
正式连续 online 推理。应用侧所有材料始终常驻，正式区间内 H2D/D2H 为 0、无逐阶段
同步、无中间解密/重加密、无运行时换入换出。30 GiB RMM 池在准备和 warmup 完成后
仍有 `1470.375 MiB` CUDA 可用显存。

应用 KeySwitch 保持 fixed `dnum=2` 和分层 P：evaluation-key payload 为
`1832 MiB`。每个 application `GpuParameterData` 现在只上传实际使用的 q-only 层，
共 19 层（6 个 rotation 层、13 个 ReLU relinearization 层）；七组主要 NTT 表的
预检大小为 `862.75 MiB`。这只是删除不可达的参数表，不改变计算图、密钥、模数链
或算子结果。

S2C/C2S 的 316 条 QP 明文对角线现在使用精确的 bit-reversed 周期表示。上传器在
准备阶段逐 residue 验证 Q/P 重构，正式路径还会拒绝 full/compact 混合布局或非精确
明文。QP 明文 payload 从 `4108 MiB` 降到 `884.787109 MiB`，压缩
`4.642925x`，减少 `3223.212891 MiB`。固定 30 GiB RMM 池在进程启动时已经整体
预留，因此 `cudaMemGetInfo` 的池外 `free_MiB` 仍为 `1470.375`；它不会反映池内
这部分 payload 的下降。

新版入口将三层 Q50/P25 C2S 的默认 baby tile 从 `4` 提高到 `8`。每层的 8 个
baby steps 由一个 tile 完成，因此三层 C2S 总共只需要 3 个 tile batch。这里不能将
full-baby 与 KeySwitch/plaintext-MAC 融合混为一谈：三层 giant group 数为 4/8/8，
只有第一层满足现有融合 kernel 的 `group_count <= 4` 条件。该选择只改变 GPU
kernel 分批和中间 accumulator 访存，不改变
旋转、模数层、明文矩阵或密文运算语义；通用 GPU 库的默认值仍保持不变。设置
`POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE=4` 可以回退并做 A/B。

ReLU 现在还会把确定处于最后一次使用位置的临时密文直接移交给下一步，不再为
“源 Q 层与目标 Q 层相同”的 ModDrop 分配新缓冲区并复制两个密文分量。19 次 ReLU
共消除 `494` 次同层 materialization，即 `988` 个复制 kernel；多项式、重线性化、
rescale 顺序和 22 层模数消耗均未改变。设置
`POSEIDON_RELU_ZERO_COPY_MODDROP=0` 可在同一二进制中恢复复制路径做 A/B。

共享的 Chebyshev basis、leaf 和原始 tail 输入不能转移所有权；这些 operand 现在由
`multiply_q_prefix`/`multiply_plain_q_prefix` 创建只读目标层 view，kernel 只读取所需
Q 前缀，原密文及其 metadata 保持不变。19 次 ReLU 共用 `1463` 个前缀 view，逻辑
截取 `1159` 个 Q limbs，进一步删除 `2926` 个复制 kernel；整网只剩每次 ReLU 开头
为了保留 tail 原输入而产生的 1 次 materialization，共 `19` 次。设置
`POSEIDON_RELU_Q_PREFIX_VIEWS=0` 可以恢复共享 operand 的物化复制路径。

| 口径 | 延迟 |
|---|---:|
| 单次应用 wall | `7000.737040 ms` |
| CUDA event | `7005.067871 ms` |
| CUPTI kernel/D2D/memset 活动区间并集 | `6906.163947 ms` |
| host enqueue | `6986.389257 ms` |

| 类别 | GPU activity 时间 | 全网 GPU 占比 |
|---|---:|---:|
| Conv+BN | `697.149504 ms` | `10.0946%` |
| ReLU | `1105.893492 ms` | `16.0131%` |
| Bootstrap | `5060.722150 ms` | `73.2783%` |
| Shortcut | `29.576809 ms` | `0.4283%` |
| Pool+FC | `12.309201 ms` | `0.1782%` |
| Residual/copy | `0.512791 ms` | `0.0074%` |
| **合计** | **`6906.163947 ms`** | **`100.0000%`** |

计时结束后只解密最终 Q4 logits。GPU 与独立明文参考都预测类别 3，最大 logit
误差为 `0.0128694541`，正确性验收通过。机器可读结果见
`performance_20260918_q32_full_baby8_relu_qprefix.json`；原始日志在 `/tmp`，属于
临时运行产物，JSON 中记录了 SHA-256。

同一 GPU、同一新二进制的相邻 Q-prefix view A/B 中，关闭/开启时 ReLU 分别为
`1161.442425/1105.893492 ms`，减少 `55.548933 ms`（`4.7828%`）；全网 GPU
activity 分别为 `6949.330720/6906.163947 ms`，减少 `43.166773 ms`
（`0.6212%`）。activity 数从 `114693` 精确降到 `111767`，差值 `2926` 与删除的
复制 kernel 数完全一致。相邻 wall 减少 `50.265873 ms`（`0.7129%`）；两次推理
均通过最终 logits 验收，且 prepared 后可用显存均为 `1470.375 MiB`。

同一 GPU、同一二进制的相邻 ReLU ModDrop A/B 中，关闭/开启时 ReLU 分别为
`1183.129951/1164.543422 ms`，减少 `18.586529 ms`（`1.5710%`）；全网 GPU
activity 分别为 `6979.256124/6955.066433 ms`，减少 `24.189691 ms`
（`0.3466%`）。activity 数从 `115681` 精确降到 `114693`，差值 `988` 与删除的
复制 kernel 数完全一致。相邻 wall 减少 `70.971903 ms`，但 wall 仍受 host/API
间隙波动影响，不把它单独作为稳定收益依据。两次推理均通过最终 logits 验收。

同一 GPU、同一二进制和输入下显式执行 tile=4/8 A/B 后，C2S 分别为
`1942.908118/1905.949585 ms`。修改默认值后第二次无环境覆盖的 tile=8 回归得到
`1905.266315 ms`；结合此前压缩版 tile=4 的 `1942.882274 ms`，两组 C2S 均值为
`1942.895196/1905.607950 ms`，稳定减少 `37.287246 ms`（`1.9192%`）。每次完整
推理的 CUPTI activity 数从 `115789` 降为 `115681`，恰好减少 `108`，对应 18 次
Bootstrap、每次三层 C2S 各删除一个 Q tile kernel 和一个 P tile kernel。严格相邻
单次 A/B 的全网 GPU activity 减少 `43.090887 ms`（`0.6136%`）；wall 差值受
host/API gap 波动影响，不作为该优化的主要归因证据。

相对同日未压缩 QP 明文的上一轮实测，GPU activity 从 `7058.020744 ms` 降到
`7010.214679 ms`，观察到减少 `47.806065 ms`（`0.6773%`）；其中 S2C+C2S
合计减少 `33.970665 ms`，Bootstrap 总计减少 `44.440886 ms`。单次 wall 只减少
`9.446040 ms`，因为本轮 host/API gap 更大；当前只把 CUPTI 阶段差值作为压缩路径
的主要性能证据，不把一次 wall 差值外推成稳定收益。未压缩基线保留在
`performance_20260916_q32_sparse_parameters.json`。

## ReLU 基函数构造精确融合（2026-09-19）

同一张 V100-SXM2-32GB（GPU 1）、同一二进制，先关闭再开启基函数融合；
Conv 多项乘加和 ReLU 叶节点融合均保持开启，各测一次完整连续推理：

| 口径 | 基函数融合关闭 | 基函数融合开启 | 开启减关闭 |
|---|---:|---:|---:|
| ReLU GPU activity | 1085.493617 ms | 1073.878851 ms | -11.614766 ms（-1.0700%） |
| 全网 GPU activity 并集 | 6856.672954 ms | 6847.917651 ms | -8.755303 ms（-0.1277%） |
| 连续推理 wall | 6955.467662 ms | 6944.887899 ms | -10.579763 ms（-0.1521%） |
| GPU activity 数 | 109469 | 108709 | -760 |

活动数差值与 `19 * (10*2 + 5*4) = 760` 完全一致。常量修正原来的加法、
c0 明文减法和 c1 复制变为一个 kernel；带修正乘积时，加法、两个分量的
明文乘法和减法变为一个 kernel。观察到的是小幅收益，不将一次相邻 A/B
外推为稳定加速比例；Bootstrap/Conv 等未修改类别的变化不算作本优化收益。

两次正式区间 H2D/D2H 均为 0，无逐阶段同步、observer 或运行时换入换出；
固定 30 GiB 池内完成，prepared 后池外 CUDA free 均为 `1470.375 MiB`。
图像 0 的预测均为类别 3，最大 logit 误差关闭/开启分别为
`0.01131988254 / 0.01675863613`，通过原有 `0.1` 验收阈值。两进程重新生成
随机密钥及输入密文，误差差值不能归因于融合；同密文精确对照见下。
完整结果和二进制/日志 SHA-256 见
[`performance_20260919_relu_basis_fusion.json`](performance_20260919_relu_basis_fusion.json)。

默认启用 `POSEIDON_RELU_BASIS_FUSION=1`，设为 `0` 回退；关闭 Q-prefix views
也会回退。它与 `POSEIDON_RELU_LEAF_FUSION` 独立，且不修改 Conv 批量乘加。
每次 ReLU 在密文乘法及重线性化之后，将 10 次 `2*product - encoded_one` 和
5 次 `2*product - basis_prefix*encoded_alignment_one` 分别融合为一个 kernel。
仍使用原来编码的正数常量/对齐明文，在各 q 模数下逐项计算并减去修正项，
不重新近似系数，不移动任何 rescale，不省略密文乘法或重线性化。

全网 19 次 ReLU 共 285 次修正：190 次明文常量、95 次带 Q-prefix 的修正乘积。
独立的 95 次修正明文乘法被融合计算，并非被舍弃；默认叶节点融合也开启时，
`prefix_multiply_plain_calls` 从 95 变为 0。Q-prefix view 总数 1463、丢弃前缀 limb
统计 1159、46 个常驻 ReLU 明文、`Q31 -> Q9` 的 22 层消耗均不变。

准备阶段强制比较全部 285 次修正的 824,311,808 个 RNS 残基及 metadata。
`RELU_BASIS_FUSION_PREPARED` 报告上述计数；正式 online 区间的
`RELU_BASIS_FUSION` 要求 `exact_checks=0`，没有新增 D2H、observer 或阶段同步。
融合不增加常驻 GPU 数据，精确性对照的临时密文仅用于计时前的预热。

独立诊断已通过 22 次 API 对照（83,755,008 个残基，含零/负系数、不同 Q/scale、
两类输出别名和输入不变性），拒绝 9 个非法输入。实际 ReLU 的 15 次修正
（43,384,832 个残基）及完整 Q9 输出（1,179,648 个残基）与关闭本融合时完全一致。
对原多项式的最终误差为 `3.87634213e-6`，对解密后输入的多项式误差为
`6.85460713e-9`。真实随机非零密钥 h=192；未改动编码 scale 或精度阈值。

**保留的诊断失败：** 全域/近零输入的中间阶段仍未通过历史 `1e-5` guard，
诊断返回 1（`final_output=PASS, stage_guard=FAIL`），不能称为全部精度检查通过。
同密文完整 ReLU 残基完全相同，说明这次融合没有改变既有误差。

```bash
bash Doc/analysis/resnet20_joint_chain_20260911/build_relu_precision.sh \
  /tmp/poseidon_relu_basis_precision relu_precision.cpp
CUDA_VISIBLE_DEVICES=1 /tmp/poseidon_relu_basis_precision --unsafe-accuracy-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt

# 同一 GPU 串行执行两个完整推理进程，只切换基函数融合
CUDA_VISIBLE_DEVICES=1 POSEIDON_RELU_BASIS_FUSION=0 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0
CUDA_VISIBLE_DEVICES=1 POSEIDON_RELU_BASIS_FUSION=1 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0

# 验收本轮已有临时日志；重跑时换为自己的日志路径
POSEIDON_RELU_BASIS_ACCURACY_LOG=/tmp/poseidon_relu_basis_accuracy.log \
POSEIDON_RELU_BASIS_OFF_LOG=/tmp/poseidon_resnet20_relu_basis_off.log \
POSEIDON_RELU_BASIS_ON_LOG=/tmp/poseidon_resnet20_relu_basis_on.log \
python3 -m unittest bench.resnet20_s2c_first.test_relu_basis_fusion
```

本轮 36 项 bench 检查全部通过（包含对既有精度 guard 失败的如实记录检查）；
完整预热的 285 次修正残基对照也全部通过，不只是独立 API 测试通过。

## Conv 多项乘加精确融合（2026-09-19）

同一张 V100-SXM2-32GB（GPU 1）、同一二进制，先关闭再开启，各正式测量一次
完整的全常驻连续推理（ReLU 叶节点融合保持开启）：

| 口径 | Conv 融合关闭 | Conv 融合开启 | 开启减关闭 |
|---|---:|---:|---:|
| Conv+BN GPU activity | 700.057132 ms | 685.580828 ms | -14.476304 ms（-2.0679%） |
| 全网 GPU activity 并集 | 6866.177970 ms | 6866.703150 ms | +0.525180 ms |
| 连续推理 wall | 6964.143024 ms | 6963.905257 ms | -0.237767 ms |
| GPU activity 数 | 110589 | 109469 | -1120 |

**局部卷积有收益，本次整网延迟基本持平。** Bootstrap/ReLU 的测量值分别增加
`12.325277 / 2.568927 ms`，抵消了 Conv 的下降。这仅是相邻单次 A/B，不足以
宣称整网有稳定加速；不将其他类别的变化归因于本次 Conv 算法修改。

两次正式区间 H2D/D2H 均为 0，无逐阶段同步或运行时换入换出，均在固定
30 GiB RMM 池内完成，prepared 后池外 CUDA free 均为 `1470.375 MiB`。
图像 0 两次预测都是类别 3；最大 logit 误差关闭/开启分别为
`0.01066989665 / 0.01325288209`，均通过原有 `0.1` 阈值。独立进程使用不同的
随机密钥及输入密文，不能把这两个误差的差值当作融合的新增误差；同密文的
严格等价检查见下。没有改动精度阈值，也未重新声称历史 ReLU stage guard 通过。

机器可读记录、完整分类耗时和二进制/日志 SHA-256 见
[`performance_20260919_conv_plain_batch.json`](performance_20260919_conv_plain_batch.json)。

默认启用 `POSEIDON_CONV_PLAIN_BATCH=1`，设为 `0` 可在同一二进制中回退。
仅作用于 prepared 连续路径中 18 次块内 Conv 的 Q9 权重乘加，不修改外部 Conv
源码、Stem、Shortcut、Head 或 Bootstrap。160 个输出组的全部 1440 项权重乘法
按原顺序组成 `4+4+1` 批次；每组从 1 次两分量明文乘法和 8 次单项累加
（10 个 kernel）变成 3 个批量 kernel，共减少 1120 个 kernel。

这不是稀疏化或降低计算精度：BN scale/bias、边界掩码、support mask、selector、
旋转、通道归约、原有两次普通单 prime rescale 和最后一次 rescale 均保留，
`Q9 -> Q6` 不变。1696 个常驻 Conv 明文不变；批次仅持有已有 GPU 数据的引用，
不会引入新常驻密文副本或运行时换入换出。任何后续消费者先执行未满批次；
共享 destination 则先物化再 copy-on-write，避免修改别名。

完整应用在准备阶段强制用同一输入密文、同一组编码明文逐批对照原计算，检查
480 批、566,231,040 个 RNS 残基及 metadata 完全一致。检查需要额外的临时参考
密文和下载，但全部在 online 计时前；正式区间要求 `exact_checks=0`、`pending=0`，
仍由 CUPTI 确认 H2D/D2H 为零。`CONV_PLAIN_BATCH_PREPARED` 与
`CONV_PLAIN_BATCH` 分别报告准备和正式推理的计数，日志解析器严格验收。

独立诊断覆盖 1～10 项长度、零/负系数、掩码、满批及尾项、capture/replay、共享
目标的 copy-on-write、混合 Q 前缀和输出别名；65 次对照共 76,677,120 个残基
全部一致，4 个非法目标被拒绝。使用 OS 随机的真实非零 h=192 密钥。

```bash
bash Doc/analysis/resnet20_joint_chain_20260911/build_relu_precision.sh \
  /tmp/poseidon_conv_batch_precision conv_plain_batch_precision.cpp
CUDA_VISIBLE_DEVICES=1 /tmp/poseidon_conv_batch_precision --unsafe-accuracy-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt

# 完整应用 A/B：同一 GPU 串行执行，只改变 Conv 开关
CUDA_VISIBLE_DEVICES=1 POSEIDON_CONV_PLAIN_BATCH=0 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0
CUDA_VISIBLE_DEVICES=1 POSEIDON_CONV_PLAIN_BATCH=1 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0

# 使用本次已生成的临时日志验收；重跑时换成自己的日志路径
POSEIDON_CONV_BATCH_EXACT_LOG=/tmp/poseidon_conv_batch_precision.log \
POSEIDON_CONV_BATCH_OFF_LOG=/tmp/poseidon_resnet20_conv_batch_off.log \
POSEIDON_CONV_BATCH_ON_LOG=/tmp/poseidon_resnet20_conv_batch_on.log \
python3 -m unittest bench.resnet20_s2c_first.test_conv_plain_batch
```

## 上一轮完整实测（2026-09-19，ReLU 叶节点精确融合）

同一张 V100-SXM2-32GB（GPU 1）、同一二进制，相邻进程只切换融合开关，各运行
一次完整 warmup 和一次正式 9-block/18-bootstrap 连续推理：

| 口径 | 融合关闭 | 融合开启 | 差值 |
|---|---:|---:|---:|
| ReLU GPU activity | 1106.411194 ms | 1087.278247 ms | -19.132947 ms（-1.7293%） |
| 全网 GPU activity 并集 | 6884.535043 ms | 6862.368149 ms | -22.166894 ms（-0.3220%） |
| 端到端 wall | 6976.971494 ms | 6955.782018 ms | -21.189476 ms（-0.3037%） |
| GPU activity 数 | 111767 | 110589 | -1178 |

活动数下降与 `19 * (30*2 + 16 - 14) = 1178` 完全一致：每次 ReLU 原来有 30 次
两分量明文乘法和 16 次独立加法，改为 14 次融合求和。只把 ReLU 的局部下降作为
主要归因证据，不把其他类别的波动算作该优化收益；这是相邻单次 A/B，不是多次
测量的置信区间。两次计时区间 H2D/D2H 均为 0，prepared 后 CUDA free 均为
`1470.375 MiB`，无逐阶段同步、observer 或运行时换入换出。

两次完整推理均通过原有验收，图像 0 的预测都是类别 3；关闭/开启时最大 logit
误差分别为 `0.0077208743 / 0.0117373062`。两个独立进程重新生成随机密钥及输入
密文，因此不能把这两个误差的差值归因于融合；同输入、同密钥的严格残基对照见下。
机器可读记录为 [`performance_20260919_relu_leaf_fusion.json`](performance_20260919_relu_leaf_fusion.json)，
包含 A/B 数据、精度诊断的保留失败、二进制和临时日志的 SHA-256。

默认启用 `POSEIDON_RELU_LEAF_FUSION=1`，设置为 `0` 可在同一二进制中回退。
它依赖 Q-prefix views；关闭 `POSEIDON_RELU_Q_PREFIX_VIEWS` 时也会回退。
每个叶节点将 1～4 个 `basis * encoded_coefficient` 在单个 kernel 内按原顺序
逐项模乘、模加，不再物化各项乘积密文。每次 ReLU 的 30 个系数项全部保留，
分为 14 个叶节点，删除 16 次独立中间加法；全网共 570 项、266 个叶节点。
零系数也不跳过。原有系数、编码 scale、Q-prefix、重线性化、rescale 次序、
`Q31 -> Q9` 的 22 层消耗、46 个常驻 ReLU 明文全部不变。

`RELU_LEAF_FUSION` 日志强校验上述计数；独立明文乘法计数由 665 降为 95，
其余 570 项由融合 kernel 计算，并非被删除。计时模式的 exact-check 数必须为 0，
防止精度 observer 的 D2H/同步混入正式计时。`summarize_performance.py` 同时接受
融合开启/关闭两种完整记录，并兼容先前无此字段的历史记录。

独立诊断自动执行同密文、同密钥的逐叶及完整 ReLU 残基对照：

```bash
bash Doc/analysis/resnet20_joint_chain_20260911/build_relu_precision.sh \
  /tmp/poseidon_relu_leaf_fusion relu_precision.cpp
CUDA_VISIBLE_DEVICES=1 /tmp/poseidon_relu_leaf_fusion --unsafe-accuracy-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt

# 相同完整应用二进制，仅切换叶节点融合；不要并发运行两组性能测试
CUDA_VISIBLE_DEVICES=1 POSEIDON_RELU_LEAF_FUSION=0 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0
CUDA_VISIBLE_DEVICES=1 POSEIDON_RELU_LEAF_FUSION=1 \
  /tmp/poseidon_resnet20_s2c_first --unsafe-performance-only \
  Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt 0
```

本次独立验证：1/2/3/4 项、零/负系数、混合 Q 前缀、输入不变性和输出别名测试
通过，9 个非法输入被拒绝；14 个实际叶节点的 `43,122,688` 个残基以及完整 ReLU
输出的 `1,179,648` 个残基与原实现逐一相同，metadata 也相同。最终对原多项式
误差为 `1.558618884e-6`，对解密后输入的多项式求值误差为 `5.800888148e-9`。

**保留的诊断失败：** 全域/近零网格的中间阶段相对无噪声输入仍未通过原有
`1e-5` stage guard，因此该独立诊断仍返回 1，不能称为全部精度测试通过。
历史未融合的 `relu_precision_run1/2/3.log` 同样失败；最终输出阈值通过，且本次
融合与原实现的完整输出残基完全相同。没有放宽阈值或隐藏失败。

可用已生成的三份日志重新执行证据检查（日志位于 `/tmp`，不是仓库资产）：

```bash
POSEIDON_RELU_LEAF_ACCURACY_LOG=/tmp/poseidon_relu_leaf_fusion_accuracy.log \
POSEIDON_RELU_LEAF_OFF_LOG=/tmp/poseidon_resnet20_leaf_fusion_off.log \
POSEIDON_RELU_LEAF_ON_LOG=/tmp/poseidon_resnet20_leaf_fusion_on.log \
python3 -m unittest discover -s bench/resnet20_s2c_first -p 'test_*.py'
```

本次 26 项 bench 检查全部通过；其中精度日志检查会确认原有 stage guard 的 FAIL
仍被如实报告，并不将这个 FAIL 改写为全域精度通过。

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
