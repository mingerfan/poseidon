# ReLU 模数位宽与跨段 scale 联合搜索（2026-09-11）

## 结论与状态

本轮找到全 30-bit 的 22-Q **静态候选**：6 + 6 + 7 + 3。
混合一个 25-bit 素数可将消耗总位数从 659.495 降至 654.529 bit，但仍为 22 个物理 Q。
有限搜索没有找到满足本模型约束的 20/21-Q 方案；这不是不可能性结论，更不是全局最优性证明。

仅新增分析脚本、公共参数账本和报告，没有修改应用、GPU 算子、自举实现、密钥配置或系数文件。没有执行 KeyGen/加密/解密/GPU，尤其没有进行任何零秘密密钥测试。6 个测试仅验证元数据，不是密态正确性测试。

前一轮 `/tmp/resnet20-scale-optimization.B8ZSoQ/` 已不存在。本轮程序从当前源码重建，使用新生成的实际素数，因此不能把所有数值当成上一轮同一素数链的严格复测。

## 当前源码依据

应用目录：`/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/`。

- `gpu_relu.cpp:92` 的 `build_odd_baby_tree` 产生固定分解树。脚本移植这一分解，不修改多项式阶数或重新拟合系数。
- `gpu_relu.cpp:244` 使用固定返回左操作数 scale 的乘法。
- `gpu_ckks_runtime.cpp:2354` 先 rescale，`:2401` 再乘编码 1 并 rescale 校正 scale。
- `src/poseidon/gpu/gpu_scale_planner.h` 仍以删除现有 Q 前缀末端素数为模型，没有更换 terminal base 的操作。
- `src/poseidon/gpu/gpu_parameter.cpp:79` 的上传检查允许值落在 uint32 范围内；NTT 模加有宽中间值。不能简单宣称当前库不支持所有 32-bit 素数，也不能据此证明每条最快 kernel 对所有 32-bit 素数都正确且同速。本轮没有执行这些 GPU 路径。

## 搜索约束和范围

- N=65536。素数满足 q ≡ 1 mod 131072，互不相同。
- 独立 ReLU 入口使用 Q38、scale=2^40，出口严格回到 scale=2^40。Q38 是隔离规划的入口，不是已经接入自举的新参数。
- 保留 degrees=[15,15,27]、原 odd-baby 树和 Chebyshev basis 关系，允许重新安排 rescale、加法与系数编码 scale。
- 第一、第二段的输出 scale 不必固定为 2^40；第三段与最后乘 x 联合定标。
- 报告中的最终候选要求中间密文 logscale≥40，非平凡系数/对齐明文 logscale≥40。没有通过把工作精度下限降为 30 得到这两个候选。
- 混合位宽试探范围 23–30、23–32 bit；实际素数按指定位宽生成。尝试每个配置 1200 次联合随机变异，以及全 30-bit 候选附近 800 次变异。
- 段间 scale 进行了整数、0.1-bit 局部网格和更宽边界网格搜索。basis rescale 采用有限工作下限网格，不是每个节点的所有可能计划。
- 搜索使用 log2 浮点算术；候选由独立遍历复核等式、物理素数收支、分支层数、模数容量和 scale 下限，等式容差 1e-7 bit。
- 不读取系数幅值作误差分析：叶子对应原树中的系数位置，编码 scale 约束对其统一施加。没有验证真实 CKKS 编码舍入、key-switch 噪声、抵消误差和区间外行为。

## 可复核的候选

| 阶段 | 全30-bit消耗Q | 全30-bit输出logscale | 混合候选消耗Q | 混合候选输出logscale |
|---|---:|---:|---:|---:|
| 输入 | — | 40 | — | 40 |
| 第一个P15 | 6 | 40.1 | 6 | 40.2 |
| 第二个P15 | 6 | 44 | 6 | 44 |
| P27 | 7 | 89.8711223871 | 7 | 84.9047021356 |
| 加0.5，再与原输入相乘并rescale | 3 | 40 | 3 | 40 |
| 合计 | 22 | — | 22 | — |

共同路径：Q38 → Q32 → Q26 → Q19 → Q16。

全 30-bit 候选：实际消耗 659.4949322133 bit，最小系数/对齐编码 logscale=40.0307968412，最大规划中间 logscale≈179.765。最后乘法前为 logscale≈129.871，再除去三个实际素数后返回 40；这三个 Q 没有漏算。

混合候选：按删除顺序（高层到低层）为 `[30 × 20, 25, 30]`，消耗 654.5285119618 bit。后续未删除部分是另外 16 个 30-bit 素数。注意：`candidates.json` 是 top-first，而库 Q 数组通常是 bottom-first，不能原样填入工程。

实际素数、basis 和全部树节点的 work/output/scale/pre-scale/encoding-scales 均保存在 `candidates.json`。

以上是全30-bit样本与混合样本，不表示统一更大位宽必然更好。本轮全32-bit的边界网格最好为23Q；不同素数排序与更自由的节点调度仍可能改善。没有得到经过验证的20/21Q结果。

## 如何理解 14×40/32

如果假定 14 次逻辑深度各净消耗约40bit，理想预算是560bit。按32bit装填是ceil(560/32)=18，按30bit装填是ceil(560/30)=19。这是忽略RNS离散性、求值分支和精度约束的目标参考，不是当前求值图已经可达的下界证明，也不包括输出要保留的Q、P或自举预算。

平方时 logscale 按 s_next=2*s_current−本次删除素数log总和 演化。残留的scale会进入后续乘法，并非总能像独立bit预算一样均匀摊销。例如理想30bit下，40→50→40的两次平方分别丢1、2个Q，共90bit，而不是两次40bit共80bit。因此需要联合考虑系数编码、basis、段间scale与实际素数，不能只除以机器字宽。

更小的素数有时减少位数浪费，却不减少物理limb数量：本轮混入25bit素数就是例子。

## 相关一手资料与适用边界

1. [Grafting: Decoupled Scale Factors and Modulus in RNS-CKKS](https://eprint.iacr.org/2024/1014)：通过可复用辅助基解耦scale与模数。论文还报告了在其比较算法参数中，分轮scale选择降低模数消耗；不能直接外推本工程ReLU必能达到同样结果。
2. [Cheddar论文，第3节](https://arxiv.org/html/2407.13055#S3)：其25/30素数系统使用rational rescaling。典型40bit调整可由删除3个约30bit素数并引入2个约25bit素数组成，净降低90−50=40bit；之后按循环重新安排两类素数。它不是固定前缀drop-chain上单纯修改位宽。
3. [Cheddar官方实现](https://github.com/scale-snu/cheddar-fhe)：提供32bit执行与rational rescaling参考。仓库明确提醒其client-side功能仅作测试用途，本轮没有使用其密钥或加解密实现。

本地枚举还确认N=65536时20bit NTT素数只有786433一个，不能重复14次用不同20+20素数对来简单构造560bit链。

rational rescaling不是无条件ModRaise，也不是自举。其缩放与基转换需要一起执行，净模数/精度预算继续下降，不会凭空刷新噪声。接入当前前缀布局需要修改有效RNS基、BConv、key-switch索引/密钥兼容和scale计划；需要在P·Q_max下重新检查安全预算。应先做独立设计，不能在现有应用里直接替换。

## 复现

无需第三方Python依赖，不加载任何HE或CUDA库：

```bash
python3 -B Doc/analysis/resnet20_scale_chain_20260911/test_metadata.py
python3 -B Doc/analysis/resnet20_scale_chain_20260911/verify.py
python3 -B Doc/analysis/resnet20_scale_chain_20260911/search.py --mode boundary --bits 30 --actual --seed-scales 40.1,44
```

6/6 测试通过，包括篡改scale、漏计一个Q、分支层数错配的拒绝测试。

## 下一阶段建议

先将22-Q候选作为可追踪的静态基线，进一步搜索每个basis节点的rescale时机、实际素数排序及更精细的段间scale，而不是仅贪心保留某个统一下限。20/21是值得继续搜索的目标，但尚未获得证据支持其在同等密态误差下可行。

若普通前缀链继续受离散性限制，再单独评估25/30 rational rescaling，不直接混入现有路径。任何候选都要先在合法非零秘密密钥下验证，再接入59阶自举及连续ReLU/卷积闭环。本轮没有重算完整自举链，不能直接宣布Q62可以缩为Q56。
