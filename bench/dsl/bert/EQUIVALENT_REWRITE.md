# v4候选：等价算术与共享旋转源Rescale

状态更新：algebra＋shared候选已真实完成单block **9.107562999s**（1warmup＋3测、
实际23Boot），同窗再加QK为8.222786937s。报告在
`test-results/bert-a800-20261115/v4-a800/20261115T140945Z/FINAL-REPORT.md`。
下面计数表仍是静态CPU计划分析，不用计数外推GPU时间。冻结v3的12层
119.297476332s属于独立snapshot，不混用本候选。两个新选项均默认关闭。

## 1. 显式 `--algebra-rewrite`

实现：`equivalent_rewrite.py`，在DSL Expr图和物理lowering之间重建一个独立图。

- 整个slot向量都为−1：MulCP改为Negate；整向量为1：移除identity乘法。
  **短prefix、head/row mask的零尾不会被误识别为全slot标量。**
- 只识别三轮reciprocal Newton的固定首步：
  `((d*(-1/T))+2)*(1/T)` → `d*(-1/T²)+2/T`。
- 不做任意BSGS系数分配/融合；不删除Newton迭代或exp/GELU/LN项。
- 所有dense原始weight/bias逐字节保留；重新用真实prime-aware lowering决定scale/level。
  不是metadata-only scale改写，也不声称CKKS ciphertext或噪声逐字节相同。

实际T128、一完整block的CPU重写计数：首步fold6个，全slot−1转Negate12个，
LN默认center=1的identity乘法2个。两层tiny polynomial oracle在容差内一致，
剩余两轮Newton仍有全部4个CC products。

## 2. 显式 `--shared-rotation-rescale`

物理lowering统计reachable图的引用次数。对被多个算子复用的pending source，
在其旋转前做一次真实Rescale并共享；包括绕过的Boot hint之后的共享源和AV diagonal sum。
不在非共享源上普遍提前Rescale，避免破坏原有积和共享策略。

例如v3中Q/K/V mask结果经过不保留的stage hint后仍pending，多个旋转各自到消费点
才Normalize；新选项让这些旋转复用同一个已经Rescale的源。AV diagonal的baby旋转
也复用一个Normalized source。数学图不变，Rescale与keyswitch的噪声顺序可能不同。

## CPU计划结果（不是GPU收益）

固定T128/H768/12×64/F3072、1block、Q52/P11/ratio256、相同28,339,200B dense权重：

| 配置 | Boot | MulCP | Rescale调用 | Rotate | compute/opaque调用 |
|---|---:|---:|---:|---:|---:|
| 冻结v3 | 29 | 4080 | 3634 | 7528 | 26545 |
| algebra | 23 | 4055 | 3609 | 7528 | 26501 |
| algebra＋shared rotation rescale | 23 | 4055 | 3393 | 7528 | 26285 |

三者均27个prepared Linear调用，内部55,188个真实dense CP terms；1624个普通MulCC
和Relin均保留。最后3CT均L12/Q13、相同canonical scale。

共享旋转源候选的Rotate输入active-Q limb总和60380，v3/algebra为60960；这是计划
数据宽度指标，不是keyswitch总次数或GPU延迟。额外216次Rescale调用被共享掉。

标准PlanVerifier均通过；候选alpha需要192个key requirements，共享旋转版本179个，
这不是实际CPU key basis数量。单层权重文件与v3逐字节相同。v4冻结时98项CPU测试通过，新增三CT跨层SSA及默认/组合CLI flags回归。

## 复现

```bash
python3 -m unittest discover -s bench/dsl/bert -p 'test_*.py'
python3 bench/dsl/bert/generate_aligned.py --output-dir NEW_CANDIDATE \
  --scope encoder --periodic-encoding --algebra-rewrite --shared-rotation-rescale \
  --profile-report test-results/bert-a800-20261115/compact-native-build/fast-validation/profile-q52-r256.json
```

CPU产物：`test-results/bert-a800-20261115/{algebra-v4-cli,algebra-shared-v4-cli}/`。
冻结12层v3已完成119.297476332s online中位并释放GPU；主线没有并行跑GPU。
用户现已长期授权继续测试；上述真实单层结果已取得。后续轮仍先做资源准入与有界清理，
不能把新CPU计划计数称作已经取得的毫秒收益；不能把少6次Boot直接乘早期单Boot108ms诊断来宣称加速。
