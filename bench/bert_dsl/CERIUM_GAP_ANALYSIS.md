# 当前BERT DSL与Cerium的差距分析

## 1. 实测结论与口径

| 实现 | 完整1层在线/内部runtime中位 | 相对Cerium延迟 |
|---|---:|---:|
| 当前Poseidon DSL | **9.839523156 s** | **162.4%** |
| Cerium | **6.059 s** | **100%** |
| 差额 | **3.780523156 s** | **多62.4%** |

测试条件：指定A800、128 tokens、hidden768、12×64 heads、FFN3072、完整1个encoder block，均为1 warmup＋3次测量。不含embedding、pooler和分类头。

**模型形状相同，但HE参数、近似计算图、刷新策略和CUDA Graph策略不同；这不是等算法、等精度、等参数比较。**

总体判断：主张量packing已经基本对齐。尚存的明确差异集中在算子执行粒度、旋转/密钥切换、逐阶段模数预算及Boot实现。但尚未完成GPU级分阶段profiling，不能把3.781s准确拆成Boot、Attention、Runtime等各自的耗时。

## 2. Packing：主布局基本对齐，统计量与Boot配对仍有缺口

| 项目 | 当前实现 | Cerium |
|---|---|---|
| Hidden | 3 CT，每CT为[128,256] | 相同主要布局 |
| Q/K/V | 各6个双head gap CT | 相同主要布局及head顺序 |
| Attention score | 6个完整cyclic score CT | 相同主要布局 |
| FFN | 12 CT | 相同 |
| LayerNorm | 跨全部feature tiles计算真实均值/方差，共享统计CT | 同样利用共享统计 |

当前不应再把主要问题概括为“仍使用16个hidden CT”。但布局对齐不等于执行效率对齐：

- Softmax分母、逆数仍按6组CT分别处理，尚未跨组压紧高度重复的统计量。
- 没有Cerium的双real流复数配对Boot；Cerium有24个core各刷新两条独立CT流。
- QK/AV虽已采用cyclic BSGS表达式，其后端仍主要是普通Runtime算子，而非完整的prepared批量执行。

实现位置：[cerium_packing.py](cerium_packing.py)。

## 3. 执行粒度：当前融合主要覆盖Linear

当前完整一层的普通Runtime算子计数：

| 算子 | 次数 |
|---|---:|
| Rotate | 7,528 |
| MulCC | 1,624 |
| Relinearize | 1,624 |
| Rescale | 3,634 |
| AddCC | 7,937 |
| MulCP | 4,080 |
| Native Boot | 29 |

一层合计26,545个compute调用，**其中包括27个opaque prepared Linear调用**。这些Linear内部仍真实执行55,188个dense plaintext products，不能把27个调用误说成27个算术操作。

| 当前实现 | Cerium |
|---|---|
| QK、AV、非线性大部分展开为独立Runtime算子 | 编译器处理向量算子、batch、hoisting和融合 |
| 普通Rotate逐条调用，采用binary key策略 | 大量使用HoistedRotate、批量旋转和RotateAccumulate |
| 小算子涉及独立提交、事件与资源管理 | 子函数启用CUDA Graph，main未capture |
| Linear已prepared融合 | 融合与静态调度范围更广 |

这意味着当前Attention中的共享源旋转，不能自动获得与Linear相同的批量hoisting收益。Binary旋转还可能展开为多次实际keyswitch。

### 已有的时间证据

六类Linear同步callback的正式样本中位约为：

| 矩阵 | Callback wall |
|---|---:|
| Q | 77.098 ms |
| K | 79.046 ms |
| V | 78.900 ms |
| O | 142.610 ms |
| FFN-up | 416.972 ms |
| FFN-down | 183.016 ms |

每轮六类callback总和的中位为978.174ms，约占online的9.9%。**Callback包含输入ready等待与输出同步，不是isolated GPU kernel时间。**

独立完整FFN-up已做到140.879ms online，而Cerium对应program.run wall约281.463ms，因此不能笼统归结为“Poseidon所有Linear都慢”。即使消掉当前全部Linear callback，也不足以填平3.781s差额；后续应重点检查Attention、Boot及Runtime调度。

实现位置：[compact_native/](compact_native/)、[native_runner.cpp](native_runner.cpp)。

## 4. 模数链：总Q接近，不代表逐阶段工作量相同

以下Q表示active prime数量，不是Poseidon的level索引：

| 线性层输入 | 当前active Q | Cerium active Q |
|---|---:|---:|
| QKV | 13 | 13 |
| O投影 | **19** | **5** |
| FFN-up | **19** | **5** |
| FFN-down | **11** | **5** |

主要区别：

- 当前使用约56-bit统一canonical scale和双prime产品调度。
- Cerium采用更细的prime乘积尺度、56/84-bit混合排程及`/1000`、`/1024`等规范化。
- 当前某些刷新发生得更早，让后续大矩阵在较高Q上执行；Cerium常让线性层先在低Q运行，再刷新输出。
- 两边encoder边界均为Q13，但当前实际scale_log2约55.3701，Cerium约83.0647。

**少Boot不一定更快：提前刷新可能减少次数，却增加后续乘法、旋转和keyswitch的RNS工作量。**

例如，当前独立FFN-up输入Q5，encoder里的FFN-up输入Q19；这也是不能把独立140.9ms直接替换进encoder耗时分解的原因。Callback还有等待因素，不能直接由Q数量推导延迟倍数。

实现位置：[cerium_lowering.py](cerium_lowering.py)、[generate_aligned.py](generate_aligned.py)。

## 5. Special-P与keyswitch：当前仍较粗粒度

| 项目 | 当前实现 | Cerium |
|---|---|---|
| Special-P | 普通执行基本沿用固定P11 | 按level、操作类型动态选择 |
| 普通旋转 | Binary分解 | 批量/hoisted旋转及相应密钥 |
| Boot家族 | 当前统一profile | b33、b35两类，并裁剪不同输出预算 |

以普通dnum1路径为例，Q5/P11约处理16个QP limbs，而Q5/P5约处理10个；Q3/P11与Q3/P3则是14对6。这是数据宽度差异，**不是实测延迟倍数**，但意味着额外ModUp、ModDown和keyswitch工作量。

动态P不能简单截断现有evaluation key：需要对应子参数、新生成的evaluation keys及正确的level-specific绑定。SDK已有部分level-specific基础设施，但当前通用BERT路径尚未完成这套机制；相关配置目前主要出现在manual ResNet20路径，并不是BERT现成可开的开关。

接口参考：`src/poseidon/gpu/gpu_bootstrap_profile.h`。

## 6. Boot：次数更少，但工作内容和核心实现不等价

| 指标 | 当前实现 | Cerium |
|---|---:|---:|
| 独立CT流刷新 | 29 | 107 |
| Boot core执行 | 29 | 83 |
| 双流配对 | 无 | 24个core处理双流 |

Cerium计数由真实compiled DAG递归展开：

```text
18 b33 + 41 b35_scalar + 24 b35_vec2×2 = 107条CT流刷新
18 + 41 + 24 = 83次core执行
```

当前次数较少，部分源于近似图更轻：

- 当前Softmax减均值，采用exp4、Newton3 reciprocal。
- Cerium包含加密max/sign、exp6/isqrt等路径。
- GELU和LN近似也不同。

因此不能说“我们完成了Cerium同一条近似计算图，只是把107次Boot优化成29次”。

当前profile对齐了d30、DA3、K16、ephemeral h32、ratio256等参数，但DFT分组、Hanki系数、scale约定、动态P和家族选择未完全复现Cerium。需要分别测量C2S、EvalMod、S2C、规范化及密钥切换，才能定位Boot内部效率差异。

现有`boot_submission_ms`是CPU提交/等待统计，不是独立GPU kernel耗时。早期单Boot诊断使用ratio32，也不能直接乘以29来归因当前ratio256整层的Boot时间。

详见：[CERIUM_ALIGNMENT.md](CERIUM_ALIGNMENT.md)。

## 7. 已定位的具体冗余：Newton表达式规范化

当前`cerium_packing.py`倒数路径的第一步与修正项可写为：

```text
第一步：(2 - d/N) / N
后续修正：d*r*(-1) + 2
```

保持三次Newton迭代及数学公式不变，可等价改写为：

```text
第一步：2/N - d/N²
后续修正：2 - d*r
```

这会合并第一步的公开系数乘法，并避免用消耗level的MulCP(-1)表达取负。

已在不修改源码、不调用GPU的临时CPU对照中检查：

| 指标 | 当前计划 | 等价代数改写候选 |
|---|---:|---:|
| Boot | 29 | **23** |
| 预算耗尽触发Boot | 9 | **3** |
| MulCP | 4,080 | 4,056 |
| Rescale | 3,634 | 3,610 |
| Negate | 0 | 18 |
| Compute调用 | 26,545 | 26,509 |
| 最终level | 12/12/12 | 12/12/12 |

Tiny完整block的slot oracle最大差异为0。

**这只是CPU表达式与调度验证的新候选，没有合入当前9.840s版本，也没有GPU延迟结果，不能宣称已加速。**

## 8. Online之外的初始化与常驻差异

当前在线中位为9.840s，Runtime wall中位为11.618s。普通PT使用Host-only缓存，每轮仍按原计划真实H2D上传；融合Linear权重已经GPU常驻。

Cerium有常驻PT和remappable绑定机制。当前显存余量已增加，可以重新测试普通PT也GPU常驻，主要改善请求wall，而不应包装成在线算术加速。

两种wall也尚未完全同口径：当前Runtime wall包含每轮Runtime setup/init，但不含最终客户端解密；Cerium另有H2D/run/D2H指标。需要单独统一cold请求、warm请求和在线执行边界。

## 9. 精度与证据边界

当前完整层最大polynomial-reference误差约6.41，精度未通过；Cerium本轮也没有任务精度验收。有限输出不等于正确BERT推理，不能把现有结果解释成等精度系统比较。

尚不能确认：

- 3.781s差额中Boot、Attention、CPU调度各占多少。
- 改成direct旋转密钥、动态P或CUDA Graph各自能节省多少时间。
- CPU计划23 Boot候选的真实GPU耗时及数值变化。

## 10. 后续优先级

1. 验证上述Newton等价改写，先取得真实单层A/B结果。
2. 对同一冻结计划测试普通旋转direct vs binary，记录key内存、setup和online。
3. 做GPU分阶段profiling：QKV、QK、Softmax、AV、O、两次LN、FFN和Boot内部；同时记录实际keyswitch数、NTT/ModUp/ModDown launches及CPU launch空洞。
4. 根据profiling推进Attention prepared融合、动态P及子函数CUDA Graph。
5. 再优化逐阶段scale和刷新位置，不以单纯减少Boot次数作为性能结论。

**结论：当前主张量布局已经接近Cerium，但后端执行粒度和逐阶段物理调度还未达到同等成熟度。明确的结构差异已经找到，具体耗时归因仍需profiling。**

## 原始证据

- [本轮汇总报告](../../test-results/bert-a800-20261115/README.md)
- [当前encoder完整报告](../../test-results/bert-a800-20261115/aligned-encoder/report.json)
- [当前物理计划统计](../../test-results/bert-a800-20261115/compact-cyclic-q52-r256-selected-v3/schedule.json)
- [当前Linear物理合同](../../test-results/bert-a800-20261115/compact-cyclic-q52-r256-selected-v3/compact-linear.json)
- [Cerium原生结果](../../test-results/bert-a800-20261114/cerium/RESULTS.json)
- [Cerium编译DAG审计](../../test-results/bert-a800-20261115/cerium-audit.json)

`test-results/`是被Git忽略的本地产物，新clone不自带这些报告。本文仅保存分析，没有启动GPU实验或修改实现。
