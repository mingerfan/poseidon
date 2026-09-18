# 上游 HE_Concat：packing 反例与局部修复

## 结论与可信边界

固定 Dacapo commit `4616402710f39df3e5f5bd7930a6c036025aaac3` 的
`HE_Concat -> shapeClosure["CC"]` 在跨密文边界及多密文尾部拼接上存在可复现的
槽数据丢失。本地已修正尾部掩码、最后一段覆盖条件，并拒绝不匹配的分支几何。

**证据层级：明文槽布局诊断，不是密态运行或 Agent helper 验收。**
当前隔离环境缺少上游要求的 `einops==0.7.0`；未擅自安装。
诊断执行实际仓库中的可信函数源码，只为其使用的五个 einops packing pattern
提供显式 Torch reshape/permute/repeat 适配。独立参考使用逐坐标索引，不复用
该适配或上游函数。实际 einops 库、Hecate Expr tracing、HEVM 执行尚需后续验证。
PlainSlots 是明确标记的明文向量，不是加密实现，不冒充 FHE。

生产 Agent 允许集合未增加 HE_Concat；语义清单仍保留 helper 未验证的状态。
本地已有逻辑 concat 的 Agent 结果不能移植为这里的 packing 契约证据。

## 系统位置与布局

- 调用示例：Dacapo `examples/benchmarks/SqueezeNet.py` 的两个 expand 分支，
  通过 `CascadeConcat` 推导拼接 shape，调用 `HE_Concat(close,out1,out2)`。
- 包装层：`python/poly/poly/Func.py: HE_Concat` 转发到 close["CC"]。
- 实现层：`python/poly/poly/MPCB.py: shapeClosure/ConcatSelecting/Concat`。
- 上游输入：两组已按相同 channel-interleaved 布局编码的对象数组，不是普通逻辑张量。
- 下游输出：与拼接后的 channel 布局一致的对象数组；后续 Conv/packing 操作依赖它。

对当前 CascadeConcat 接受的等通道分支，令：

- `n = ci*hi*wi`：一个输入分支的有效元素数；
- `nt`：一个诊断向量的槽数；
- `ni = ceil(n/nt)`，`no = ceil(2*n/nt)`：输入/输出密文数量；
- `k`：通道与空间的交错因子，支持的分支要求通道数是 `k*k` 的倍数；
- `pi/po`：输入/输出槽中的重复副本数，由上游 InferShapes 确定。

单个通道元素 `(c,y,x)` 的 pack 索引为：
`(((tile*height+y)*k+subrow)*width+x)*k+subcol`，其中
`c = tile*k*k + subrow*k + subcol`。参考还包含除以 `bb` 的缩放、补零和重复。
诊断将 branch OP、concat MPP 和输出 OP 分别与独立索引 oracle 比较。

`MPCB.roll(v,s)` 调用 `v.rotate(-s)`；诊断按 Hecate 正数左旋约定实现。
整密文对齐时直接拼接对象数组；非对齐时旋转 B 并用 FF/BB 合并尾部。

## 两个反例与修复

1. **FF 选择长度错误**：应保留 A 的有效尾部 `r`，原代码使用 `tt-r`。
   当 nt=16、n=10 时原代码仅保留前6槽，A 的第6..9槽丢失。
   本地修正为前 r 个1，后 tt-r 个0。
2. **末尾赋值覆盖已合并结果**：循环已经产生 `2*ni-1` 个输出，
   原条件 `ni != no` 仍会覆盖最后一个已拼接密文。
   当 nt=16、n=24 时，第三个密文中已拼好的 B 数据被覆盖为零。
   本地仅在 `no == 2*ni` 时补额外 carry，并使用 FF 遮掉重复移入的内容。

另有 `CascadeConcat` 只检查等通道而忽略右分支空间/packing 参数的问题。
现对 nt、bb、ko、ho、wo、no、po 不一致明确抛出 ValueError；
原有不支持通道配置的返回行为未改。

新增 carry 乘掩码可能改变编译后的乘法/level安排，需要真实编译验证；
此处不推断性能收益或额外深度已经可执行。

## 验证

修复前10个边界案例：6通过、4失败；修复后10个全部逐槽精确一致。
失败项为 crosses_output_boundary、multi_partial_half、multi_partial_small_tail、
interleaved_multi。空间形状不一致原先被接受，修复后明确拒绝。

另外覆盖480组组合：nt为8/16/32，k为1/2，channel tiles为1..5，
height/width各为1..4。全部与独立参考逐槽一致。
这些 nt 是**明文向量长度**，没有降低真实 CKKS 安全参数进行测试。

回归测试还直接通过只读 `git show` 加载固定原始commit，重新产生原先4个失败，
防止把人工构造的替代算法当成上游反例。周边回归64项通过、0跳过、0失败，
包括既有逻辑concat、reshape、BN、模型转换、语义清单及最新v22付费结果离线复核。

修复前报告：
`/home/lhy/poseidon-work/results/upstream-concat-layout-r0sbzecw/report.json`

SHA256：`6cbe320a636f0d20eaf34fabbb9d814bef271c9ecba1782b6cc554415dafa24a`。

最终修复后报告：
`/home/lhy/poseidon-work/results/upstream-concat-layout-ihcjr258/report.json`

SHA256：`25aba5937753eb9f98e3764adf42a767aeab754367602c1e582ca2d0d7fff2a8`。

报告记录了实际读取的 MPCB.py 与 Func.py 哈希。旧报告没有覆盖。
本轮新付费API调用0次，新密态执行0次；没有理由在确定性helper链路未验证时付费生成。

## 文件与复现

- 原始实现的局部修改：`third_party/dacapo/python/poly/poly/MPCB.py`。
- 可保存于主仓库的补丁：`scripts/baseline/patches/dacapo-concat-layout.patch`。
- 诊断与测试：`scripts/baseline/probe_upstream_concat.py`、`test_upstream_concat.py`。
- 语义清单只更新证据说明，没有开启新的 Agent helper 权限。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 220s python3 scripts/baseline/probe_upstream_concat.py
git -C third_party/dacapo apply --reverse --check \
  '/mnt/d/Code Space/Poseidon/scripts/baseline/patches/dacapo-concat-layout.patch'
```

后一个命令只验证补丁已经存在于当前源码，不撤销修改。
固定gitlink不变；子模块原有expr.py、frontend.cpp修改保留；没有commit/push。

## 下一门禁：请求批准一个隔离依赖

版本依据：Dacapo `requirements.txt` 第31行 `einops==0.7.0`。
PyPI官方元数据中的纯Python wheel：`einops-0.7.0-py3-none-any.whl`，44599字节。
SHA256：`0f3096f26b914f465f6ff3c66f5478f9a5e380bb367ffc6493a68143fbbf1fd1`。

拟缓存到 `/home/lhy/poseidon-work/cache/`，校验后仅安装到既有
`/home/lhy/poseidon-work/venvs/hecate-2.0.1-cpu`，采用no-deps/no-index，
不sudo、不改系统Python、不下载Torch或其他依赖、不改LLVM/MLIR/CUDA。
安装批准后，首先使用真正einops复验相同矩阵，再做actual Expr tracing和真实密态验证；
通过后才设计生成接口和付费Agent覆盖。完整DSL目标尚未完成。
