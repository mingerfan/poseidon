# 固定密钥的 HEVM waterline 精度对照

后续进展：已完成 [8个计算图的40/45配对实验](hevm-precision-graphs.md)，
包括密文平方、Linear/MLP和分支图，48次真实HEVM执行通过；同时发现乘法图
的rescale安排发生变化。本文保留原单图实验的范围与结论，默认配置仍为40。

2026-09-17。本实验沿用真实 Dacapo compiler 与 upstream SEAL HEVM CPU，
比较原失败四槽求和程序在 waterline40/45/50 下的数值误差。预先固定3组新密钥，
每组同一套密钥运行全部3个配置，所有结果保留，不自适应追加重跑。

**结论：** 本实验中45相比40将最大误差降低约32倍，50相比40降低约1024倍。
三组新密钥的40配置恰好也通过，但不能抹去原失败，也不能据此声称40已经稳定。
默认Agent入口仍为40，原compiler JSON、原数据和原失败报告均未改变。
本轮0次API；不是新的Agent生成、全语义证明或生产配置切换。

## 动机与被固定的变量

前一批 `native-loop-goldens-z4ul_cpj` 的sum4在零reference处超差，
最大绝对误差1.579301745589292e-5。随后独立SEAL诊断显示误差主要在
rotation/key-switch阶段增长，而不是modswitch。这两个结果继续保留。

本次使用其子运行 `candidate-replay-t9aglvbt` 中已保存的真实Earth IR，
不是重写更容易通过的求和程序，也不重新调用模型生成代码。

原报告SHA256：
`4ea525b5baa833cf40df36050a00b01f683a86c5a6e307fc28d3c5e05f1f00e0`

固定变量：

- 同一份Earth IR、CST和独立reference；输入为原来的零、有符号、seed42和边界四组。
- SEAL4.0.0、N32768、14×60-bit模数及其实际数值、tc128。
- 密钥策略与原运行相同：rotation keys覆盖-3/-2/-1/1/2/3。
- 每组40/45/50使用相同实际密钥文件，并在执行前后校验哈希。
- 每次仍进行新的正常随机加密，不声称三种scale复用了同一个密文。
- LLVM/MLIR18.1.2、原compiler JSON、原SEAL HEVM二进制。
- 数值门限 `abs(actual-reference)<=1e-5+1e-4*abs(reference)`。

唯一实验配置变化是 `--waterline`。三份产物实际arg/result log2(scale)
分别为40、45、50，arg level均13，result level均1。
逐指令对照确认有效HEVM指令完全相同；仅跳过编译器未初始化其非opcode字段的
tensor.empty占位指令。rotation仍为1/2/3，modswitch及加法顺序未改变。

这里没有新后端：编译与执行均使用既有bubblewrap边界和可信worker。
运行时只接收input数组，不挂载reference。使用原请求仅为了固定I/O与layout，
报告明确标记 `source_request_reused_for_io_only`；不能把它称为Agent在新配置下的生成。

## 结果

| waterline | 新密钥组通过数 | 输入执行组数 | 输出值数 | 最大绝对误差 | MAE |
|---|---:|---:|---:|---:|---:|
| 40 | 3/3 | 12 | 48 | 9.070249937966522e-6 | 2.0314567544095734e-6 |
| 45 | 3/3 | 12 | 48 | 2.8352518217432117e-7 | 6.366160717266553e-8 |
| 50 | 3/3 | 12 | 48 | 8.849117327791589e-9 | 1.977960173897897e-9 |

合计9次真实HEVM执行，每次处理4组输入，共36组、144个输出。
没有bootstrap、输入/reference回填、门限放宽或安全参数缩减。

分组最大误差：

| 密钥组 | 40 | 45 | 50 |
|---|---:|---:|---:|
| 0 | 3.2889076893581427e-6 | 1.04258287447756e-7 | 3.2210192291159956e-9 |
| 1 | 5.758232581642986e-6 | 1.8014281560930306e-7 | 5.603283634936794e-9 |
| 2 | 9.070249937966522e-6 | 2.8352518217432117e-7 | 8.849117327791589e-9 |

Confirmed fact：在这3组密钥及固定数据上，45/50均通过且显著降低误差，
编译器实际scale和运行时实际scale与实验设定一致。

Evidence-based inference：增大编码scale是当前rotation精度问题的可行修复方向。
CKKS解码会除以scale；40到45是2^5倍，观察到的改善与这一趋势相符。
这不是在数学模型中改变权重或算法。

Unconfirmed：不能从一个纯加法/rotation程序推断乘法、深层MLP、完整高层helper、
所有输入或所有密钥均满足门限，也没有统计失败概率上界。
更大的scale还会影响乘法后的scale/level安排及数值范围余量；50不应只因
本例误差更小就自动设为默认。

## 证据与复现

报告：
`/home/lhy/poseidon-work/results/hevm-waterline-diagnostic-ds51hu27/report.json`

SHA256：
`142e95ac072b8b539c6e313365fc21a9452d73d04ad1834481bc381410cd89b4`

`probe_hevm_waterline.py` 固定实验对象、3个waterline和3组密钥，无外部模型调用，
无参数覆盖或until-pass循环。它保留编译模板、每组解密数组、原reference副本、
源程序/原报告/二进制哈希，以及生产者源码快照。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 950s python3 scripts/baseline/probe_hevm_waterline.py
```

该命令生成独立实验目录，不改变默认Agent配置。复现会使用3组新的随机密钥，
结果可能与本次不同，必须保留全部结果。

新增4项计划/证据测试验证范围、同密钥记录、三个真实scale、指令一致、
四组输入、reference不可见、独立重算、不可变产物和清理结果。
包含既有语义/真实付费证据与原数值失败审计的159项综合回归全部通过，无跳过。
回归通过不是默认scale40已经修复的证明。

## 空间和下一门禁

三套密钥共2,036,940,030 bytes（约1.90GiB）已按既有保留策略删除；
原随机密钥不可恢复，可以重新生成新的密钥。其余证据保留。
无需新依赖、安装、sudo、大型下载或新构建；沿用已有隔离环境。

下一步先把45作为候选实验精度，覆盖含c×c、Linear/MLP、较深图和混合运算的
真实编译执行，再考虑版本化的production compiler配置。
新配置标识应同时绑定compiler JSON与waterline，不能只靠现有JSON哈希把40/45
当成同一配置；旧请求/报告保持不变。不得由Agent自行修改这些参数。

当前production_waterline=40，production_profile_changed=false，
all_semantics_proven=false。完整DSL支持目标仍未完成。
修改全部留在本地working tree，未commit/push/PR。
