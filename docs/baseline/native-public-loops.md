# Native helper 公开循环：实现完成，数值门禁部分失败

2026-09-17。新增 native helper/golden 内固定公开 range 循环的版本化支持。
8 个正确人工程序中 **7 个通过、1 个在密态数值比较层失败**；另一个错误循环上界
反例被正确拒绝。批次状态保持 failed，不能称作 8/8 或循环端到端全部通过。
本轮 agent_calls=0，不属于真实 Agent 生成覆盖。

## 系统位置与语义

输入仍为受限 Hecate Python 函数声明。新增 native_public_loops.expand 在完整
native c/p 检查前展开公开循环；随后沿用 decorated_functions 的类型检查及
可信 AST 分发，调用真实 Hecate frontend，再由 Dacapo 生成 CKKS/HEVM/CST。
不 exec/eval 候选代码；保留 @hc.func 的函数边界，不把它降格为普通 Python helper。

复用 function_construction.INTEGER_OPS 的公开整数运算定义，但不直接使用其
普通 helper 内联器，否则会丢失本阶段要保留的 native 调用边界。
新增请求 v7、native 契约 v5、native core plan v5；入口为
`--native-public-loops`。旧 native v1-v6 请求及 nf/na exercise 均不变。

允许：
- `for i in range(stop)`、`range(start,stop)`、`range(start,stop,step)`。
- 字面整数和已经绑定的循环下标构成边界、索引、rotation 参数及公开系数。
- 纯整数 +、-、*、//、%、一元符号在构造阶段折叠。
- 嵌套循环、负步长、零次循环、顺序复用下标；非空循环保留最后下标值，
  零次循环不修改此前绑定，也不凭空创建绑定。
- 循环累加、真实 helper 重复调用、object 数组算术组合。

限制：
- 不支持密文/Plain 参数决定循环次数，不产生密态动态控制流。
- 下标只读，不得遮蔽形参/常量/函数/保留名，内层不能复用活动的外层下标。
- 每个 range 最多128项，整数绝对值<=1048576，展开总次数及AST节点<=4096，
  嵌套<=16，仍受 native IR 资源门禁约束。
- 不新增 while、break、continue、for-else、循环内 return、增量赋值、
  任意调用或数组写入。即使循环为零次，禁用语法和危险调用也会先被拒绝。
- 限定版本不是全部 Python/Hecate 控制流；新契约不能与旧构造契约擅自混合。

类型计划保存原源码哈希、每个循环的源码位置/实际公开边界/次数、展开AST哈希。
真实调用事件保留原源码位置；同一位置的重复 helper 调用按实际次数审计。

## 人工密态结果

| 案例 | 实际检查 | 结果 |
|---|---|---|
| sum4 | helper 内 range(1,4)，三次rotation完成四槽求和 | **数值失败** |
| descending | 负步长range(2,0,-1)，三槽求和 | 通过 |
| nested | 内层range依赖外层公开下标 | 通过 |
| index_binding | 非空循环后接零次循环，保留最后索引 | 通过 |
| helper_repeated | 循环内两次真实native调用 | 通过 |
| array_loop | 循环内object数组乘法 | 通过 |
| zero_trip | 累加器不被零次循环改变 | 通过 |
| public_weight | 循环下标参与公开乘法系数 | 通过 |
| wrong_bound | 三槽求和误写成四槽求和 | 数值比较拒绝 |

8个正确程序均已真实编译、执行并解密，共32组输入、128个输出；
其中7个通过（28组、112个输出），失败的16个输出不得从分母静默删除。
另一个反例也有4组输入、16个输出。后端为 upstream SEAL HEVM CPU，
N32768、14×60-bit模数、tc128、scale/waterline40，无bootstrap。

冻结门限为 `abs(actual-reference)<=1e-5+1e-4*abs(reference)`。

sum4 最大绝对误差 `1.579301745589292e-5`、MAE `4.165565733763418e-6`；
两个零reference输出不满足绝对误差门限。未重抽密钥反复运行直到通过。
descending 最大绝对误差 `6.499353066125035e-6`，精度余量也并不宽裕。
其余六个正确程序的最大绝对误差均<=`3.348236754519007e-8`。

反例与原reference的最大误差 `1.0000010830204757`。
进一步对它自身的错误数学公式（四槽求和）比较，最大数值残差约
`1.01315393e-5`，也未过同一冻结门限。因此保留两个不同事实：
循环上界带来约1.0的模型语义误差，同时rotation路径还有约1e-5的数值误差。
证据测试检查误差分解，不放宽门限以强称反例精确实现了错误公式。

## 失败分层与独立最小诊断

Symptom：正确sum4在零reference处超差。

Confirmed fact：
- Earth IR确实是原输入的rotate(1/2/3)及三次加法，与公开循环展开一致。
- CKKS IR先从13个数据模数modswitch至1个，再执行rotation/add，scale仍为40。
- 原批次密钥按保留策略已删除，不能精确重放该随机密钥。
- 历史 object-array-cpu.md 已记录旧四槽求和接近门限，不能把它当本次新语法特有问题。

Minimal test：新增独立 `seal_rotation_precision_probe`，使用既有SEAL4.0.0，
固定两组新密钥、每组零/有符号/边界三种输入，记录72条阶段观测。
相同输入密文分别比较低level旋转、高level旋转及旋转后再modswitch。
它是诊断程序，不注册为Agent运行后端，不保存密钥，也不替代已有HEVM执行链。

| 阶段（前4个slots的最大误差，取固定诊断集最大值） | 测量 |
|---|---:|
| encode/decode | 9.1144e-13 |
| encrypt/decrypt | 7.6524e-9 |
| modswitch至level1 | 7.6524e-9 |
| level1 rotate(1/2/3) | 7.8753e-6 / 6.3835e-6 / 8.9655e-6 |
| level13 rotate(1/2/3) | 1.6627e-5 / 2.9429e-5 / 3.0739e-5 |
| level1求和 | 4.2540e-6 |
| level13求和 | 3.0797e-5 |
| level13求和后再modswitch | 3.0797e-5 |

Evidence-based inference：主要数值增量位于SEAL rotation/key-switch路径，
不是此次循环展开；仅推迟modswitch不是本诊断所支持的修复方案。

Unconfirmed：尚未证明所有密钥/输入的误差分布，也未确定达到冻结门限所需的
稳定precision budget。该新密钥诊断不是原失败密钥的精确复现。
生产compiler profile、安全参数和门限均未改动。

## 证据、复现与保留

原批次：
`/home/lhy/poseidon-work/results/native-loop-goldens-z4ul_cpj/report.json`

SHA256：
`a9c002c6901ca4e8bbfa1b2dbf2fec215a657fcb6f84c027989073db81cb2915`

诊断：
`/home/lhy/poseidon-work/results/seal-rotation-precision-eqqq7v6t/report.json`

SHA256：
`335be08ee55f82c4221d006ed0465fb775714cabe0ed7d6f31104f0141abef11`

```bash
cd '/mnt/d/Code Space/Poseidon'
# 人工密态门禁；可能因已记录的precision gap失败，不能修改门限掩盖
timeout -k 5s 1200s python3 scripts/baseline/run_native_loop_goldens.py
# 固定2组新密钥的分层诊断；不修改前一个结果，也不是验收通过
timeout -k 5s 230s python3 scripts/baseline/probe_seal_rotation_precision.py
```

诊断target使用既有SEAL4.0.0，仅在原生build目录按-j2编译；无新增安装。
全部9个密态案例共6,110,820,090 bytes（约5.69GiB）的可再生密钥已清理。
原随机密钥不可恢复；DSL、模型、IR、HEVM/CST、reference、解密数组及失败日志保留。
诊断密钥只存在进程内存，退出后释放。

初始34项定向回归通过。综合证据审计首次发现反例自身也未过数值门限，
随后改为明确保留这项已观测失败及误差分解，未调整数值阈值。
证据审计的通过仅说明保存的数据、原始失败和诊断可复核，不表示sum4被修好。
最终155项综合回归全部通过，无失败、无跳过，包括新循环、旧数组/starred契约、
既有付费native-function证据及独立rotation诊断。该测试数不是155个密态模型，
也不把本批7/8的数值结果重新标为8/8。

## 后续门禁

2026-09-17后续进展：已用原失败程序的Earth IR完成固定3组密钥、waterline40/45/50
的真实编译与HEVM对照，45的误差约缩小32倍；未更改本批失败或默认配置。
详见 [HEVM waterline精度对照](hevm-waterline-precision.md)。

先使用独立、明确标注的实验配置研究rotation的精度预算，再决定是否需要
版本化compiler profile。不得重写旧报告、删掉失败样本或用更多Agent修复掩盖
执行链精度问题。该原生循环契约的真实Agent生成覆盖仍为0。
完整DSL、native与普通helper控制流的组合、高层poly helper及真实bootstrap
仍未全部完成。所有修改保留本地，不commit/push/PR。
