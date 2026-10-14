# 固定 SEAL 4.0.0 的产物检查 v2

适用代码：scripts/baseline/seal_artifact_gate.py；固定 Dacapo gitlink、语义补丁、
SEAL 4.0.0 与原 tc128 参数保持不变。本检查不是数值正确性证明。
HEVM/SEAL runtime 和编译器均未修改；seal_cpu_golden.py 的改动仅为执行前校验及元数据观察。

## 参数与三个 level 概念

运行期只读检查模块 artifact_parameters.cpp 用已安装 SEAL 加载实际 parm.seal，
核验 CKKS、N=32768、tc128、14个锁定素数和完整数据链，并记录公开参数文件哈希。
不加载私钥、不生成参数、不改变 runtime。主机预检核验可信 key helper 输出的公开参数，
沙箱执行时再次核验实际加载的文件，与固定 profile 逐素数相同后才允许执行。

HEVM level = 数据素数数量；SEAL chain_index = 数据素数数量 - 1。
最高数据层为13个素数/index12；第14个素数是密钥切换特殊素数，不算数据容量。
各层容量使用 Python 整数精确乘积的 bit_length，并与 SEAL context 的实际 bit count 交叉验证。
当前锁定链恰好每层得到60×level，但检查逻辑不再依赖该公式，也不接受其他安全参数。

## 操作约束

| 操作 | 固定实现对应检查 |
|---|---|
| 向量编码 | 正有限scale；int(log2(scale))+1 < 当前模数乘积位数；实际编码器另查值/FFT系数容量 |
| HEVM输入与常量编码 | 实际在最高数据层编码，再保持scale逐层modswitch至声明level |
| 密文/密文、密文/明文乘法 | 双精度scale相乘；int(log2(product)) < 所在层容量，首次越界即拒绝 |
| rescale | 除以实际移除的最后一个素数的double值；level减1，不把SEAL没有的decode检查伪称rescale API约束 |
| modswitch | scale不变；检查目标层容量；禁止无效drop和链耗尽 |
| rotate/negate | 保持level/scale，沿用已有旋转密钥限制 |
| add | 层必须相同；先检查尺度相容性，再按实际HEVM同时更新lhs及dst，即使dst与lhs不同 |
| 输出decode | 核验最终寄存器、实际scale容量、声明level和声明scale |

保留全部格式、大小、寄存器、常量、密钥、透明密文、bootstrap/upscale和未知opcode拒绝规则。
所有EncodeP按真实preprocess语义预处理；重复plain目标仍拒绝。
本门禁不静态证明未知输入值的编码容量、噪声预算或结果精度，必须继续真实执行和数值比较。

## 尺度匹配依据

逐条同时保存实际double scale与名义整数log2 scale。
名义值按固定编译器的素数位数约定传播：乘法相加、rescale减实际素数bit_length。
输出整数声明必须与名义值完全相等，不能用浮点容差接受差一位的声明。
实际double则严格按与固定SEAL相同的乘除顺序传播；执行观察值与预测值精确比较，
取代旧执行观察中固定1e-6的log2容差。真实rescale的微小偏差作为证据保留。

加法只允许相同名义尺度。偏差来源必须是从已验证输入/常量出发的真实素数rescale传播；
记录每次lhs scale覆盖带来的相对数值解释变化。另用已有冻结相对数值预算1e-4作为
元数据覆盖的拒绝上界，这是项目数值完整性策略，不是SEAL API本身的允许误差。
这项上界不代替最终 abs(error) <= 1e-5 + 1e-4*abs(reference)，也不放宽其门限。
明显不同名义尺度直接拒绝；小差异不被悄悄抹去。

## 验收与边界

新增测试包括：精确乘积位数、锁定参数篡改、超过180的合法scale、编码余量、
中间乘法越界、错误输出声明、链耗尽、modswitch容量、加法不一致、原地/覆盖、
实际参数及输出观察记录篡改。

独立SEAL探针使用原N=32768/14×60/tc128参数，14个正反例与预期一致：
scale=2^400的密态平方/解密最大绝对误差约5.20e-18；
level1/scale=2^75仍拒绝；
直接level1编码2^59拒绝，而HEVM式最高层编码后modswitch到level1合法。
探针亦实证把2^40与2^41的加法操作数强行改成同scale会把目标0.25改成约0.1875。

新绑定下完成Linear self-test、既有mlp4x4x2、SiLU0112/0113/0117和偏置修复回归。
这里包含人工候选和旧Agent源码重放，不计新Agent生成。
后续真实Agent批次单独记录，旧失败和历史绑定不改写。

已有安装若尚无检查模块，使用现有锁定Nix环境构建独立目标 seal_artifact_parameters；
不能回退到无参数绑定执行。CMake仍要求SEAL 4.0.0 EXACT，x86/ARM分别编译自己的模块。
不需新依赖、下载或完整SDK重建。

源码依据：
- 固定 third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp 的 encode_internal/addcc/addcp/run。
- SEAL v4.0.0 native/src/seal/ckks.h 的 encode_internal/decode_internal。
- SEAL v4.0.0 native/src/seal/evaluator.cpp 的 is_scale_within_bounds、multiply、mod_switch_scale_to_next、mod_switch_drop_to_next。
