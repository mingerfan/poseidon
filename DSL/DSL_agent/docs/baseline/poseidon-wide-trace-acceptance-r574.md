# 大周期布局与 ResNet20 首层诊断验收

原尺寸诊断模型 `[1,3,32,32] → Conv → BatchNorm → 16 通道均值` 已通过 Poseidon CPU 编译、密态比较和程序包重载。输出位于 ReLU 之前；残差块与完整 ResNet20 尚待验收。

## 结果

| 范围 | 结果 |
|---|---|
| 当前静态与兼容测试 | 121 passed，0 failed/errors/skipped |
| 固定 MLIR 常量往返 | 168 个常量、688128 个 f64 位模式一致 |
| 浮点边界解析 | 4/4，覆盖正负零、指数、极值和非规格化数 |
| 显式精简配置 | 4096 元素逐项模型及原首层 2/2 数值通过，2 个包重载通过 |
| 原首层 | 四组输入、64 个输出值，最大误差 3.4448e-9 |
| 大周期原回归 r562 | 5 个规则模型与 2 份历史 Agent 源码通过；原首层大小失败已在 r573 解决 |
| 历史 Agent MLP/RMSNorm r571 | 原源码重新编译、数值和重载 2/2 通过 |

各批次保留各自源码绑定。上述回放不增加真实 Agent 新生成数。没有付费调用、安装、下载、提交或推送。

## 使用

大周期只对 `resnet20-v1` 开放。显式选择 `--unified-chunk-period 512/1024/2048/4096`，使用 V4 输入协议和公开张量 v8；逻辑输入、输出各最多4096元素，最多16个物理输入／输出块。旧 V2/V3 配置继续使用原周期和身份。

大常量 IR 可显式选择：

~~~sh
--compiler-configuration poseidon-cpu-eva-w45-compact-v1
~~~

该配置使用 schema4，将可缩短的 f64 dense 十六进制缓冲区转为精确往返的十进制文本。首层 trace 从11138317降至6812765字节。原始、精简和入口 IR 均保留，审计重新计算转换并比较哈希与浮点位模式。前端文件16 MiB、最终编译入口8 MiB限制保持；private符号标注后的最终文本也检查8 MiB。

旧 w40/w45 配置及 schema2/3 请求保持原身份和默认行为。参数、模型、reference、编译器及Runtime核心均未修改。

## 复现

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/qualify_poseidon_wide.py \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-stem-next \
  --compiler-configuration poseidon-cpu-eva-w45-compact-v1 \
  --case-id resnet20_frozen_stem_pre_relu --execute
~~~

输出目录须为新目录。该命令复用原冻结首层模型，使用规则候选，串行执行；1800秒、产物1 GiB、全局32 GiB和磁盘余量4 GiB检查继续生效。

原始结果：`poseidon-trace-static-r572`、`poseidon-stem-compact-r573`；全部位于 ARM work root 的 results。机器报告见 [r574 JSON](poseidon-wide-trace-acceptance-r574.json)。

下一步处理原空间残差的中间密文内存，再推进残差块、近似校准、bootstrap和真实Agent验收。
