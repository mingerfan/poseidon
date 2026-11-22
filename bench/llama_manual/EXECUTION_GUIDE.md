# Cerium 源顺序组件与完整模型入口

## 当前目标与状态

按用户2026-11-13要求，首先得到真实GPU执行的有限输出及实测耗时。精度报告保留，**不以1e-3作为退出条件或后续测试的前置门槛**。非有限输出、程序异常和资源错误仍需修复，不能把这些当成一次成功执行。

`source_benchmark.cpp`支持1..32个Decoder Block，固定Llama 3.1-8B真实尺寸，128 token容量；目前样本38有效token。边界是加密embedding到最后一个Block hidden，没有最终Norm、LMhead和tokenizer。真实权重从safetensors按层读入，HF Q/K权重转为Meta交错布局；不复制或展开整套权重到新磁盘文件。

2026-11-13最终结果：source15于16:20完成组件预检及真实权重连续32层，退出0、输出有限；计算7558.47秒，模型进程约8022.75秒。精度未通过，最大绝对误差约955万。C++与source12/source13逐文件SHA相同，后端库和fixture相同。模型与worker均已退出。Slurm5432427/5432428最后确认均在未运行时hold，不应再放行重复测试。结果与差异原因见`TEST_RESULTS_AND_CERIUM_GAPS.md`（开发目录中位于上一级；交付包内同级）。

## 文件职责

| 文件 | 内容 |
|---|---|
| `source_ops.hpp` | ResNet20后端API适配、实际scale/Q契约、显式源位置自举；不持有秘密钥匙 |
| `source_runtime.hpp` | 测试侧密钥、自举profile和GPU上下文准备；最终输出解密 |
| `source_nonlinear.hpp` | 原Sign与SwiGLU分支和运算顺序 |
| `source_rms.hpp` / `source_softmax.hpp` | 原非线性算法、源位置自举及必要物理参数适配 |
| `source_attention.hpp` | 原QK与AV布局和共享旋转 |
| `source_linear.hpp` | 原权重对角及融合RoPE掩码；现有DFT API的流式适配，存在已记录调度差异 |
| `source_model.hpp` | 全部子层与残差组合成一个Block |
| `source_benchmark.cpp` | 真实权重、输入、1..32 Block循环、最终输出与报告 |
| `source_chain_probe.cpp` | 各组件接口/预算执行检查；不是完整Block，不替代模型测试 |

## 模数预算计划

这里的Q是**当前剩余模数个数**。下表列出代码执行契约；组件预检验证了非线性及Attention接口Q，完整32层日志验证了5376条线性输出记录、9856次自举Q6→Q30及32个Block末尾第一个输出密文Q6。不能把这些日志扩大解释为逐个记录了每条中间密文。

| 阶段 | 输入Q | 输出Q | 接口scale |
|---|---:|---:|---:|
| 初始embedding / 上层残差 | 6 | 6 | 2^40 |
| Attention RMSNorm | 6 | 16 | 2^40 |
| Q/K融合RoPE、V投影 | 16 | 14 | 2^40 |
| QK | 14 | 9 | 2^40 |
| 原QK后ModDrop | 9 | 7 | 2^40 |
| Softmax（输出2倍概率） | 7 | 11 | 2^40 |
| AV（保留原补偿掩码） | 概率11、V14 | 8 | 2^40 |
| O投影及残差 | 8 / 残差6 | 6 | 2^40 |
| FFN RMSNorm | 6 | 16 | 2^40 |
| Gate、Up投影 | 16 | 14 | 2^40 |
| SwiGLU | 14、14 | 10 | 2^40 |
| Down投影 | 10 | 8 | 2^40 |
| FFN残差 / 下一Block输入 | 8 / 残差6 | 6 | 2^40 |

每个RMSNorm有22次native Boot（输入16次、指数种子1次、Newton5次）；16个Softmax lane各13次；56个SwiGLU lane各1次。source15完整日志已确认每Block 308次、32Block共9856次；这是**当前单lane适配的实测调用数量**，不能作为Cerium原始配对版本的次数。源码位置日志`SOURCE_BOOT`与程序总计一致，implicit_bootstrap_calls=0。

## 运行与输出

排队集群工作目录：

`/work/home/gaoyang123/llama-api-migration-20261112/cerium-execution-v2`

编译复用已校验哈希的ResNet20后端库，只编译新应用。集群`source11/`保存已完成的单Block证据；`source13/`、`source14/`最后确认在执行前hold。完成的单卡目录是`/root/private_data/llama-api-20261111/cerium-execution-v2/source15`，A80080GB、7核容器/7线程、245GiB主存配额；组件与完整模型串行。后台每60秒资源采样，用户要求的人工巡检于15:21改为每15分钟。源码包和每轮产物冻结保存，现有结果不覆盖。

命令参数顺序：

长测Slurm提交需要显式设置足够的`--time-min`并回读实际时限，避免平台默认12分钟令backfill缩短分配时间。source13预设15分钟上限/最短；source14预设4小时30分上限/4小时最短，程序计算超时4小时，两者未执行。单卡source15最终32层计算实测2小时5分58秒；不同机器/版本不能直接作单项优化的加速比较。原5432410在运行前取消，替代的5432427/5432428最后确认hold；不得重复提交或恢复。

```text
source_benchmark FIXTURE WEIGHTS INDEX EMBEDDINGS REFERENCE OUTPUT BLOCKS VALID_TOKENS REPORT_JSON
```

- `BLOCKS=1`必须用`reference-block01.bin`，`BLOCKS=32`必须用`reference-block32.bin`；文件头会验证层数。
- `OUTPUT`保存完整128×4096个最终解密hidden数值，原GPU计算输出本身是密文；解密属于验收驱动器。
- `REPORT_JSON`分别记录执行状态、精度、计算/整体耗时、自举次数、RMM活跃显存峰值和池保留量。误差只统计有效token，有限性检查覆盖全部槽。
- 不允许覆盖已有输出。错误保留日志，不能写成成功结果。
- 原模型参考数据已存在且经过校验；测试期间不从中间解密值修正模型计算。

当前对照文件来自已有 `original_callpath_v2`：真实权重的Cerium原调用路径、float64 dense、原packed Softmax与未修改的approx_silu/sign_approx/sgn22，含固定种子的complex128 jitter。它不是标准Hugging Face精确算子的输出，也不是当前C++密文DAG的逐指令浮点模拟。因此报告误差包含参照路径差异与CKKS数值误差，不能仅据最终误差归因于自举或说成纯加密误差。

## 与原Cerium仍不同的地方

单GPU native自举替代Cerium自举，物理模数链不同；双lane配对尚未实现。本版选择的DFT封装每输入tile完成Rescale，原源码跨输入tile先累加；底层已有不立即Rescale的入口，相关适配与旋转复用仍未完成。采用流式权重准备，没有移植原全图缓存重绑定或CUDA Graph。详见`SOURCE_MAPPING.md`及`TEST_RESULTS_AND_CERIUM_GAPS.md`。不能把未完成的移植一概说成API或硬件不支持，也不能因为32层执行完成就声称全部照抄或精度相同。
