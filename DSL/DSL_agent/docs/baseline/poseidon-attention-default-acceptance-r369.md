# Attention 默认生成：Poseidon CPU 验收

MLP、RMSNorm、Attention decode 的默认 `--self-test` 路径均已通过。Attention 保留原来的投影、RoPE、近似 Softmax、显式 KV 输入和三项输出。模型、公开权重、request、reference、模数链与数值门限未修改。

| 检查 | 结果 |
| --- | --- |
| 默认整模型：MLP / RMSNorm / Attention | 3/3 |
| LayerNorm / GroupNorm / InstanceNorm / 批量 matmul | 各 3/3 |
| 默认入口完整批次 | 15 passed，0 failed，0 skipped，0 not_run |
| RMSNorm / Attention / LayerNorm 程序包重载 | 3/3 |
| 相关静态、参考与组件测试（去重） | 120 passed，0 failed，3 历史 skipped |
| 真实 Agent 生成 / 付费调用 | 0 / 0 |

默认批次执行 60 组密态输入，比较 400 个值，最大绝对误差 **1.0194734547663842e-6**。其中 Attention 比较 40 个值，最大误差 **3.161456541000973e-9**。三个程序包重载另比较 80 个值，最大误差 **7.203329043664608e-8**；生成、编译调用均为 0。完整批次耗时 156.55 秒，产物约 58.46 MiB。

所有密态结果经过真实 frontend tracing、编译、RuntimePlan / OperatorSpec / bundle 检查、Poseidon CPU 执行、解密和逐项数值比较。使用冻结条件：

```text
abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)
```

## 修改

- 分组计算公开 power-basis 多项式，保留原系数。
- 根据已绑定输入布局，将完整平方和／旋转相关式点积转换为 packed 计算；初始范围为周期不超过 16、总输入不超过 16 个元素。
- 在加减分支中提取完整共享密文乘积，保持共同逆函数乘积的分组。
- 公开系数由可信代码计算，以当前契约允许的有限标量字面量表达。候选中没有 Plain/Plain 算术；非零系数下溢时放弃改写。
- 保留工作量、图深度、源码、AST 和密文操作限制；超出改写范围沿用原生成方式。

涉及 `unified_graph_lowering.py`、新增 `unified_graph_quadratic.py` 和相关测试。改写只处理规则基线，Provider 和重放候选保留原程序。

判断依据为图依赖、类型、公开参数和输入布局。没有模型名分派，没有人工 Attention 源码替换。不同浮点求值次序经过独立数学解释器和 CPU float64 Torch 比较；这属于有限数值证据。

## 修复依据

r355 的默认 Attention 在 Earth 乘法容量检查失败。r356 的点积改写通过明文参考，但分散的多项式项与公共乘积仍导致 CKKS modswitch 编译失败。分组多项式与完整因子提取后，r359 首次通过真实密态验收；r360 的原三模型和三个尺寸的平方和／点积共 9/9 通过。

r357/r358 发现公开因子提取可能产生 Plain/Plain 运算；现已在类型检查中保留原分支。原失败源码、日志和绑定全部保留，未计入 Agent 失败或成功。

最终默认入口为 r363–r366，组件兼容为 r367。25 项相关测试与 115 项兼容测试重叠 17 项，去重为 123 项：120 通过、3 项历史证据测试跳过。首次测试启动缺少动态库环境变量的失败日志也保留；没有新增依赖。

## 范围与剩余工作

这是当前小型、显式近似模型的规则生成验收。归一化与 Softmax 相对原数学函数的近似误差单列；九个归一化模型的原函数误差边界仍为 0.632650。完整网络精度、大张量布局和 ResNet20 尚未验收。

原 Benchmark 的 12 项深层多项式编译容量失败不变。真实 bootstrap 有符号输入的精度失败仍保留；r235 独立配置试验尚待批准。CUDA、x86 原生执行与空白机器安装继续暂缓。

Dacapo、CKKS Runtime、Poseidon 库、参数及冻结 1,200 模型集未修改。应用核心 518 文件、完整执行源码 575 文件分别绑定；旧 516 文件保存在 r361 的源码快照中。新的 r368 三模型付费提案替代未执行的 r354，公开请求及后端身份一致，仍待单独批准。

[机器报告](poseidon-attention-default-acceptance-r369.json) · [三模型提案](poseidon-whole-models-live-proposal-r368.json)

## 使用

在 ARM Ubuntu checkout 运行默认 Attention 自检：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/poseidon-whole-models-v1/attention_decode.json \
  --self-test --max-repairs 0 --compiler-configuration poseidon-cpu-eva-w45-v1
```

已验收程序包：

```text
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-scheduled-packages-r365/attention_decode/program
```

直接重载：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay \
  --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-scheduled-packages-r365/attention_decode/program \
  --execute
```

开发位于 `/home/lhohy/Code/Poseidon`、`lhy-agent-dsl`，HEAD 保持 `7588e48093631eaf8385506e3db058e7ca3cfe7e`。本批安装、下载、付费、commit、push 均为 0。
