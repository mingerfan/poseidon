# 4096元素 PyTorch 输入与 ResNet20 准备

ARM Ubuntu，lhy-agent-dsl，2026-10-02。可信本地 PyTorch 的4096元素入口完成代表性 Poseidon CPU 数值验收。

## PyTorch 验收

| 模型 | 输入元素 | 输出元素 | 输入／输出密文块 | 结果 |
|---|---:|---:|---:|---|
| 公开缩放、偏置、平方 | 4096 | 4096 | 16／16 | 通过 |
| CIFAR形状 [1,3,32,32]，sum/mean组成向量 | 3072 | 2 | 12／1 | 通过 |
| 四种rank输入、均值交叉融合、四命名输出 | 4096 | 4096 | 16／16 | 通过 |
| rank-4归约与标量stack | 24 | 3 | 3／1 | 通过 |
| 双输入标量融合与stack | 9 | 3 | 3／1 | 通过 |

五个源程序各完成16组原模块、高层图及两套独立分解参考检查。全部经真实tracing、Dacapo-modified编译、RuntimePlan／OperatorSpec／bundle检查、Poseidon CPU密态执行、解密及逐项比较；20组输入、32800值，max 1.75709971018e-6。五个包重载同样20组／32800值，max 7.21292815342e-6，重载生成和编译0。冻结数值门限保持1e-5+1e-4*abs(reference)。

全部是规则候选，新增Agent生成与API调用0。部分数学计算与此前JSON容量案例相同；这是Python入口与构造验收，不增加冻结Benchmark的去重模型数。

静态323项：300 passed、0 failed/errors、23旧探针／历史证据 skipped。开发清单另8项通过，验证旧medium清单、配置选择、哈希和分片限制。最后602执行源码绑定一致，五包完整性审计通过，底层编译器、库、runner和参数身份未变。

公开CLI也完成4096元素四输入模型的隔离导入和请求准备，数学图与已验收案例一致。CLI复核未再次编译或执行。

## 修正

PyTorch零维sum/mean的图内表示为[1]。stack后仅移除这个额外表示维度，使输出与PyTorch一致。辅助节点名避开真实FX节点；直接零维返回仍要求reshape(1)，错误算子先报告原有诊断。

修改仅涉及捕获器、相关测试、开发验收清单和文档。旧V1清单保持medium-v1／P256；V2选择已配置的medium-v1或resnet20-v1与合法周期。逻辑、物理、源码、工作、AST和沙箱门禁继续生效。

r522捕获失败、r524诊断次序回归、r529报告收集错误及中间r528结果保留，各有原绑定。三个原中型规则候选与r487逐字节一致；空间残差旧内存失败未重跑或记通过。

## ResNet20 源模型

已核对本地19层卷积、9残差块、16／32／64通道和10类输出。97份权重／BN文件共271098个公开值，包含BN运行统计量；卷积为OIHW，Linear为输出×输入。源文件和现有数据全部绑定SHA-256。

标准ReLU模型以CPU float64 PyTorch和独立NumPy卷积／BN／Option-A残差实现对照。40张固定分层图片（每类4张）和4组观察范围诊断探针共440 logits，max差异1.95399e-14，在1e-12+1e-12*abs(reference)内。现有1000张子集914正确，Top-1 91.4%。

该子集不是完整CIFAR-10测试集，没有训练校准集，预处理来源尚待补齐。这里没有近似拟合、Agent生成、编译或密态ResNet执行。4组诊断探针也不替代最终有效输入域验收。

完整尺寸Conv+BN stem的[1,16,32,32]中间张量需要64个P256块。数学形状检查通过，但公开路由估算442368，超过当前262144工作预算，在请求准备返回Capacity spatial map work budget。诊断输出取通道均值；这不缩小stem，也不算完整网络通过。下一步改进大特征图打包和共享卷积路由，保持原模型和预算。

## 使用

~~~sh
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-extended-v1/fx_extended_multi_io.json \
  --trust-local-code --resource-profile resnet20-v1 --write \
  --output "$results/imported-4096"
python3 -B scripts/agent.py --backend local candidate -- \
  --case "$results/imported-4096/model.json" --self-test \
  --unified-chunk-period 256 --unified-guidance explicit-v10 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

self-test使用规则候选。开发分片工具支持只读计划：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/qualify_poseidon_pytorch.py \
  --manifest scripts/baseline/cases/trusted-pytorch-extended-v1/acceptance.json \
  --output "$results/pytorch-next"
~~~

## 剩余工作

容量／Python新增能力的真实Agent评测、Attention修复、空间残差中间密文释放、12深多项式、重复bootstrap精度和完整ResNet20仍待完成。x86原生、空白机器安装、CUDA暂缓。当前Attention提案r536只更新源码绑定，仍待批准，未调用API。

本批安装、下载、参数变更、编译器／Runtime核心修改、commit和push均0。

[机器报告](poseidon-pytorch-extended-acceptance-r535.json)。原始证据保存在ARM work root的poseidon-pytorch-extended-static-r531、poseidon-pytorch-extended-r532、poseidon-resnet20-source-r533和poseidon-pytorch-extended-audit-r534。
