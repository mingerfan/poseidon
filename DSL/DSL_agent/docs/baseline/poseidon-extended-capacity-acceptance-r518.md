# 4096元素物理布局验收

2026-10-02，ARM Ubuntu，lhy-agent-dsl。新增容量ABI已完成代表性Poseidon CPU数值验收。

## 配置

| 配置 | 逻辑输入／输出，各自 | 物理输入／输出块，各自 | DSL入口 |
|---|---:|---:|---|
| medium-v1 / capacity V2 | 1024 | 4 | x,y,z,t，末尾zero_ct |
| resnet20-v1 / capacity V3 | 4096 | 16 | x0,x1,...，末尾zero_ct |

周期仍从4、8、16、32、64、128、256选择；4096元素使用256。每个张量分别分块和补零，补零占用物理块。因此满足逻辑元素上限的组合仍须通过物理块检查。逻辑输入和命名输出各最多4个，rank最多4。

V3仅由resnet20-v1启用；medium和旧V2保持四块限制。校验器从完整布局核对参数顺序、zero_ct、输出数量和周期。golden最多17参数；native helper仍最多16参数，对象数组仍最多16单元。源码、AST、工作、深度、安全和沙箱限制保持。

归约基线复用完整周期求和native helper，每次调用仍计入展开工作，primitive运算保留。候选源码没有自动改写路径。生成规则中“四块”与V3冲突的文本已修正；V2请求保持原文本和哈希。

## 验收

| 范围 | 数量 | 结果 |
|---|---:|---|
| 逐项乘、加、平方；含4096元素及CIFAR形状 [1,3,32,32] | 3 | 3/3 |
| 公开偏置后全sum/mean；含补零尾块 | 3 | 3/3 |
| 四输入、不同rank／长度、均值交叉融合、四命名输出 | 3 | 3/3 |
| 同一4096元素模型的公开构造入口 | 1 | 1/1 |
| 原Linear、Linear→square→Linear规则回归 | 2 | 2/2 |

共12任务、11个不同数学模型，其中9个新增模型；公开构造变体另计。全部为规则候选，新增Agent生成和API调用0。

真实frontend tracing、Dacapo-modified编译、RuntimePlan／OperatorSpec／bundle检查、Poseidon CPU密态执行、解密及逐项比较通过。48组输入、105072个输出，最大绝对误差6.35924800463e-06。10个容量程序包重载40组、105064值，最大误差7.52768067969e-06；重载生成／编译0。门限保持1e-5 + 1e-4*abs(reference)。

静态299项：276 passed、0 failed/errors、23 skipped。跳过项为旧隔离frontend探针及历史helper-composition证据测试，不计通过。新容量16项、旧请求精确重建及非法ABI／布局／padding预算／签名／helper边界均通过。r511两项错误、r512初次通过和r513初次密态记录保留。

最终602执行源码与静态、密态及包审计绑定一致，10包完整性复核通过，底层后端身份与r508一致。三个原中型候选源码与r487逐字节一致；空间残差内存失败没有重跑或标记通过。本批执行及重载137.29秒，产物95.41MiB；低于1800秒／1GiB限制，4GiB地址空间上限保持。未测量逐例峰值RSS。

## 使用

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/agent.py --backend local candidate -- --case scripts/baseline/cases/capacity-extended-v1/capacity_extended_elementwise_0.json --unified-chunk-period 256 --self-test --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

self-test使用规则候选。已编译包可重复执行：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-extended-capacity-r515/capacity_extended_multi_io_2/program --execute
~~~

## 待完成

4096元素IO已有真实执行证据；任意大图／算子组合、65536元素中间张量、大权重包及完整ResNet20仍未验收。容量与Python新入口真实Agent测试需单独预算批准。Attention重测、空间残差Runtime释放方案、12深多项式和重复bootstrap精度问题保持原状态。x86原生、空白安装与CUDA暂缓。

原始记录：
- results/poseidon-extended-capacity-static-r514
- results/poseidon-extended-capacity-r515
- results/poseidon-extended-capacity-audit-r516

[机器报告](poseidon-extended-capacity-acceptance-r518.json)。本批未修改编译器、Runtime核心、库、参数、reference或门限，未安装、下载、付费、commit或push。


## CLI复核 r519

公开启动器的4096元素self-test通过，另4组／16384值，最大误差5.403973935e-09，独立审计通过，源码绑定不变。这是同一模型的入口回归，不增加不同模型数量。完整逐项输入、reference和解密数组保留在客体results；仓库机器报告只列指标、证据路径与哈希。
