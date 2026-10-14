# PyTorch v22 公开张量算术接入与 CPU 验收

已接入应用的 `submit_source`、来源校验、程序包导出与本地执行路径。显式选择 `poseidon-trusted-pytorch-v22`；来源 CLI 默认 v18，`latest` 保持 v21。

## 新增输入

- 公开标量、向量和张量除法，除数由原始 NPY 权重绑定。分解为可信的固定倒数乘法。
- 公开布尔 mask 与浮点张量相乘，按原 PyTorch 语义转换为 0/1，保留原始 bool 状态绑定。
- 与静态索引、残差、命名多输出、BatchNorm、显式 GELU 近似和 Linear 组合。
- float32 权重显式选择 `convert-float64-v1`；原始权重和 float64 计算副本分别绑定。

静态广播按 shape 检查，原地计算保持原 shape。除数和倒数均须有限、非零，绝对值各不超过 1024；本版 tensor 除数是直接声明的公开状态。旧 v21 规则保留。

## 本次结果

157 项控制和兼容检查通过，failed/errors/skipped 均为 0。包括来源/权重篡改、类型、广播、原地写入、旧版约束，以及依赖身份缓存失效检查。

| 模型 | 编译、密态、包复用 | 比较值数 | 最大绝对误差 |
|---|---|---:|---:|
| 索引＋布尔掩码 | 通过 | 100 | 8.2901e-09 |
| float32 权重＋BatchNorm | 通过 | 60 | 5.2050e-09 |
| 按行掩码＋残差 | 通过 | 120 | 1.6428e-08 |
| 公开除法＋GELU＋Linear | 通过 | 40 | 5.2000e-09 |

本次 4 个模型全部从源码入口重新捕获、编译并执行，完成 Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle 检查与 Poseidon CPU 数值验证。每例 4 组首次验收输入，重载后另执行 3 组新输入，再重复这 3 组；合计 12 次密态执行调用、40 组输入、320 个输出值，4 个程序包均可复用。

候选来自规则基线。真实 Agent 新生成尚未运行。GELU 比较目标为清单中冻结的近似计算；原函数近似误差和原生 float32 应用精度另计。

门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

此前首例 `bool_multiply` 已按旧生产绑定完成数值和包复用，本次按授权保留该记录。旧的 SDK 预检失败、原窗口超时和中断记录均保留；没有把旧记录重标为当前绑定。

## 身份和资源

开发 checkout：`/home/lhohy/Code/Poseidon`；分支：`lhy-agent-dsl`；HEAD：`7588e48093631eaf8385506e3db058e7ca3cfe7e`。

- 当前 SDK：`1066e8ae9e5ffed034a3c8341ac25aa4d1603f1070d2cec67235475a6ec8a7b6`。
- 续验计划：`6e79a979ad15abda780e0583e3009ad679c05d1fa3b2a8bd726eba061ff56f5e`。
- 续验结果：`590641bd6cb1d229422e6ddc4517a6afe605404f6812f386c25f86efe5cafb47`。
- 完整报告：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-v22-cpu-continuation-v1/report.json`。
- 用时：191.35 秒 / 480 秒。
- 新产物：51.76 MiB / 256 MiB；总产物：28.21 GiB / 32 GiB；剩余磁盘：41.59 GiB。

资源计数沿用执行边界：取 apparent size 和计入全部硬链接的已分配空间之较大值。

实际沿用 LLVM/MLIR 18.1.2、GCC 13.2.0、Python 3.10.14、Torch 2.0.1、NumPy 1.25.2 和现有 Poseidon CPU 工具链。参数、构建、库与来源哈希见同名 JSON 和每个程序包的 `artifacts/dependencies.json`。x86 配置保留，本次真实执行平台为 ARM Ubuntu。

本次仅修改 Agent 接入、校验、相关测试和文档。Poseidon、Runtime、Dacapo、模型、reference 和门限保持；API/HTTP、安装、下载、网络修改、commit/push 均为 0。

依赖扫描只缓存相同源码字节的语法分析，每次仍读取当前文件并重新计算身份哈希。5 次身份计算的记录为 [0.30227321699931053, 0.005708216000130051, 0.005653381000229274, 0.0057014230005734134, 0.005600046999461483] 秒；源码变化、依赖出现/消失和锁文件变化均有失效测试。

## 后续

1. 为 v22 新入口冻结真实 Agent 生成与修复测试，再按具体范围申请付费授权。
2. 继续输入写法、容量、通用算子和组合回归；本次四例代表性通过范围见表。
3. Bootstrap、密文子图交接、CUDA 实机与整网目标继续单独验收。

下一条只读命令：`python3 -B scripts/prepare_pytorch_source.py --help`。
