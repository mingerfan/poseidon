# PyTorch 矩阵入口验收 r815

可信 PyTorch v7 已接入向量、矩阵及多维批次广播。规范化复用已有 Linear／广播／乘法／归约，Poseidon、Runtime、Dacapo、参数及 reference 保持。

## 验收范围

| 分类 | 案例 | 结果 |
|---|---:|---|
| 密文向量点积 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 密文矩阵乘向量 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 密文向量乘矩阵 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 多维密文批次广播 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 公开向量左乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 公开向量右乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 公开矩阵左乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 公开矩阵右乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 批量公开矩阵左乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |
| 批量公开矩阵右乘 | 3 | 原模型／双参考／Poseidon CPU 通过 |

共30例，每例16组隔离原Module、纯forward及独立参考检查。真实Hecate、Earth／CKKS IR、RuntimePlan、OperatorSpec、plaintext bundle、Poseidon CPU、解密和逐项比较通过。

数值合计150组输入、1388输出值，最大绝对误差6.7626560685e-07。门限保持 `1e-5 + 1e-4*abs(reference)`。三份包重载、三组新输入各执行两次及缓存复用通过，覆盖标量、结构化输出和公开批量权重。

来源为规则候选；当前新真实Agent生成0。原float32应用精度、形式化全输入等价、bootstrap及CUDA未在本批认证。

## 工程与边界

424项：421 passed、0 failed/errors、3旧SEAL条件证据skipped。旧36项捕获逐字一致。60项矩阵来源哈希缺失／漂移拒绝；770包文件、710编译产物哈希核验通过。CLI准备成功，未生成DSL；开发生成器79份文件逐字复现，已有目录拒绝并保持。

v7沿用工厂、NPY状态、索引、近似及静态输出接口；matrix_capture绑定形状、广播计划、派生权重和图节点。旧v1–v6数学捕获及默认请求继续按原规则处理，旧报告／旧源码绑定保留。JSON高层matmul的原rank范围保持，v7在PyTorch接入层规范化后提交现有图。

general-tensors-v1仍为rank 1–32、输入／输出各4096、单中间65536元素。产品物化张量参与预算；8192元素的Linear中间结果再归约例只完成参考测试，未新增密态通过。逻辑规模检查、输入密文内存下界与实际峰值分开；中间密文／密钥／峰值估计继续未完成。

## 身份与资源

绑定 `871473681e20af11d81a91c33e78c47b0b6c5b586f755aa15d981776ae7f81d0`；670 Agent／配置源码、78来源文件、648后端文件及6 reference／门限文件保持。生成源码闭包598，编译／执行／验证594；仍共享语义模块。原1394修改／未跟踪路径保留，分支和HEAD保持；diff check通过。

记录worker约749.62秒，新增约139.96MiB，全局22.93GiB，余量45.81GiB。编译≤2／链接≤1／密态串行。候选沙箱网络隔离、私有路径隐藏、环境清理和原资源限制通过；执行器仅用评估密钥。

API／HTTP、安装、下载、SDK重建、后端／参数／reference修改和Git发布0。首轮公共探针多余placeholder、新夹具维度及索引计数问题已保留原诊断；修正后分别建立新检查，没有改写历史结果。

## 下一步

新增v7功能的自由生成评测已准备r816具体请求和预算，待单独批准。随后继续常见算子写法／未见组合及编译后资源估计。原Attention深度、bootstrap、密文交接、CUDA、x86原生／空白安装和ResNet20分别保留状态。

[输入说明](operator-decomposition-and-python.md#pytorch-矩阵运算) · [机器报告](agent-pytorch-matrix-acceptance-r815.json)

预检／审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-matmul-preflight-r806`；执行／程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-matmul-backend-r812`。
