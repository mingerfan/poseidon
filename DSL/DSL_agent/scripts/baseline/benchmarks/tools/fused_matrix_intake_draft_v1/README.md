# PyTorch 融合矩阵调用接入草案

## 范围

| 写法 | 规范化计算 | 矩阵形状 |
|---|---|---|
| addmm | beta × input + alpha × (mat1 @ mat2) | 两个二维矩阵 |
| baddbmm | beta × input + alpha × bmm(batch1, batch2) | 批量大小相同的两个三维张量 |
| addbmm | beta × input + alpha × sum(bmm(batch1, batch2), axis=0) | 三维张量，输出二维矩阵 |

支持函数调用、Tensor 方法和关键字参数。alpha、beta 为有限公开数值，绝对值上限 1024；
input 按结果形状广播。矩阵乘积至少含一个加密操作数。
beta=0 保留忽略 input 数值的语义；图中原有的更新顺序和视图别名继续由有序捕获处理。

## 实现

targets/pytorch_fused_matrix_normalization.py 将融合调用转换为现有矩阵、乘法、
加法和求和操作，保留来源、shape、公开状态及规范化绑定。
targets/pytorch_static_trace.py 增加显式的元数据规则。
targets/pytorch_ordered_capture.py 在 common-call 和有序别名捕获前接入规范化。

原图、规范化图、分解图及 CPU float64 原函数分别参与差分检查。
该目录属于开发草案，应用接入状态见 acceptance.json。

## 开发验证

在 ARM Ubuntu checkout 执行：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/fused_matrix_intake_draft_v1/run_isolated.py agent-fused-matrix-source-r1268
~~~

结果名须为新的数字编号。运行器使用已有固定 Python/Torch 环境和独立 bubblewrap，
设置网络隔离、内存、CPU 和墙钟限制，核验生产及前置草案的源码绑定。

r1266：26 passed，0 failed/errors/skipped；41 个检查上下文、656 组明文输入、
16,096 个输出值，最大差异 4.44e-16。PyTorch 2.0.1、NumPy 1.25.2、Python 3.10.14。
覆盖公开/密文矩阵、公开偏置、函数/方法、shape 查询、图组合、视图更新、
零系数、权重及元数据篡改、参数和 shape 拒绝、旧默认行为。

## 后续接入

先完成有序 common-call 的已冻结 Poseidon 回归，再联合接入新的显式输入契约。
新增融合矩阵分区随后进行 Poseidon 编译、密态及真实 Agent 验收。
