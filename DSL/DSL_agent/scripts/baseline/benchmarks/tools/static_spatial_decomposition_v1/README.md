# 静态空间算子分解草稿

Agent 开发工具。当前完成明文分解检查，应用入口仍沿用现有能力清单。

| 运算 | 范围 |
|---|---|
| resize | 一维 nearest、nearest-exact、linear；二维 nearest、nearest-exact、bilinear |
| ConvTranspose | 一维／二维、公开权重、分组、stride、padding、dilation、output_padding、静态 output_size |
| 组合 | 平方、归约和多输出 |

resize 按固定 Torch 2.0.1 CPU Float64 连续张量语义处理。nearest 的通用和向量化路径分别核验；来源见 [对应版本源码](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/UpSampleKernel.cpp) 和 `source-basis.json`。

分解复用 Agent 的核心图：resize 使用各轴公开 Linear；转置卷积使用公开插零／裁剪 Linear 及分组普通 Conv。矩阵大小、公开常量总量和中间张量在分配前检查。

42 个新增语义上下文、每例 16 组探针通过四路比较：直接 NumPy 数学、原始 Torch、分解图 NumPy 和分解图 Torch。共 672 组／13120 个逻辑输出值，最大差异 `4.440892098500626e-16`。8 项单元检查通过；分解后 30 种结构签名、15 个拓扑组。

本目录保留新增模型和分解绑定，原 1,200 模型集保持。应用接入、真实编译、密态执行和真实 Agent 生成均为待验收。

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
PYTHONPATH=scripts/baseline python3 -B scripts/baseline/benchmarks/tools/static_spatial_decomposition_v1/run_math.py
~~~

`run_math.py` 使用已有固定 Nix 环境，首次写入冻结结果；已存在结果时采用独立目录开展后续版本验收。
