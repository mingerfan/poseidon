# Attention / RMSNorm DSL 源码交付

交付给已有 Hecate/Dacapo 和 GPU 运行环境的工程。目录结构相对于对方的 Dacapo 仓库。
本包只提供 DSL 源码、公开近似配置和接口说明，GPU 构建和运行由对方工程负责。

## Git 中需要的文件

- `examples/benchmarks/QwenAttention.py`：完整 Attention 的 `@hc.func` 跟踪入口。
- `examples/benchmarks/QwenRMSNorm.py`：单 token RMSNorm 的 `@hc.func` 跟踪入口。
- `examples/benchmarks/qwen_layer_profiles.json`：显式近似配置。
- `python/hecate/hecate/qwen_layers.py`、`qwen_nonlinear.py`、`ops.py`：共享层和算子实现。
- 本说明。`manifest.json` 是交付校验信息，可按对方工程习惯保留。

将文件放入现有 Dacapo 工程对应目录；对方已有相同共享模块时无需重复新增。
复用其现有 `hecate.expr`、包初始化、编译器和运行库。
这里的共享模块来自已验证版本；若对方同名模块有其他修改，应合并差异。

## Attention

输入权重 NPZ 的键及矩阵形状：

| 键 | 形状 |
| --- | --- |
| q_weight | [query_heads * head_dim, hidden] |
| k_weight、v_weight | [kv_heads * head_dim, hidden] |
| o_weight | [hidden, query_heads * head_dim] |
| q_bias（可选） | [query_heads * head_dim] |
| k_bias、v_bias（可选） | [kv_heads * head_dim] |

在 Dacapo 仓库根目录执行，例如：

```bash
python examples/benchmarks/QwenAttention.py --weights attention.npz --slots 32768 --positions 0 1 2 3 4 5 6 7 --output-dir traced/attention_prefill
python examples/benchmarks/QwenAttention.py --weights attention.npz --slots 32768 --mode decode --positions 8 --cached-positions 0 1 2 3 4 5 6 7 --output-dir traced/attention_decode
```

默认 Q 头=14、KV 头=2、head_dim=64、RoPE theta=1000000，可由命令行显式修改。
序列长度由 `--positions` 的数量确定；decode 要求一个新 token 和非空缓存。
这是静态计算图：换序列长度、缓存长度或公开位置，需要重新跟踪对应图。

设新 token 数为 N，缓存长度为 C：

- 密文输入顺序：N 个 hidden、C 个 cached K、C 个 cached V。
- hidden 每个密文有效前缀长度为 hidden；K/V 有效长度为 kv_heads * head_dim。
- 密文输出顺序：N 个 Attention 输出、C+N 个 K、C+N 个 V。
- K 已包含 RoPE；decode 直接复用这些 K/V，不重复旋转缓存。
- 计算包括 Q/K/V、RoPE、缩放点积、因果 Softmax、加权求和和 O 投影。
- RMSNorm 和残差由外层 block 组合。

## RMSNorm

权重 NPZ 只需 `weight`，形状 `[hidden]`。输入和输出各一个密文向量，有效前缀长度为 hidden。

```bash
python examples/benchmarks/QwenRMSNorm.py --weights rmsnorm.npz --slots 32768 --eps 1e-6 --output-dir traced/rmsnorm
```

## 配置及交付边界

两个入口都支持 `--profiles path.json`；默认使用同目录的 `qwen_layer_profiles.json`。
本包默认 Softmax 配置要求中心化分数在 [-2,2]、每行长度不超过 32；
RMSNorm 要求 mean(x*x)+eps 在 [3.7225003325147554e-5,0.0016316197579726577]。
这些是已有单层测试的有界配置，需要调用方按实际数据选择或校准，不是任意输入的默认保证。
真实尺寸完整 block 试验使用了另一组配置，不能混用其精度结论。

权重作为公开常量参与跟踪；输入、输出和 KV 是密文表达式。
`hc.save` 导出 MLIR 及常量，由对方现有编译/运行链处理模数链、自举、密钥和 GPU 调度。
本次只验证入口参数接线、prefill/decode 的 KV 传递及 RMSNorm 的 NumPy 数值结果；
没有在本包中运行原生前端或 GPU，也未附带权重、密钥、运行库、日志和大型诊断图。
