# Qwen2.5-0.5B：24 层完整前向 DSL 源码交付

本包以 `my_qwen2.py` 的模型结构为依据，实际串联 **24 个 Decoder Block + 最终 RMSNorm + 完整词表 LM head**。默认序列长度为 **2 token**，位置为 **0、1**。这次不再是单个 Block。

**交付状态：源码和接口已整理；24 层小尺寸 NumPy 算术验证通过；真实尺寸结构检查通过。未完成真实尺寸 24 层的原生 Hecate 编译、CKKS 精度测试或 GPU 测试。没有附带预训练权重，默认近似参数尚未针对真实 Qwen 权重校准。**

## 1. 完整计算范围

```text
token IDs
 → 客户端 embedding 查表 → 加密每个 token 的 hidden 向量
 → Decoder Block 0 → 1 → … → 23
 → 最后一个新 token 的 RMSNorm
 → LM head → 151936 个原始 logits（分块密文）
同时返回全部 24 层的密文 K/V
```

每个 Block 都包含：

- RMSNorm、Q/K/V 投影及偏置、RoPE、因果 GQA、近似 Softmax、O 投影、残差。
- 第二个 RMSNorm、gate/up 投影、近似 SiLU 与逐元素乘法、down 投影、残差。

结构配置为 24 层、hidden=896、FFN=4864、Q heads=14、KV heads=2、head_dim=64、vocab=151936、RoPE theta=1e6、RMS epsilon=1e-6。

客户端 embedding 查表是明文步骤；本包没有实现对密文 token ID 的查表。服务器前向输出是密文 logits，采样、argmax、词表 Softmax 和 tokenizer/text generation 循环不在该计算图中。与参考 `my_qwen2.py` 一致，仅返回最后一个新 token 的 logits。

## 2. 三种入口与密文接口

默认 slots=32768，logit_chunk_size=1024，149 个 logit 密文块；前 148 块各有 1024 个有效槽，最后一块有 384 个。

| 模式 | 新位置 / 已缓存位置 | 输入密文数 | 输出密文数 |
| --- | --- | ---: | ---: |
| prefill | `[0,1]` / 空 | 2 | 245：149 logits + 96 KV |
| decode-pair | `[0,1]` / 空 | 2 | 245：149 logits + 96 KV |
| prefill，建立首 token 缓存 | `[0]` / 空 | 1 | 197：149 logits + 48 KV |
| decode，追加第二个 token | `[1]` / `[0]` | 49：1 embedding + 48 KV | 245 |

`prefill` 在每层同时处理两个 token。`decode-pair` 在同一静态图内先让首 token 经过全部 24 层，保留各层 K/V，再让第二个 token 经过全部 24 层并复用对应缓存。首 token 的 logits 不作为此模式的输出，因此不构建其无用 LM head。

`decode` 是独立的单步入口，接收已有密文缓存。缓存顺序是：

```text
layer 0: K[position 0], K[position 1], …, V[position 0], V[position 1], …
layer 1: K[…], V[…]
…
layer 23: K[…], V[…]
```

- 输入先放新 token 的 embedding，再放上述缓存；每个 embedding 有效前缀 896，每个 K/V 有效前缀 128。
- 输出先放 149 个 logits 块，再放包含新 token 的全部缓存。
- 连接首 token prefill 与第二 token decode 时：decode 输入为 `[新 embedding 密文] + prefill 输出[149:]`。直接传递密文对象，不解密再加密。
- K 已包含 RoPE，不要再旋转一次。
- 每份导出都有 `signature.json`，按其顺序绑定；若修改 chunk size，不能继续使用固定下标 149。
- 后端必须保证跨调用缓存的 CKKS 上下文、密钥、level 和 scale 兼容；本次数值接线测试未证明 GPU 后端的跨调用兼容性。

## 3. 查看默认真实尺寸接口

只需 Python >=3.10 和 NumPy，计划命令不会加载 Hecate 或生成大权重：

```bash
python verify_manifest.py
python trace_qwen24.py --mode prefill --positions 0 1 --plan
python trace_qwen24.py --mode decode --positions 1 --cached-positions 0 --plan
```

`config/weight_schema.json` 列出全部 291 个权重名称与形状；`interfaces/` 保存上述默认模式的接口快照。

## 4. 权重与测试输入

### 合成权重：用于接线和数值诊断

```bash
python prepare_fixture.py --output-dir fixture_full
```

该命令会在本机生成所有真实尺寸权重，并运行独立明文参考。权重为 float32，NPY 数据约 **2.52 GB**，需要预留额外空间；不会下载模型。NPY 按矩阵分文件，生成时分行处理，读取时使用内存映射。原生跟踪和编译的内存需求远高于权重本身，尚未测量。

各层使用按权重名称派生的不同随机种子，24 层不共享测试矩阵。embedding 和 LM head 按参考代码分别存储，不自动绑定；因此存储标量数量不能直接按“0.5B”估算。

输出含 `weights/weights.json`、291 个 NPY 权重、`input_slots.npy`、`input_embeddings.npy`、token IDs、`reference.npz` 和各层原函数的范围统计。两个输入 token IDs 固定为 1、3，位置为 0、1。合成模型没有语言生成质量含义。

### 已有真实权重

也可向 `--weights` 传入 NPZ，键与 `weight_schema.json` 完全一致，采用 `my_qwen2.py` 的名称，例如 `model_list.23.attention.q_weight.weight`。可以从已有模型实例导出所需参数：

```python
import json
import numpy as np
schema = json.load(open("config/weight_schema.json", encoding="utf-8"))["weights"]
state = model.state_dict()  # 已由调用方安全加载的参考模型实例
arrays = {name: state[name].detach().float().cpu().numpy() for name in schema}
with open("qwen24_weights.npz", "xb") as output:
    np.savez(output, **arrays)
```

本包不自动加载 PyTorch pickle，也不隐式转换 Hugging Face 键名或绑定 embedding/LM head。NPZ 路径会把权重载入内存，内存映射目录更适合大模型。

客户端准备加密前输入：

```bash
python prepare_inputs.py --weights qwen24_weights.npz --token-ids 1 3 --output input_slots.npy
```

真实权重必须另做逐层近似校准及明文参考验证，不能直接沿用合成权重的精度结论。

## 5. 在已有 Hecate / GPU 工程导出

先配置对方现有的 Hecate 原生依赖、`HECATE` 和库路径。私有 `qwen24.kernels` 只导入其 `hecate.expr.Expr/Plain`，不用覆盖对方已有 Python 模块。

```bash
python trace_qwen24.py --mode prefill --positions 0 1 --weights fixture_full/weights --output-dir traced_prefill
python trace_qwen24.py --mode decode-pair --positions 0 1 --weights fixture_full/weights --output-dir traced_pair
python trace_qwen24.py --mode prefill --positions 0 --weights fixture_full/weights --output-dir traced_first
python trace_qwen24.py --mode decode --positions 1 --cached-positions 0 --weights fixture_full/weights --output-dir traced_decode
```

每条命令应在独立进程运行，输出目录必须是新目录。函数名为 `_hecate_qwen25_24layer`。入口使用原模型类的真实 24 层循环；完整词表投影也进入 DSL。

目录中没有预生成的巨大 MLIR 或 GPU 二进制。`hc.save` 完成只是前端导出，之后仍需对方编译、安排 CKKS 参数/真实自举并运行。

**默认矩阵乘法仍是矩形对角线的旋转、乘法、求和实现，主要用于表达计算，尚未做完整模型的 GPU 性能优化。24 层和 LM head 会产生很大的图及明文常量，后端可能需要进一步分段、优化矩阵布局和管理内存。**

## 6. 近似参数、自举与验收

`config/approximations.provisional.json` 显式包含 24 组 Block 参数及最终归一化参数。它们来自此前通过单 Block CPU CKKS 的系数，本次复制为 24 组独立配置：

- RMSNorm 倒平方根：12 阶 Chebyshev，方差加 eps 区间 `[0.5,2]`。
- SiLU：12 阶 Chebyshev，输入区间 `[-2,2]`。
- Attention Softmax：中心化分数界 0.5、最大长度 2、指数 8 阶、倒数迭代 6 次。

**配置中的 max_sequence_length=512 是结构上界，不表示随附近似支持 512 token。默认配置只允许总长度不超过 2，超出会拒绝；增加长度需要新的近似配置和测试。**

DSL 没有显式 bootstrap。scale、rescale、modswitch、重线性化、评估密钥和真实自举由后端负责；不能把旧单 Block 的 103 次自举简单乘 24 作为完整模型的已验证计划。跨层误差累积及真实数据分布均尚未验证。

GPU 测试端将完整解密输出保存为 `[输出密文数,32768]` 的 NPY，建议 complex128：

```bash
python unpack_outputs.py --signature traced_prefill/signature.json --slots gpu_prefill_slots.npy --output gpu_prefill.npz
python check_outputs.py --actual gpu_prefill.npz --reference fixture_full/reference.npz
```

第一步检查尾部并拆成 `logits[151936]`、`keys[24,2,128]`、`values[24,2,128]`。第二步逐项比较参考，默认绝对误差阈值 1e-3；这是初始测试门槛，尚未证明本模型能够在 CKKS 下通过。实部输出不能认证虚部误差，前缀比较不能代替完整槽位尾部检查。

## 7. 本次验证证据

| 检查 | 结果与范围 |
| --- | --- |
| 真实尺寸结构检查 | 24 层、896/4864、完整 LM head、所有缓存形状通过；线性运算被替换为形状模拟，非数值证明 |
| 24 层小尺寸实际 DSL 槽运算 | hidden=8、FFN=16、4 Q heads、2 KV heads、词表 19；prefill / pair / external decode 全部通过 |
| 相对独立原函数参考 | logits 最大误差约 1.87e-8，K 约 3.10e-8，V 约 2.09e-8 |
| 三条两 token 路径一致性 | logits、K、V 差异均为 0；完整槽位尾部为 0 |
| 原有缓存 | 独立 decode 直接复用 Expr 对象，旧缓存未被修改 |
| 全尺寸 24 层数值、CKKS、GPU | 未执行 |

小尺寸测试运行真实 Python 算术与 CLI 入口，但以 NumPy 替代 Hecate Expr，不模拟噪声或模数链。可重跑：

```bash
python verification/validate_numpy.py --output-dir smoke_result
python verification/validate_structure.py
```

`verification/numpy_validation.json` 保存本次数值记录，`verification/structure_validation.json` 保存全尺寸结构检查摘要。四个核心内核与之前交付版本保持逐字节一致，校验来源在 `provenance.json`。

本包由本次工程整合产生，没有经过 `lhy-agent-dsl` 分支的 DeepSeek 生成/验收流程，不能计为那个 Agent 的生成成绩。
