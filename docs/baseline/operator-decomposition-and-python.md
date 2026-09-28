# Python 与高层算子输入

这两个入口先生成统一模型 JSON，再交给 Agent。它们不调用模型 API，也不会执行上传的 Python 源码。

## 高层算子图

格式为 `poseidon-operator-graph-v1`，字段与[统一图](supported-inputs-models-configurations.md)相同。除基础算子外，还支持：

| 算子 | 分解方式 |
|---|---|
| dot、rank-2 matmul | 基础乘法与归约 |
| exp | 显式 Taylor 次数和有效区间 |
| reciprocal | 固定 Goldschmidt 迭代 |
| rsqrt | 固定 Newton 迭代 |
| rms_norm | 平方均值、倒平方根近似和增益 |
| softmax | 指数近似、归约和倒数近似 |
| rope | 向量 split-half RoPE |

近似方法、区间和参数必须在模型中声明。原图与分解后图都要满足节点和容量限制。分解结果、原始图和节点来源会一起保存。

```bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-decomposition-v1/rmsnorm.json
```

默认只检查计划。实际执行参考检查并生成新目录：

```bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-decomposition-v1/rmsnorm.json \
  --write --output "$HOME/poseidon-work/imported-rmsnorm"
```

完成后将输出目录中的 `model.json` 传给 `scripts/agent.py`。

## 受限 Python

清单格式为 `poseidon-python-model-v1`，包含 id、files（文件名和 SHA-256）、entry（模块.函数）、inputs、公开 constants 和命名 outputs。

最多 8 个平铺的 `.py` 文件，总源码 64 KiB，每文件最多 4096 个 AST 节点。

支持：

- 纯函数、清单内本地 helper、名称赋值。
- 公开有界 range 循环、列表/元组和公开容器索引。
- 算术、已映射的 torch / F.linear 调用。
- 附带显式近似参数的 exp、rsqrt、reciprocal、softmax。
- 静态命名空间 `poseidon.rms_norm` 和 `poseidon.rope`。

不支持 `nn.Module` 类定义、默认/可变函数参数、递归、动态张量索引、数据相关分支、任意属性或动态调用、文件/网络访问、torch.load、eval/exec、动态导入。近似函数的额外参数是本前端约定，不是普通 PyTorch 的调用签名。

```bash
python3 -B scripts/prepare_model.py \
  --python-manifest scripts/baseline/cases/operator-decomposition-v1/mlp-python.json \
  --write --output "$HOME/poseidon-work/imported-mlp"

python3 -B scripts/agent.py --backend local candidate -- \
  --case "$HOME/poseidon-work/imported-mlp/model.json" \
  --self-test --compiler-configuration seal-cpu-eva-w45-v1
```

平台与工作目录环境变量沿用 [README](../../scripts/README.md)。输出目录必须尚不存在。

## 当前示例

| 示例 | 状态 |
|---|---|
| 4→2→2 平方激活 MLP | Python 导入、参考检查、真实密态计算、导出重放通过 |
| 小型 RMSNorm | 参考检查通过，编译 level/scale 阻塞 |
| 单头小型 Attention decode | 参考检查通过，编译 level/scale 阻塞 |

Attention 示例含投影、RoPE、密文点积、近似 Softmax 和显式 K/V 输入输出；不是完整 Transformer 或任意长度 Attention。

参考检查分别记录展开计算误差和相对原函数的近似误差。通过展开计算的数值检查，不代表近似已达到应用的精度要求。

导出 DSL 包时可通过 `--model-import` 带入原始模型来源，见[程序包接口](agent-component-v1.md)。上述示例是离线导入验收，不是新的真实 Agent 生成成绩。
