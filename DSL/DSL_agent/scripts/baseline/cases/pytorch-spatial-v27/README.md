# PyTorch 空间计算示例

| 示例 | 计算 |
|---|---|
| linear_resize | 1D linear resize，长度 3 → 5 |
| grouped_conv_transpose | 分组 ConvTranspose1d，公开 NPY 权重 |
| masked_resize | 字典输入、resize、公开布尔 mask、命名输出 |

来源清单采用 `poseidon-trusted-pytorch-v27`。导入命令：

~~~bash
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/pytorch-spatial-v27/linear_resize/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/imported-pytorch-spatial"
~~~

清单绑定模型源码、输入调用结构及公开权重。生成与执行用法见 [Agent README](../../../README.md)。

三个示例均通过 Poseidon CPU 编译、数值验证和程序包重载。`masked_resize` 还通过新输入及缓存复用。[验收结果](../../../../docs/baseline/agent-spatial-source-offline-results-r1406.md)。候选来源为规则基线，真实 Agent 生成另行评测。
