# 模型输入格式

新模型使用 `poseidon-model-graph-v1` JSON。它描述数学计算，不包含 DSL 源码、API 配置或测试输入数据。

## 示例：两层平方激活 MLP

保存为仓库根目录的 `model.json`：

```json
{
  "format": "poseidon-model-graph-v1",
  "id": "small_mlp",
  "inputs": [{"name": "x", "shape": [4]}],
  "constants": {
    "w1": [[0.1, 0.2, -0.1, 0.1], [0.2, -0.1, 0.1, 0.2]],
    "b1": [0.01, -0.01],
    "w2": [[0.2, -0.1]],
    "b2": [0.01]
  },
  "nodes": [
    {"id": "fc1", "op": "linear", "inputs": ["x", "w1", "b1"], "attrs": {}, "outputs": ["h"]},
    {"id": "act", "op": "square", "inputs": ["h"], "attrs": {}, "outputs": ["a"]},
    {"id": "fc2", "op": "linear", "inputs": ["a", "w2", "b2"], "attrs": {}, "outputs": ["y"]}
  ],
  "outputs": [{"name": "prediction", "value": "y"}]
}
```

`inputs` 声明名字和形状；`constants` 保存公开权重；`nodes` 按依赖顺序排列；`outputs` 选择需要返回的节点结果。节点通过名字连接，不需要指定模型类别。

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case model.json --prepare --compiler-configuration seal-cpu-eva-w45-v1
```

## 支持的算子

| 用途 | 算子名 | 说明 |
|---|---|---|
| 算术 | add、subtract、multiply、negate、square、power | power 指数仅 2 或 4 |
| 多项式 | polynomial | 幂基或 Chebyshev 基，系数公开 |
| 全连接 | linear | 最后一维加权求和，可含偏置 |
| 归一化 | batch_norm | 固定统计量的推理计算 |
| 形状 | flatten、reshape、transpose、permute | 静态形状；reshape 不改变元素数 |
| 组合 | concat、stack、slice、split | 轴、范围和分段大小静态确定 |
| 归约 | sum、mean | 显式指定 axes、keepdims |
| 旋转 | rotate | 按逻辑张量定义旋转，由布局转换为密文操作 |
| 卷积 | conv1d、conv2d | 支持受限 stride、padding、dilation、groups |
| 平均池化 | avg_pool1d、avg_pool2d | 窗口、步长、padding 和边界除数显式指定 |

完整属性检查在 [benchmark_graph.py](../../scripts/baseline/benchmark_graph.py)；更多 JSON 见[案例目录](../../scripts/baseline/cases/)。

可以连接成 MLP、卷积小网络、残差、多分支、多输入融合和多输出图。张量形状与广播需匹配；不是任意参数组合都能编译。

## 规模与数据

| 项目 | 上限 |
|---|---:|
| 输入 / 命名输出 | 各 4 个 |
| 输入 / 输出总元素 | 各 256 |
| 张量 rank | 1–4 |
| 图节点 | 64 |
| 公开常量条目 | 32 |
| 常量元素 | 由 validator 按常量形状检查，最多 4096 |
| 模型 JSON | 128 KiB |

常量必须是有限实数或规则数组，绝对值不超过 1024。模型是静态无环图，仅支持推理；不支持训练、随机 forward、动态 shape 或加密数据决定的分支。

输入数据不放进模型 JSON。验收器生成零、有符号、固定随机和边界四组输入。

默认使用 packed 布局。显式 `--unified-chunk-period P` 可选择 4/8/16/32/64/128/256 的分块周期；当前还受最多 4 个模型输入块和 4 个输出块限制。增加逻辑元素上限不会自动增加密文容量。

## Python 和高层算子

[受限 Python / 算子图入口](operator-decomposition-and-python.md) 先把源码或高层算子分解为上述 JSON，再进入生成流程。普通 PyTorch `nn.Module` 项目、ONNX、checkpoint 不直接接受。

非线性函数必须给出近似方法、区间和参数。不会自动把 ReLU、Softmax 或 MaxPool 替换成其他函数。

## 旧输入兼容

原 CLI 的 `--case` 保留以下格式；新应用组件只直接接收统一图。

| schema | 内容 |
|---|---|
| 1 | 8 类内置模型，各 6 个配置 |
| 2 | 单输入自定义图，4 个输入元素 |
| 3 | 2–4 个输入，每个 4 个元素 |
| 4 | 单输入 5–16 个元素，分为 period-4 密文块 |
| 5 | 单输入 1–256 个元素，可变 packed 周期 |

旧 schema 的算子和输出布局限制各不相同，不能套用统一图的全部能力。新项目优先使用统一图；旧文件可以继续使用原格式。
