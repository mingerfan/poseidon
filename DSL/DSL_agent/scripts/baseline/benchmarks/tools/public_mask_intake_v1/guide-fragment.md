## PyTorch v33 公开条件选择

选择 `poseidon-trusted-pytorch-v33`，使用固定公开布尔 mask 选择加密张量的元素。
mask 可以广播，按原始 NPY 权重绑定；输入和输出形状在请求前确定。

| 调用 | 用途 |
|---|---|
| `torch.where(mask, a, b)` | 从两个张量或公开值中逐元素选择 |
| `a.where(mask, b)` | Tensor 方法形式的逐元素选择 |
| `x.masked_fill(mask, value)` | 用有限公开标量填充选中的元素 |
| `x.masked_fill_(mask, value)` | 原地填充，保留别名与写入前快照的语义 |

全公开的 where／masked_fill 在接入时计算并保存为固定公开参数；
逻辑 dtype、类型提升和 float32 舍入随来源记录绑定。
加密选择分解为已有的公开 0/1 乘法与加法，使用当前编译与 Poseidon CPU 链路。

`model.py`：

~~~python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("keep", torch.zeros(4, dtype=torch.bool))

    def forward(self, x):
        return torch.where(self.keep, x, 0.125)

def make():
    return Model().double().eval()
~~~

`keep.npy` 保存公开布尔数组，例如 `[True, False, True, False]`。
通过已有 NumPy 环境创建：

~~~python
import numpy as np
np.save("keep.npy", np.array([True, False, True, False], dtype=np.bool_), allow_pickle=False)
~~~

~~~sh
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:4 --output output \
  --weight keep=./keep.npy \
  --source-format poseidon-trusted-pytorch-v33 \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

随后使用 `prepare_model.py --trusted-pytorch-manifest` 或应用 `submit_source`。
v33 沿用公开构造、参数变换、本地源码包、结构化输入输出和程序包执行接口。

原地填充使用已登记的有序别名规则。可信参考在计算输入的副本上执行，
调用者传入的 Tensor 和输入容器保持原值；公开权重与 buffer 作为固定状态绑定。

条件限于已绑定的公开 bool 张量。加密计算分支沿用 float64 语义，公开填充值为有限标量；
各张量、派生常量、图和物理密文资源受所选配置约束。
本版本的数值验收针对有限静态输入，输入相关的加密条件分支列为未支持范围。

默认来源版本为 v18，latest 为 v28；v33 使用上述显式选项。
