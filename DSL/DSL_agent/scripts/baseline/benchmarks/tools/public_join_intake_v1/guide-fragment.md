## PyTorch v34 静态拼接

`poseidon-trusted-pytorch-v34` 支持 `torch.cat`、`torch.concat`、`torch.concatenate` 和 `torch.stack`。
拼接序列可包含加密张量、公开参数和 buffer，也可在模型中先组合公开量，再参与后续计算。

~~~python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tensor(
            [[0.2, -0.1], [0.3, 0.4]], dtype=torch.float64))

    def forward(self, x):
        return torch.cat((x, self.bias), dim=0)

def make():
    return Model().eval()
~~~

输入调用文件 `call.json`：

~~~json
{"format":"poseidon-pytorch-call-v1","args":[{"kind":"tensor","name":"x"}],"kwargs":{}}
~~~

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2,2 --output output \
  --source-format poseidon-trusted-pytorch-v34 --call ./call.json \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

随后使用 `prepare_model.py --trusted-pytorch-manifest` 或应用 `submit_source`。
公开段由接入层绑定并规范化，派生常量、额外密文运算及来源随程序包记录；
Agent 生成规则选择 `explicit-v14`。

| 配置 | 范围 |
|---|---|
| 序列 | 1–8 个静态 Tensor |
| cat / concat / concatenate | 相同 rank，非拼接轴尺寸一致 |
| stack | 相同 shape，可堆叠标量 |
| 轴 | 静态整数，支持合法负轴 |
| 公开 dtype | bool、int32、int64、float32、float64 |
| 加密计算 | float64；混合拼接的公开整数限于 ±2^53 |
| 组合 | 固定 mask、公开变换、拆分、Linear、残差和结构化输出 |

公开 float32 保留其实际舍入结果。张量、派生常量和图大小采用所选资源配置。
默认来源 v18，latest 为 v28；v34 使用上述显式选项。
