# PyTorch 幂运算与公开除法

可信 PyTorch 来源新增显式 `poseidon-trusted-pytorch-v21`，支持静态平方、四次方和公开标量除法。
本批 8 个新来源模型与 2 个兼容模型具有 Poseidon CPU 编译、密态数值和程序包复用证据。
候选来源为规则基线；本批真实 Agent 生成与付费调用均为 0。

## 输入范围

| 输入写法 | 范围 |
|---|---|
| `x ** n`、`torch.pow`、`x.pow`、`x.pow_` | 公开固定指数 2 或 4 |
| `x / d`、`torch.div/divide/true_divide` | 有限非零公开标量 |
| `x.div/divide/true_divide` 及其原地形式 | 同上，保持共享视图与快照语义 |
| 标量 Tensor 除数 | 直接声明的 float64 NPY 状态，绑定形状与内容哈希 |

除数满足 `abs(d) <= 1024` 和 `abs(1/d) <= 1024`。
动态指数、密文除数、计算得到的 Tensor 除数、向量除数、rounding 和 out 写入返回不支持。

幂运算分解到已有 square；公开除法分解到固定公开倒数乘法。
原始源码、NPY 权重、FX 规范化过程及独立参考保留在来源包中。
资源上限沿用所选配置；默认来源清单仍为 v18。

## 使用

~~~python
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.pow(x, 4) / 2

def make():
    return Model().eval()
~~~

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:2 --output out \
  --source-format poseidon-trusted-pytorch-v21 \
  --resource-profile general-tensors-v1 --manifest ./source.json
~~~

将清单交给现有 `prepare_model.py` 或应用 `submit_source`。
本批采用 `capacity-public-tensors-v9`、`explicit-v13`、period 4 和 Poseidon w45。

## 验收

| 独立绑定 | 模型结果 | 成功编译 | 密态调用 | 输入组／输出值 |
|---|---:|---:|---:|---:|
| 首轮 | 8 通过／2 编译失败 | 8 | 16 | 64／200 |
| 相邻掩码修复试验 | 2 通过／2 编译失败 | 2 | 4 | 16／40 |
| 当前槽位范围修复 | 4 通过／0 失败 | 4 | 8 | 32／112 |

当前 4 例是原两个失败模型、标量共享写入回归和 MLP 回归。
首轮其余 6 个新模型的候选逐字节保持，编译／执行身份相同；复用原证据并单独列出绑定。
总计 10 个不同模型，14 个通过程序包；重复执行不增加模型数。

静态控制与兼容测试共 416 项通过，failed/errors/skipped 为 0。
全部执行共 112 组、352 个输出值，最大绝对误差 `2.6638145586566075e-7`。
每份通过程序包完成哈希检查、重新加载、4 组新输入执行和独立参考比较，复用时生成／编译调用为 0。
真实产物包含 frontend tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec 和 plaintext bundle。

门限保持：

~~~text
abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)
~~~

## 两项编译失败的处理

`scalar_shared_power_division` 在首轮累计 scale 750，`diagonal_square_division` 为 735，均超过固定编译预算 720。
冗余掩码使 gather/scatter 与原地视图写回增加连续公开乘法。
相邻掩码试验中的新零分支触发了 level 合并失败，该处理已撤回；原失败记录保留。

当前 `capacity-algebraic-schedule-v4` 根据逻辑槽位范围删除可证明多余的公开二值掩码，
覆盖旋转、加减、平方及密文乘法。未知公开权重保留全范围，互不相交掩码保留原运算。
该处理仅用于可信规则基线，Provider 及历史 Agent 源码保持原样；逻辑槽位证明不替代 CKKS 数值验收。

## 环境与后续

开发与验收位于 ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e480…`。
654 项后端／参考保护文件、两份历史请求及旧付费提案哈希保持。
Poseidon、CKKS Runtime、Dacapo、安全参数和 reference 算法修改为 0；x86 配置保持，本批实际执行平台为 ARM。
新增结果约 104.80 MiB，总 results 约 26.16 GiB，保持原时间与空间边界。

后续将新增来源写法汇总到当前绑定的真实 Agent 评测，原未执行 r1341 提案保留并标为失效。
bootstrap、较大模型、密文交接、应用精度及 CUDA 实机验收继续按原计划推进。

[机器报告](agent-pytorch-exact-arithmetic-acceptance-r1348.json) ·
[输入说明](operator-decomposition-and-python.md#pytorch-v21-幂运算与公开除法)。
