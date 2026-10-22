# R8a：通用化（上）——导入前端补齐、开发模型库、训练两个小模型、规则规划器支持向量输入

前置：R7c 已完成（它正在占用 nix-portable，两轮不要并行）。目标日期：2026-10-24。用户已于 2026-10-21 批准：用本机已有的 CIFAR-10 训练集训练两个小模型，权重存放在仓库之外。

## 0. 背景

- 通用化分两轮完成。R8a 补齐输入侧：前端、模型和规则基线；R8b 再做能力注册表、模式标注、示例库和通用提示词，并在全部开发模型上运行 agent。
- **开发模型（共 4 个）**：

  | 模型 | 来源 | 用途 |
  |---|---|---|
  | `resnet20` | B 线，ReLU | 已有 |
  | `resnet20_silu` | Dacapo 自带的定义和权重 | 检验 SiLU 激活，以及 `size()` 写法的导入 |
  | `cnn_plain` | 本轮训练 | 没有残差的 CNN |
  | `mlp` | 本轮训练 | 纯向量路径 |

- **已知的导入缺口**：Dacapo 的 ResNet 定义（`third_party/ckks-runtime/third_party/dacapo/python/poly/poly/models/ResNet.py`）在 forward 中使用了 `F.avg_pool2d(out, out.size()[3])` 和 `out.view(out.size(0), -1)`。R3 的前端不支持 `size()`。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/intake/torch_mapping.py   改：§2.1 静态解析 size/shape
pt_agent/intake/decompose.py       改：§2.2 global_avgpool_flatten 规则
pt_agent/models/zoo.py             新：§3
pt_agent/models/train_small.py     新：§4
pt_agent/planners/rule.py          改：§5
pt_agent/run.py                    改：§6
pt_agent/experiments/r8a_dev.py    新：§7
pt_agent/tests/test_r8a_*.py       新：§8
../docs/pt_agent/STATUS.md         改
```

## 2. 导入前端

### 2.1 静态解析 `size` 和 `shape`（torch_mapping.py）

- 维护一个字典 `static: fx 节点 -> Python 值`。
- `call_method` 的 target 为 `"size"` 时，取 `args[0].meta["tensor_meta"].shape`：带 dim 参数（位置参数或关键字参数）时结果为对应的 int，否则为 int 组成的 tuple。不生成图节点。
- `call_function` 为 `getattr(x, "shape")` 时同样处理，得到 tuple。
- `operator.getitem` 的第一个参数在 `static` 中时，直接取下标，结果也放进 static。
- 后续节点的参数如果是 static 中的节点，映射时就替换成对应的 Python 值。覆盖的情况包括：`F.avg_pool2d` 的 kernel_size、`view`/`reshape` 的形状参数，以及 `torch.flatten` 的参数。
- 除此之外，如果 static 值被当作张量运算的操作数，按不支持处理，原因写 `"size value used as tensor"`。

### 2.2 新的分解规则 `global_avgpool_flatten`（decompose.py，放在 adaptive_avgpool 规则之后）

触发条件全部满足时，把两个节点合并为一个 `mean` 节点，`axes=(2,3)`、`keepdim=False`，输出张量名沿用 reshape/flatten 的输出名：

- 节点是 avgpool，kernel 等于 (H, W)，stride 等于 kernel，padding 为 (0,0)，输入形状为 (1,C,H,W)；
- 它唯一的消费者是 reshape（目标形状为 (1,C)）或 flatten（start_dim=1）。

只满足 avgpool 本身的条件、后面没有跟 reshape/flatten 时，保持原样不变。

## 3. models/zoo.py

```python
WEIGHTS_SILU = REPO / "third_party/ckks-runtime/third_party/dacapo/examples/data/resnet20.silu.model"
NAMES = ("resnet20", "resnet20_silu", "cnn_plain", "mlp")

@dataclass
class ZooModel:
    name: str
    module: torch.nn.Module            # float32、eval 模式
    example_input: torch.Tensor        # resnet 和 cnn 为 zeros(1,3,32,32)；mlp 为 zeros(1,3072)
    normalization: tuple               # (mean, std)，各 3 个数
    def adapt(self, images: np.ndarray) -> np.ndarray:
        """images 是已经归一化的 (N,3,32,32) 数组，输出模型的输入。mlp 为 reshape(N, 3072)，其余原样返回。"""

def get(name: str, models_dir: Path) -> ZooModel
```

各模型的具体规定：

- **resnet20**：调用 `models.resnet20.load_torch()`，normalization 取 ImageNet 的 mean/std，与 R3 相同。
- **resnet20_silu**：
  - 用 `importlib.util.spec_from_file_location` 加载 Dacapo 的 `ResNet.py`，调用其中的 `resnet20()` 构造模型；
  - 权重用 `torch.load(WEIGHTS_SILU, map_location="cpu", weights_only=True)["state_dict"]` 读入，去掉键名的 `"module."` 前缀后，`load_state_dict(strict=True)`；
  - 归一化方式按 §7 第 1 步的验证结果确定，写在 zoo.py 里的常量 `SILU_NORMALIZATION` 中。
- **cnn_plain**：

  ```python
  class CnnPlain(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False);  self.bn1 = nn.BatchNorm2d(16)
          self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(32)
          self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(64)
          self.fc = nn.Linear(64, 10)
      def forward(self, x):
          x = F.relu(self.bn1(self.conv1(x)))
          x = F.relu(self.bn2(self.conv2(x)))
          x = F.relu(self.bn3(self.conv3(x)))
          return self.fc(x.mean(dim=(2, 3)))
  ```

  权重为 `models_dir / "cnn_plain.pt"`，normalization 取 ImageNet 的 mean/std。
- **mlp**：

  ```python
  class Mlp(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(3072, 256); self.fc2 = nn.Linear(256, 10)
      def forward(self, x):
          return self.fc2(F.relu(self.fc1(x)))
  ```

  权重为 `models_dir / "mlp.pt"`，normalization 取 ImageNet 的 mean/std。

## 4. models/train_small.py（在 Nix 中运行）

```bash
python3 -m pt_agent.tools.nixrun --seconds 4000 -- -m pt_agent.models.train_small --out-dir <models_dir> --model cnn_plain|mlp
```

- **数据**：CIFAR-10 训练集全部 50000 张，用 `data.cifar.read_rows("train", range(50000))` 读取后做 ImageNet 归一化；mlp 的输入展平成 3072 维。
- **训练设置**：
  - `torch.manual_seed(20261022)`，`np.random.default_rng(20261022)`，`torch.set_num_threads(2)`；
  - batch 128；优化器 SGD（momentum 0.9，weight_decay 5e-4）；初始学习率 0.05，按 epoch 做余弦退火到 0；
  - cnn_plain 训练 15 个 epoch，并做数据增强：用 numpy 补 4 个像素的零边后随机裁剪回 32×32，再以 0.5 的概率水平翻转；
  - mlp 训练 20 个 epoch，不做数据增强。
- **输出**：`<model>.pt`（`state_dict`）和 `<model>-train.json`，后者记录每个 epoch 的 loss、训练准确率、耗时，以及训练完成后在 10000 张测试图上的 top-1（torch float32）。测试精度**只用于报告**，不用于选择任何参数，也不提前停止训练。
- **限制**：两个模型各自墙钟不超过 60 分钟，超时就保存当时的权重，并在 json 中标记 `timeout=true`。

## 5. planners/rule.py 的通用化

- 输入张量的 rank 为 2（形状 (1,F)）时，第一行写成 `pack(x, layout="flat")`，参数注解为 `Cipher[1, F]`。
- 一个 linear 的输入如果已经是 flat 值（来自 flat 输入、上一个 linear 的输出，或 flat 上的 act），就不插入 relayout。只有输入来自 `mean` 时才插入 relayout，这与 R6 的行为一致。
- act 的 fn 直接取图节点的 fn（relu、silu 等），不再假定一定是 relu。
- 其余规则不变。对 resnet20 生成的文本必须与 R6c 运行（deg=127、margin=1.5）得到的 program.ptdsl **逐字相同**，这一点作为回归检查。

## 6. run.py 的改动

- `--model` 接受 zoo 中的 4 个名字。新增参数 `--models-dir`，默认值为 `$PT_RESULTS/models`，由外层解析成绝对路径后传给内层。
- graph 由 `import_module(zm.module, (zm.example_input,), name)` 导入。
- 校准、验证和测试图先按 `zm.normalization` 做归一化，再经过 `zm.adapt` 得到模型输入。`data.cifar` 中的读取函数需要新增一个可选参数 `normalization=(mean, std)`，默认值保持 ImageNet 的取值不变。
- 规则规划器的输入 pack 方式由图输入的 rank 决定。

## 7. experiments/r8a_dev.py

运行模式同前：外层和内层分开，结果目录为 `$PT_RESULTS/r8a-dev-<时间戳>`，registry 写 round="R8a"。

1. **resnet20_silu 的归一化**：分别用 ImageNet 的 mean/std 和 CIFAR 常用的 mean=(0.4914, 0.4822, 0.4465)、std=(0.2023, 0.1994, 0.2010)，在 `validation_set(2000)`（训练集）上测 torch 的准确率。取准确率 ≥95% 的那一种；两种都满足时取 ImageNet；都不满足时，记录两种的结果，然后停下并写进"阻塞"栏。**不要用测试集来选。** 确定后把结果写入 `SILU_NORMALIZATION`。
2. 分别训练 cnn_plain 和 mlp（§4）。
3. 对 4 个模型依次执行：
   - import_module → coverage；
   - 用 `reference_forward` 对比 torch float64 的 logits，取 8 张校准图上的最大差，要求 ≤1e-9；
   - 在 10000 张测试图上统计 torch float32 的 top-1；
   - 用规则规划器（deg=127、margin=1.5）跑 `pt_agent.run` 的 `plan,check,mock,count,emit,compile` 各阶段，记录 source_lines、校准集 agree、counts、max_depth、编译结果（耗时、峰值内存、boots），编译后调用 prune。
4. REPORT 中每个模型一行：nodes、coverage、torch_top1、ref_diff、source_lines、agree、max_depth、boots、compile_s。

## 8. 测试（nixrun）

1. **size 解析**：一个 forward 为 `F.avg_pool2d(x, x.size()[3]).view(x.size(0), -1)` 的小模块，输入 (1,4,8,8)，导入后只有一个 mean 节点（axes=(2,3)），reference 与 torch 一致。
2. **resnet20_silu 导入**：74 个节点，op 计数与 resnet20 完全相同，所有 act 的 fn 都是 "silu"；reference 与 torch float64 的差 ≤1e-9（用 2 张校准图）。
3. **cnn_plain 和 mlp 导入**（用随机初始化的权重即可）：cnn_plain 的 op 计数为 {conv2d:3, batchnorm:3, act:3, mean:1, linear:1}；mlp 为 {linear:2, act:1}；两者的 reference 都与 torch 一致。
4. **规则规划器**：
   - mlp 的程序以 `pack(x, layout="flat")` 开头，不含 relayout，能通过 parse 和 typecheck，并且在 Mock 精确模式下与 reference 一致（batch=2，随机权重）；
   - cnn_plain 的程序在 Mock 精确模式下与 reference 一致；
   - resnet20 的程序与 R6c 的 program.ptdsl 逐字相同。
5. `size()` 的值被当作张量操作数时，抛 IntakeError，原因是 "size value used as tensor"。
6. R1–R7c 的测试全部通过。

## 9. 不要做

- 不用测试集选择任何参数，也不根据测试精度调整训练；不下载任何东西。
- 不修改 kernel、lowering、代价模型和 agent，也不修改 Dacapo 的 `ResNet.py` 和权重文件。
- 不在本轮运行 LLM（R8b 才运行）。
- 训练出的权重不放进仓库。
