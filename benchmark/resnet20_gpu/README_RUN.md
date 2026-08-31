# Poseidon GPU ResNet20 运行指南

本文说明如何在 Poseidon 仓库中编译、检查和运行 GPU ResNet20。
本目录已经包含运行所需的 ResNet20 权重、CIFAR-10 测试输入、标签和
ReLU 多项式系数，不依赖外部 `Trident/resnet20` 或兄弟目录
`benchmark/resnet50_gpu`。

## 1. 目录位置

```bash
cd /home/guoshuai/github/poseidon/poseidon/benchmark/resnet20_gpu
```

后续命令均在这个目录执行。推荐通过 `run.sh` 运行；它会自动配置 CMake、
增量编译程序、执行命令，并将标准输出和错误输出保存到 `output/`。

## 2. 环境要求

- Linux；
- CMake 3.26.4 或更高版本；
- 支持 C++20 的编译器；
- CUDA Toolkit 和可用的 NVIDIA 驱动；
- NVIDIA GPU。完整同态推理需要较多显存，推荐使用空闲的 32 GiB GPU；
- Poseidon 主源码和仓库内的 `third_party/rmm`。

当前机器使用 CUDA 12.2 和 Tesla V100-SXM2-32GB。V100 的 CUDA 架构是
`70`，也是 `run.sh` 的默认值。运行前可以检查 GPU：

```bash
nvidia-smi
```

下面的示例使用物理卡 2。设置 `CUDA_VISIBLE_DEVICES=2` 后，程序内部看到的
设备编号会变成 0，这是 CUDA 的正常设备映射行为。

## 3. 数据文件

程序默认从当前目录读取：

```text
data/resnet20/
├── pretrained_parameters/resnet20_new/  # ResNet20 权重和 BN 参数
├── relu_param/d13.txt                    # ReLU 多项式系数
└── testFile/
    ├── test_values.txt                   # 1000 张 CIFAR-10 测试图片
    └── test_label.txt                    # 1000 个标签
```

测试图片编号为 `0` 到 `999`。通常不需要设置额外数据路径。如果需要对比另一
份兼容数据，可以设置：

```bash
export POSEIDON_TRIDENT_RESNET20_ROOT=/path/to/resnet20
```

该变量只覆盖权重和测试输入的位置；ReLU 系数仍使用本目录中的文件。取消覆盖：

```bash
unset POSEIDON_TRIDENT_RESNET20_ROOT
```

## 4. 首次运行和快速检查

先检查网络拓扑和本地数据：

```bash
./run.sh --topology-check
./run.sh --weights-check
```

权重检查成功时应看到类似输出：

```text
weights_check conv=19 bn=19 fc=640 image_values=3072 label0=3
```

然后在卡 2 上执行最小 GPU 正确性测试：

```bash
CUDA_VISIBLE_DEVICES=2 ./run.sh --smoke
```

成功时会输出 `GPU ResNet20 smoke max_error=...`，误差应小于 `1e-5`。
当前 40-bit 计算 scale 版本在卡 2 的验证误差约为 `1.1e-7`。

还可以检查 ResNet20 降采样残差分支：

```bash
CUDA_VISIBLE_DEVICES=2 ./run.sh --shortcut-check
```

首次构建会编译 Poseidon、RMM 和 GPU 源码，耗时会比后续增量构建长。

## 5. 模数链和 scale

当前配置在 `gpu_config.h`，物理 Q/P 素数值由 `gpu_config.cpp` 生成。它参考
CPU 最新配置，逻辑参数为：

- 普通卷积、ReLU 等网络计算使用 `scale = 2^40`；
- Bootstrap 的 EvalMod 内部使用 `scale = 2^45`；
- Bootstrap 输出恢复为 `scale = 2^40`，后续网络不需要额外升 scale；
- Bootstrap 使用 Q 链前缀 `Q34`，刷新后扩展回完整应用链 `Q36`。

GPU RNS residue 使用 `uint32`，因此不能直接放入 CPU 的单个 40/45-bit
素数。当前物理链是：

```text
Q: 36 个物理素数，总计 1132 bit
   q[0..1]   : 2 x 32-bit q0 base
   q[2..33]  : Q34 Bootstrap 验证链的 32 个 tail 素数
               (20x32, 8x31, 2x30, 2x28 bit)
   q[34..35] : 2 x 32-bit 应用扩展
P: 18 x 32-bit
dnum: 2
```

一个 scale-`2^40` 的密文乘密文逻辑层使用两个32-bit物理 Q：乘法后 scale
为 `2^80`，第一次 rescale 后约为 `2^48`，再乘校正常数并 rescale 回
`2^40`。因此 `[15,15,27]` ReLU 的14个逻辑乘法层实际消耗28个物理 Q；
stem 中可看到 `q=35 -> q=7`。普通 Bootstrap+ReLU 则是刷新回 Q36 后输出
`q=8, log2_scale=40`。

程序启动时会打印实际 `Q/P/dnum` 和三个 scale。逐层日志同时打印 `q` 与
`log2_scale`，可用于检查层数消耗和 scale 漂移。

## 6. 完整加密推理

对第 0 张 CIFAR-10 图片执行完整 ResNet20 推理：

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --infer 0
```

命令格式：

```text
./run.sh --infer IMAGE_ID [MAX_BLOCKS]
```

- `IMAGE_ID`：测试图片编号，范围为 `0–999`；
- `MAX_BLOCKS`：可选，最多运行多少个 BasicBlock，完整网络为 9 个；
- `--infer 0 0`：只检查加密 stem；
- `--infer 0 1`：运行 stem 和第一个 BasicBlock；
- 不传 `MAX_BLOCKS`：运行全部 9 个 BasicBlock 和加密分类头。

完整推理结束后重点查看：

```text
true_label=...
predicted_label=...
inference_total_elapsed_seconds=...
```

`inference_total_elapsed_seconds` 是普通端到端计时，包含权重加载、运行时准备、
加密网络和结果处理等工作，不等于纯 GPU 网络执行时间。

如需逐层解密并和 CPU 明文参考结果比较，可以运行：

```bash
CUDA_VISIBLE_DEVICES=2 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1 \
  ./run.sh --infer 0
```

逐层验证需要 CPU/GPU 同步和解密，会显著增加运行时间，不适合性能计时。

## 7. GPU-only 计时

如果只想测量数据、密文、编码权重、评估密钥和 Bootstrap 常量都准备好之后的
GPU 网络执行时间，使用：

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --gpu-only 0
```

这个模式会：

1. 加载本地权重和第 0 张图片；
2. 初始化运行时、密钥和 GPU 常量；
3. 执行一次不计时的准备推理，使模型操作数和缓存驻留 GPU；
4. 同步 GPU；
5. 对第二次相同推理计时；
6. 在计时结束后传回、解密并检查结果。

重点查看输出：

```text
gpu_only_preloaded_elapsed_seconds=...
gpu_only_elapsed_seconds=...
preloaded_replay_max_logit_error=...
```

其中 `gpu_only_preloaded_elapsed_seconds` 是准备完成后的 GPU-only 墙钟时间。
最终 D2H、解密、预测检查以及第一次准备推理不在这个区间内。

## 8. 单独检查加密分类头

只验证 global average pooling 和全连接分类头：

```bash
CUDA_VISIBLE_DEVICES=2 ./run.sh --head-check 0
```

输出中的 `max_logit_error` 是加密分类头和明文参考结果之间的最大误差。

## 9. 构建和日志设置

`run.sh` 支持以下环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `70` | 设置目标 GPU 架构 |
| `BUILD_JOBS` | `2` | 设置并行编译任务数 |
| `POSEIDON_GPU_RESNET20_BUILD_DIR` | `./build` | 设置构建目录 |
| `POSEIDON_GPU_RESNET20_OUTPUT_DIR` | `./output` | 设置日志目录 |
| `POSEIDON_GPU_RESNET20_LOG_FILE` | 自动生成 | 指定单次运行日志文件 |
| `POSEIDON_GPU_RESNET20_DNUM` | `2` | 设置 HYBRID key switching 的分解数量 |
| `CMAKE_BIN` | PATH 中的 `cmake` | 指定 CMake 程序 |

例如，使用独立构建目录进行一次全新验证：

```bash
POSEIDON_GPU_RESNET20_BUILD_DIR=/tmp/resnet20_gpu_build \
CUDA_VISIBLE_DEVICES=2 \
  ./run.sh --smoke
```

每次执行都会产生类似下面的日志：

```text
output/resnet20_gpu_YYYYMMDD_HHMMSS.log
```

## 10. 常见问题

### 找不到 CUDA 或 `nvcc`

确认 `nvcc --version` 和 `nvidia-smi` 正常，并检查 CUDA Toolkit 的 PATH、
库路径以及驱动版本。

### `no kernel image is available`

编译架构与 GPU 不匹配。V100 使用 `70`；其他 GPU 请传入对应架构，例如：

```bash
CMAKE_CUDA_ARCHITECTURES=80 CUDA_VISIBLE_DEVICES=2 ./run.sh --smoke
```

### 找不到权重或测试数据

先检查：

```bash
./run.sh --weights-check
```

确认没有错误设置 `POSEIDON_TRIDENT_RESNET20_ROOT`，并确认
`data/resnet20` 下的数据文件完整。

### GPU 显存不足

使用 `nvidia-smi` 检查目标卡是否有其他进程。完整推理和 Bootstrap 的显存需求
远高于 `--smoke`，应优先选择空闲的 32 GiB 卡。本机可使用卡 2：

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep ./run.sh --gpu-only 0
```

### 为什么第一次运行很慢

第一次运行包含完整编译、密钥生成、Bootstrap 初始化、模型编码和 GPU 缓存准备。
需要观察准备完成后的执行性能时，以 `--gpu-only` 输出的
`gpu_only_preloaded_elapsed_seconds` 为准。

当前卷积、Option-A shortcut、平均池化和全局池化使用 GPU 融合明文乘累加：
同一阶段的乘积先保持在高 scale，由 `multiply_plain_accumulate` CUDA 内核直接
累加，阶段结束后只做一次 rescale。预热后的第二遍会直接复用 GPU 上的编码明文，
不会在计时区间重复做 CKKS 编码或明文 H2D 上传；模数 level 消耗与优化前一致。
卡 2 的完整 image-0 验证结果为 `8.77687 s`，预测类别为 `3`，两遍 logits
完全一致（`preloaded_replay_max_logit_error=0`）。

## 11. 最短运行流程

```bash
cd /home/guoshuai/github/poseidon/poseidon/benchmark/resnet20_gpu
unset POSEIDON_TRIDENT_RESNET20_ROOT
./run.sh --weights-check
CUDA_VISIBLE_DEVICES=2 ./run.sh --smoke
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep ./run.sh --gpu-only 0
```
