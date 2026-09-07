# Poseidon GPU ResNet20 运行指南

本文说明如何在 Poseidon 仓库中编译、检查和运行 GPU ResNet20。
本目录已经包含运行所需的 ResNet20 权重、CIFAR-10 测试输入、标签和
ReLU 多项式系数，不依赖外部 `Trident/resnet20` 或兄弟目录
`benchmark/resnet50_gpu`。

## 1. 目录位置

```bash
cd /home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu
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

批量 Hoist 旋转可单独与普通直连旋转比较：

```bash
CUDA_VISIBLE_DEVICES=2 ./run.sh --hoist-check
```

成功时 `hoisted rotate max_error` 应小于 `1e-5`。

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

要使用逐 Q 直连密钥计划进行一次可截断的正确性运行，使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_GPU_MAX_POOL_MB=24576 \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --direct-infer 0 1
```

`--direct-infer IMAGE_ID [MAX_BLOCKS]` 会初始化精确的 139-key 计划，但只执行
一次，不建立完整模型预加载缓存。它适合在运行完整 `--gpu-only` 前逐步验证
dnum=4。不传 `MAX_BLOCKS`、完整运行 9 个 block 时，还会在计时区间之后用独立
明文 ResNet20 计算参考 logits，要求预测类别一致且最大 logits 误差不超过 `0.1`。
截断运行仍可用 `POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1` 打开逐层解密验证。
`POSEIDON_GPU_MAX_POOL_MB` 限制 RMM 池最大容量，超过预算时会受控失败；它不
包括 CUDA 上下文和第三方非 RMM 分配，因此必须给物理显存保留余量。

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

网络需要 109 个不同的应用直连旋转 step，但不会再在 Q3--Q8 的每个 level
重复生成全部 109 把密钥。一次完整拓扑 trace 得到的逐层数量为：Q3 `9`、
Q4 `7`、Q5 `4`、Q6 `16`、Q7 `23`、Q8 `80`，合计 `139` 把逐层密钥；旧布局
为 `109*6=654` 把，因此减少 `515` 把（约 `78.7%`）。109-step 并集保持不变。
Bootstrap 的 39 个 DFT 旋转步长使用独立密钥。卷积、Pool 和 FC 的同源多旋转
会共享一次 HYBRID 分解，不在计算过程中临时生成或组合旋转密钥。

应用直连旋转默认使用按层缩减的特殊模数基。全局参数仍为
`Q=36/P=18/dnum=2`；ResNet20 应用 KeySwitch
实际访问的低层分别建立 `Q3/P3`、`Q4/P4`、...、`Q8/P8` 的单分解上下文，
同时覆盖直接旋转和低层 relinearization。也就是说，Q8 KeySwitch 不再携带
18 个 P limb，而只使用 8 个；Q7 同理只使用 7 个。高层 relinearization、
主密文 Q 链、scale 和网络层数都没有改变。

选择规则等价于 `P_active=min(Q_active, P_global)`：当当前 Q 不大于全局 P
数量时令 `P_active=Q_active`，因此也适用于其他全局 dnum 配置。运行时只为
模型明确声明会访问的 Q level 预生成密钥，避免为没有出现的 level 常驻无用
密钥。需要做旧路径 A/B 时可关闭此功能：

```bash
CUDA_VISIBLE_DEVICES=2 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_LEVEL_AWARE_ROTATION_P=0 \
  ./run.sh --gpu-only 0
```

另外提供 `fixed_dnum` 实验路线。它按照当前 Q 和配置的目标 dnum 选择
`P_active=ceil(Q_active/dnum)`，从而在降低 P 基宽度的同时尽量保持分解数量。
Poseidon 在 `P=1` 时会切换到 BV，因此 GPU HYBRID 路径强制最小 `P=2`。
对于 `Q36/P9/dnum4`，典型映射包括 Q32/P8/dnum4、Q18/P5/dnum4、
Q10/P3/dnum4、Q8/P2/dnum4 和 Q7/P2/dnum4。Q3--Q6 等极低层受整数分块和
最小 P2 限制，effective dnum 可能小于目标值。

固定-dnum 模式既用于卷积/Pool/FC 的 rotation，也用于 ReLU 的高 Q
relinearization。rotation context 在推理前创建；只含 relinearization 的高 Q
context 在第一次非计时准备推理中按需创建，第二次 GPU-only 推理直接复用。
相同 P 宽度的多个 Q 层共享最大 Q 的一套 context/key，例如 Q29--Q32 共享
Q32/P8、Q25--Q28 共享 Q28/P7，避免为每一个 Q 常驻一套重复密钥。
它是显式 opt-in，原有 `P=Q/dnum1` 路径仍为默认值：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_APPLICATION_KEYSWITCH_P_MODE=fixed_dnum \
POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --gpu-only 0
```

先运行小型正确性检查时使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_GPU_MAX_POOL_MB=8192 \
POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --fixed-dnum-check
```

该检查覆盖卷积代表层 Q7/Q8 的 hoisted rotation，以及 ReLU 代表层 Q32 的
relinearization，不初始化 Bootstrap，也不运行完整网络。固定-dnum rotation
key 的 digit 数多于默认单-digit 路径，因此完整模型的密钥显存会增加，应在
H100 上重新采样峰值后再设正式内存池上限。

当前 V100 小型检查在 8 GiB RMM 上限内通过：Q7/P2/dnum4 与
Q8/P2/dnum4 hoisted rotation 的最大槽误差为 `5.91352e-5`，Q32/P8/dnum4
relinearization 的最大槽误差为 `4.15906e-8`。使用 classic Bootstrap 和
28 GiB RMM 上限的 `--direct-infer 0 1` 也通过，首个 block 的 stem、conv1、
conv2 和 block-output 最大误差依次为 `8.57304e-7`、`1.71907e-5`、
`3.36059e-5`、`3.9305e-5`。该次 `171.744 s` 包含初始化、密钥生成、编码和
逐层解密验证，不是 fixed-dnum 性能结果。

Bootstrap 也会按实际 KeySwitch 层级缩减 P，但高层仍保留配置的全局 HYBRID
分解。当前 Q36/Bootstrap-Q34 计划实际在 Q10、Q12、Q14、Q15 做旋转或共轭，
并在 Q17 执行需要按层路由的 relinearization：

- 全局 dnum=2/P18 时建立 Q10/P10、Q12/P12、Q14/P14、Q15/P15 和 Q17/P17；
- 全局 dnum=3/P12 时只有 Q10/P10 比全局 P12 更小；
- 全局 dnum=4/P9 时最低 KeySwitch 层级 Q10 已大于 P9，没有 P>Q 冗余，因而
  不建立额外上下文。

只在一次 Bootstrap 调用期间启用该分派，不会改变卷积、ReLU 或普通应用旋转
选择的密钥。可用下面的环境变量回退到全局 P，进行同二进制 A/B：

```bash
CUDA_VISIBLE_DEVICES=2 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_LEVEL_AWARE_BOOTSTRAP_P=0 \
  ./run.sh --gpu-only 0
```

也可以在同一密钥、同一输入密文下直接比较按层 P 和全局 P 的 Bootstrap 输出：

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_RESNET20_DNUM=2 \
  ./run.sh --bootstrap-check
```

只执行一次 Bootstrap、同时输出驻留显存快照、延迟和复数槽误差时，使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=classic \
  ./run.sh --bootstrap-single-check
```

ResNet20 的 double-hoist Bootstrap 是显式 opt-in，默认仍为 classic。建议给
workspace 设置硬上限，使过大的计划在分配前失败：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=double_hoist \
POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE=4 \
POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB=4096 \
  ./run.sh --bootstrap-single-check
```

`[GPU memory]` 是 CUDA 上下文在几个同步检查点的驻留量，不是采样器意义上的
瞬时峰值。完整网络仍应另行使用 `nvidia-smi`/Nsight 检查峰值。

重点查看输出：

```text
gpu_only_preloaded_elapsed_seconds=...
gpu_only_elapsed_seconds=...
preloaded_replay_max_logit_error=...
preloaded_plain_pred=... gpu_pred=... max_logit_error=...
```

其中 `gpu_only_preloaded_elapsed_seconds` 是准备完成后的 GPU-only 墙钟时间。
最终 D2H、解密、预测检查以及第一次准备推理不在这个区间内。
最后一行是独立 CPU 明文 ResNet20 与密态 logits 的对照，也位于计时区间外；
它用于避免两次 GPU 回放虽然一致、但被同一个 KeySwitch 错误同时污染的情况。

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
| `POSEIDON_GPU_LINEAR_TRANSFORM_MODE` | `classic` | Bootstrap 线性变换路径：`classic` 或 `double_hoist` |
| `POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE` | `4` | double-hoist 一次保留的 baby rotation 数量 |
| `POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB` | `0` | double-hoist workspace 上限；`0` 表示不设限，建议测试时显式设置 |
| `POSEIDON_DOUBLE_HOIST_P9_DIGIT_BATCHED` | `0` | 对 N65536/P9 double-hoist ModUp/NTT 按 digit 合并 kernel 提交；当前为 V100 验证用 opt-in |
| `POSEIDON_GPU_MAX_POOL_MB` | `0` | RMM 内存池硬上限；`0` 表示使用全部可用上游容量 |
| `POSEIDON_TRACE_ROTATION_STEPS` | `0` | 按 Q 记录应用请求的逻辑 rotation step；不包含 Bootstrap 内部旋转 |
| `POSEIDON_LEVEL_AWARE_ROTATION_P` | `1` | 应用 rotation/relinearization 使用按层 P；设为 `0` 回退全局 P |
| `POSEIDON_APPLICATION_KEYSWITCH_P_MODE` | `single_digit` | 应用逐层 P 策略：`single_digit` 使用 P=Q，`fixed_dnum` 按目标 dnum 缩小 P |
| `POSEIDON_LEVEL_AWARE_BOOTSTRAP_P` | `1` | Bootstrap KeySwitch 使用按层 P；设为 `0` 回退全局 P |
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

当前卷积、Option-A shortcut、平均池化和全局池化使用 GPU 融合明文乘累加。
预热后的第二遍会直接复用 GPU 上的编码明文，不会在计时区间重复做 CKKS
编码或明文 H2D 上传。block 卷积会把 kernel 累加、BSGS giant 放置和 support
mask 都延后到所有输出组融合完成，再连续 rescale 两个物理 Q prime；随后
BSGS baby selector 再统一 rescale 一次。因此普通卷积由10次降至3次，两个
transition 卷积由18次降至3次，18个 block 卷积合计由196次降至54次，减少
72.4%。它仍然消耗3个物理 Q prime，输出 level 和原实现相同，不改变后续
Bootstrap 的预算。Global Pool 使用二叉归约和 4x4 BSGS 压紧，FC 使用单密文
BSGS，10 个 logits 位于同一密文的前10个 slot。

Stem 会把27个 im2col patch 各自复制到16个输出通道 page，用27次打包 PMult
和一次统一 rescale 同时计算所有输出通道；旧路径需要432次标量
PMult-rescale和15次输出旋转。卡 2 的预加载 Stem 时间从 `238 ms` 降到
`11 ms`。

卡2的延后 rescale 版本 GPU-only 结果为 `7.91179 s`，预测类别为 `3`，两遍
logits 完全一致（`preloaded_replay_max_logit_error=0`）。原版本在同卡上的
波动范围为 `7.89794--7.94257 s`，所以时间基本持平。完整逐层验证的最终
`max_logit_error=0.019248`，没有出现精度退化。

按层缩减应用旋转 P 基后的同二进制 A/B 结果为：全局 P18 路径
`7.96925 s`，默认按层 P3--P8 路径两次为 `7.56954 s` 和 `7.50207 s`，观察到
约 `5.0%--5.9%` 的端到端提升。后一轮的明文/密态预测均为 `3`，最终
`max_logit_error=0.0139039`，回放误差为 `0`。Q8/P8、Q7/P7 的 direct 与
hoisted 旋转检查最大槽误差约 `1.4e-4`；旧 P18 路径约为 `9.4e-7`，说明缩减
P 后数值噪声有所增加，但当前仍通过完整网络的既有误差门限。
同一低层检查也在全局 `Q36/P9/dnum=4` 配置下通过，实际打印的应用参数为
`Q8/P8/dnum1` 和 `Q7/P7/dnum1`；平方后的低层 relinearization+rescale 最大
槽误差约 `4.2e-8`，确认按层 P 选择同时覆盖 rotation/relinearization 且不依赖
全局 dnum=2。

Bootstrap 按层 P 的同密钥 A/B 检查结果如下：dnum=2 的按层输出相对全局 P
最大槽差为 `4.10977e-5`，dnum=3 为 `1.91482e-5`；dnum=4 没有需要缩减的
Bootstrap 层级，两次输出最大槽差为 `0`。包含 18 次 Bootstrap 的完整运行也
在 dnum=2 和 dnum=3 下通过：前者 GPU-only 为 `7.44308 s`、最大 logits 误差
`0.0147011`，后者为 `7.56478 s`、最大 logits 误差 `0.0165811`；两者预测均为
类别 `3`，回放误差均为 `0`。

2026-09-04 在空闲 Tesla V100-SXM2-32GB、`N=65536/Q36/P9/dnum4` 上进行的
单次 Bootstrap 基线为：classic `217.127 ms`、复数最大槽误差
`1.28040e-4`；double-hoist（baby tile 4、workspace 上限 4096 MiB）
`167.133 ms`、复数最大槽误差 `4.52746e-5`，相对 classic 下降约 `23.0%`。
两条路径初始化后的 CUDA 驻留显存快照均约 `12860.8 MiB`，尚有约
`19640.4 MiB` 可用。double-hoist 在同一把密钥上连续执行两次时，两个输出的
最大槽差为 `0`，输入最大误差为 `3.02912e-5`。这些是 V100 数据，不能作为
H100 性能结论；H100 必须使用 `CMAKE_CUDA_ARCHITECTURES=90` 重新编译和测量。

逐层应用旋转计划通过一次不预加载直连密钥的完整 dnum=2 trace 获得。该次运行
使用 Bootstrap 已有的幂次密钥组合 rotation，峰值显存采样约 `7.74 GiB`，最终
明文/密态预测均为类别 `3`。固化 139-key 计划后，dnum=4 的 Q8/Q7 direct、
hoisted rotation 与 relinearization 局部检查通过，最大 rotation 误差分别为
`8.38793e-5` 和 `1.4248e-4`，最大 relinearization 误差约 `4.56e-8`。随后完整
dnum=2 预加载推理通过，GPU-only 为 `7.40234 s`、最大 logits 误差
`0.0144873`、预测类别 `3`、回放误差 `0`。密钥生成在 GPU-only 区间外，因此
本优化的直接目标是降低初始化时间和显存，而不是改变算子延迟。

在同一张空闲 V100 32GB 上，使用 `POSEIDON_GPU_MAX_POOL_MB=24576` 执行
`Q36/P9/dnum4 --direct-infer 0 1` 通过。初始化期间显存采样约 `12.60 GiB`，
第一个 BasicBlock 的中间正确性结果为：stem 卷积 `7.862e-7`、第一层 conv1
`1.4119e-5`、conv2 `3.40709e-5`、block 输出 `4.39305e-5`。运行完成后显存
全部释放。该测试包含两次真实 Bootstrap，并证明 dnum=4、逐 Q 应用密钥和
受限内存池可以共同工作。随后 3-block 和 6-block 分阶段运行也均通过，完整
9-block classic 路径最终得到明文/密态预测类别 `3/3`、最大 logits 误差
`0.00804929`，总墙钟 `153.576 s`。这些墙钟都包含密钥生成、编码和初始化，
不是 GPU-only 性能结果。

相同 V100、Q36/P9/dnum4 和逐 Q 139-key 计划下，使用
`POSEIDON_GPU_MAX_POOL_MB=22528`、
`POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB=4096` 的 double-hoist 路径按
1/3/6/9 blocks 分阶段运行均通过，没有 OOM；初始化期间采样的驻留显存约
`11.58 GiB`，完整运行结束后显存全部释放。完整网络明文/密态预测类别为
`3/3`，最大 logits 误差 `0.0106437`。稳态单次 Bootstrap 从 classic 的约
`172-175 ms` 降至约 `111 ms`；按逐算子日志求和，网络阶段约从 `46.44 s`
降至 `45.25 s`（约 `2.6%`）。double-hoist 总墙钟为 `241.054 s`，高于
classic，是因为 direct 模式包含 QP 预旋转密钥生成；该总墙钟不能代表预加载
后的推理性能。两次完整测试使用独立随机加密/密钥，不能仅根据 `0.00805` 与
`0.01064` 的差异判断数值精度回退。

P9 digit-batched 原型保留 row-tiled BConv 算法，并把 Q34 的 `3x9+1x7`
decomposition 分成两个宽度组；每组使用一个三维 grid 完成各 digit 的 ModUp、
four-step phase1 和 phase2，Q decomposition limb 的复制融合进 phase1。它额外
保留约 `86 MiB` 的四-digit 系数域临时区。可用同一 runtime、同一密钥和同一
输入交替测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_GPU_MAX_POOL_MB=22528 \
POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB=4096 \
POSEIDON_GPU_RESNET20_DNUM=4 \
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=double_hoist \
POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --bootstrap-digit-batch-check 10
```

V100 上 10 轮 A/B 为 baseline `150.059 ms`、batched `148.757 ms`，加速
`1.00876x`；清理后 5 轮回归为 `150.596 ms` 对 `148.917 ms`，加速
`1.01128x`。两次测试中 batched 与 baseline 的最大槽差均为 `0`。收益很小，
说明每个 digit 已有足够并行度，因此在 H100 Nsight 验证前默认仍关闭。第一版
强制融合 BConv 与 phase1 的 all-digit 方案为 `179.018 ms`，相对同二进制
baseline `166.688 ms` 慢约 `7.4%`，实现已删除，仅保留此结果记录。

## 11. 最短运行流程

```bash
cd /home/guoshuai/github/poseidon/poseidon/benchmark/resnet20_gpu
unset POSEIDON_TRIDENT_RESNET20_ROOT
./run.sh --weights-check
CUDA_VISIBLE_DEVICES=2 ./run.sh --smoke
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep ./run.sh --bootstrap-check
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep ./run.sh --gpu-only 0
```
