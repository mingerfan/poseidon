# ResNet20 GPU运行说明

当前默认是经过非零密钥验证的22阶自举。完整参数、59/30/22对比、显存限制及复现命令统一见 [README.md](README.md)。

```bash
cd benchmark/resnet20_gpu
./run.sh --topology-check
./run.sh --weights-check
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1 ./run.sh --infer 0
CUDA_VISIBLE_DEVICES=0 ./run.sh --gpu-only 0
```

对比三种阶数：

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1 ./run.sh --bootstrap-suite 10
```

切换为30阶：

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE=30 ./run.sh --infer 0
```

要求CMake >=3.26.4与CUDA工具链。必要时设置CMAKE_BIN及CUDACXX。请自行选择空闲GPU；不要把CPU验证总耗时当作GPU-only耗时。旧H0/短链约9秒结果不适用于当前配置。
