# ResNet20 支持代码来源

这些文件用于让 S2C-first 应用独立构建，不用于恢复旧应用调度器。

`resnet20_weights.cpp/.h`、`gpu_multiplexed_tensor.cpp/.h` 和 `data/` 来自
`mingerfan/poseidon` 的 `lfy-application-improve`，固定提交
`6a71bda7aaeabccb6f8a52b5ab20ea53b685092b`。除 tensor 头文件移除旧 runtime
include 并要求经过本地 adapter 引入外，导入文件保持原样。数据来自该分支已有的
Trident ResNet20 资产；不据此新增或推定第三方资产许可，公开发布前仍需核对许可。

`secure_sparse_secret.h` 来自本地 `resnet20-9.3` 的未提交成果，按原内容收录。
原工程、未提交补丁和文件清单均保存在
`/home/liufuyao/Work/poseidon_gpu_other/GitHub/archive-20261012/local-snapshots/`。
采样器测试使用确定性随机数；实际加密仍使用原调用处的密码学随机源。

本次没有导入 `gpu_ckks_runtime.cpp`、旧 inference、旧 ReLU 调度或旧 main。
采样器不等于参数安全性证明；Q50/P25 配置继续要求显式 `--unsafe-*`。
