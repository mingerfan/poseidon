# 公开张量除法源码接入草稿

接入原始 PyTorch 源码中的公开张量除法，将除数绑定为公开权重，确定性转换为倒数乘法。
原始模型、公开状态、追踪图和派生倒数分别绑定。计算使用 float64。原始 float32 权重通过既有 convert-float64-v1 策略显式转换，原始字节与转换后的计算值分别绑定；公开索引、布尔掩码和计数器保留原始类型。

当前 58 项静态与历史捕获元数据检查通过，failed/errors/skipped 均为 0。
6 项历史来源记录经过文件哈希与外部报告绑定核对，仅用于策略兼容检查。
18 个原始源码上下文待隔离捕获：6 个基础组合、6 个混合公开类型组合、3 个显式近似组合，以及 float32 向量、BatchNorm、GELU→Linear 3 个 dtype 组合。
生产入口保持 v21；本草稿尚未激活。

近似配置沿用 `pytorch_approximation.py` 的节点／模块／默认规则。
现有独立参考分别比较原函数、显式近似和分解结果，并单独记录近似误差；本草稿复用此验证方法。
float32 策略的参考使用明确转换后的 float64 数学计算，原始 float32 执行精度另行认证。应用精度认证和 CKKS 验收分别保留；当前检查为静态检查与历史数据核对。

## 验证

在 ARM Ubuntu checkout 中，等待原批次释放 Nix 启动锁后运行：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_tensor_source_v2/build_fixtures.py
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_tensor_source_v2/controls.py
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_tensor_source_v2/test_controls.py
```

使用现有固定 Python、Torch、NumPy、Nix 闭包和 bubblewrap。
来源验收预算保持 480 秒／128 MiB，源码沙箱保持 4 GiB／150 CPU 秒／180 秒墙钟。
源码和权重只读，网络隔离，产物使用新的独立目录。此阶段 Provider、编译与密态调用均为 0。

## 记录

- `origin.json`：移植来源和参考语义复核。
- `before.json`：生产和前版草稿的原始文件哈希。
- `controls-static-v4.json`：当前 58 项检查。
- `controls-static-v1/v2/v3.json`：此前 29／35／38 项历史检查。
- `agent-public-tensor-source-draft-v2`：预定新的来源产物目录。
