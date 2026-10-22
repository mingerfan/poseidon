# ResNet50 模型级 Python DSL

前端入口为 `ResNet50.py`，必要的布局辅助源码为同目录 `imagenet_layout.py`。参考当前 framework 的 MLP 写法，还原原非 DSL ResNet50 的计算语义，由编译器安排自举、rescale 和 level，采用 Native NoHoistBsgs。

## 验收结果

2026-10-22，CPU 语义对齐、编译后整网明文对照、整网双卡 GPU 验收及独立核验均通过。

| 项目 | 结果 |
|---|---:|
| 原非 DSL CPU 对照最大差异 | 2.1316282072803006e-14 |
| packed/dense CPU 最大差异 | 6.217248937900877e-15 |
| 编译后整网明文差异 | 0 |
| GPU 输出最大绝对误差 | 0.004135181894192375 |
| GPU 输出 RMS 误差 | 0.0007792449202023056 |
| 自举次数 | 23892 |
| 预测 / 标签 | 65 / 65 |

GPU 通过配置必须保留 **`CUDA_LAUNCH_BLOCKING=1`**。原多项式、Native NoHoist 和验收门槛未变（abs 0.1、rel 0.006、imag 0.01）。本结果覆盖 image0，不代表整数据集准确率；默认异步双卡执行曾失败，具体异步根因尚未修复。历史失败报告保留在 `evidence/`，不能与最终通过配置混用。

整网双卡每卡使用 31744 MiB 显存池。单卡整网容量估算约 36.84 GiB，未在 32 GiB 卡上运行。

## 源码与运行要求

交接时须同时提供 `ResNet50.py` 和 `imagenet_layout.py`。环境需要 NumPy、PyTorch、当前 framework 的 Hecate 和原模型权重、d13 系数；本目录不包含权重、大型中间表示或二进制。

模型保留 267 个权重、boundary120、Bottleneck 3/4/6/3、原 d13、多项式最大池化及四轮 sign refinement、残差投影和 1000 类 FC。全局平均池化采用先乘倒数再累加的等价形式。输入为8个密文包，每包32768槽，使用 `pack_input` 打包。

在 Hecate Python 环境导出前端（输出目录须尚不存在）：

```sh
python ResNet50.py --weights /path/to/original/weights --relu-coefficients /path/to/original/d13.txt --output /path/to/new/output
```

使用 `compile_model.py` 接入现有 framework 编译入口，设备数设为2。`profiles/` 保存此次验收所用的 compiler、OperatorSpec 和 Native 参数。

GPU 运行前设置：

```sh
export CUDA_VISIBLE_DEVICES=<two-healthy-idle-gpu-ids>
export CUDA_LAUNCH_BLOCKING=1
export POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=31744
export POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary
```

正常 GPU 入口参数和已验收计划/程序 SHA 见 `evidence/gpu-full.provenance.json`。其中绝对路径是验收服务器上的产物位置，其他环境需替换路径；源码可重新导出、编译。`check_semantics.py` 提供独立 CPU 对照入口，原权重和系数须与语义报告来源记录一致。

## 验收证据

- `evidence/frontend-contract.json`：两份前端源码 SHA。
- `evidence/frontend-semantics.json`：完整逐层 CPU 语义对照。
- `evidence/clear-full.json`：编译后完整明文对照。
- `evidence/gpu-full.json`：整网 GPU 结果与精度门槛。
- `evidence/gpu-full.provenance.json`：计划、二进制、参数 SHA 及运行环境。
- `evidence/final-audit.json`：最终独立核验 PASS。

其余报告用于保留历史排查依据。ResNet18、ResNet20 不在此次变更范围内。
