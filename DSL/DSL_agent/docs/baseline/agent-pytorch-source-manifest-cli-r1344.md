# PyTorch 来源配置工具

已完成 `scripts/prepare_pytorch_source.py`。可以从本地模型文件、静态输入、
输出名称、工厂参数和 NPY 权重生成现有来源清单，随后进入原有应用或 CLI。
默认 v18，可选择 v12–v18；SDK 与冻结请求保持原身份。

## 使用

~~~bash
python3 -B scripts/prepare_pytorch_source.py ./model.py \
  --entry model.make --input x:1,4 --output logits \
  --manifest ./source.json
~~~

`model.make` 返回 `nn.Module`。公开权重按 `--weight linear.weight=weight.npy` 声明；
helper、工厂参数、float32 转换与近似配置见
[来源清单用法](operator-decomposition-and-python.md#pytorch-来源清单)。
工具读取配置、AST 和哈希，模型在后续明确选择的隔离导入中捕获。

## 本轮验收

| 范围 | Passed | Failed / Error / Skipped |
|---|---:|---:|
| 配置、数据边界、哈希、拒绝测试 | 59 | 0 / 0 / 0 |
| NPY 与固定 NumPy 解码交叉检查 | 20 | 0 / 0 / 0 |
| 隔离捕获与独立参考 | 3 | 0 / 0 / 0 |

三个来源是矩阵 einsum、NPY 公开权重 einsum、chunk/unbind 多输出组合。
自动生成的逻辑清单和捕获后的计算请求均与原请求一致。
NPY 覆盖五种 dtype、两个文件版本以及标量/二维形状。

首轮控制检查有一项失败和一项错误，原因是工厂参数在详细校验之前被 JSON
序列化；调整校验顺序后重验 59 项通过。首轮日志及绑定保留。
没有更改拒绝规则来接受 callable 或非有限参数。

本轮执行 Python 3.10.14、NumPy 1.25.2、ARM Torch 2.0.1 CPU。
SDK身份 `ee05605763e8f26e106c8c48167afb7017cee7c6fa946093626dc9c35bf29995` 保持。
654 项后端/参考文件、两份历史请求和 r1341 方案字节一致。
本轮没有编译、密态执行、解密、Provider、安装、下载、参数调整或提交推送。
现有 x86 配置保留，本轮实际验收平台是 ARM Ubuntu。

## 当前边界

源码仍为最多 8 个同目录平铺 Python 文件、合计 64 KiB。
输入及输出各 1–4 个，通用配置 rank 1–32；元素、节点、权重和物理资源沿用
现有能力检查。生成清单不会扩大算子或编译容量。

r1341 八例真实 Agent 方案尚未批准或启动；六例拆分和两例 MHA 的新来源准备
保持为离线证据。Bootstrap、CUDA、应用精度、密文交接及整网目标继续单独推进。

机器记录：[r1344 JSON](agent-pytorch-source-manifest-cli-r1344.json)。
原始记录位于 Ubuntu `scripts/baseline/benchmarks/tools/source_manifest_v1`，
来源验收位于 ARM work root 的 `results/agent-source-manifest-acceptance-v1`。
