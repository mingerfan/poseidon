# 结构化输入 Agent 验收

2026-10-11 · ARM Ubuntu · PyTorch v26 · Poseidon CPU

12 个上下文全部通过，11 个首次通过，1 个经一轮自动修复通过。每个程序包完成编译、四组冻结输入比较、三组新输入、一次重复执行，以及缓存复用。

| 输入写法 | 上下文 | 首次通过 | 修复后通过 | 最终失败 |
|---|---:|---:|---:|---:|
| 公开默认参数 | 3 | 3 | 0 | 0 |
| 字符串键字典 | 3 | 3 | 0 | 0 |
| 关键字参数 | 3 | 2 | 1 | 0 |
| 嵌套 tuple/list | 3 | 3 | 0 | 0 |

## 执行结果

- 13 次生成、13 次 HTTP；13 次候选编译、37 次 Poseidon CPU 密态调用。
- 最终程序比较 96 组次、672 个输出值；最大绝对误差 `1.0562915032930054e-06`，MAE `9.084953728191275e-09`。
- 非零 reference 最大相对误差 `1.2751334523832016e-05`；有定义时最小 cosine `0.9999999999944174`。
- 门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。重复输入另计组次。
- 12 份程序包均完成重载、新输入执行、结构化调用和无生成调用的缓存复用。
- 38 项执行控制检查通过，失败／跳过 0；最终只读审计核验 1,264 个证据文件。

## 自动修复

`keywords1` 为二维矩阵乘法上下文。首次候选的旋转、掩码和乘法组合与目标计算不一致；编译和密态执行成功，数值检查失败，最大绝对误差 `0.5644424260678192`。组件将受控反馈交给同一个生成模型，一轮修复后通过，冻结四组输入的最大绝对误差 `5.853348321950594e-9`。

原始响应、首轮候选、失败产物、诊断和后续成功候选均保留。人工候选替换为 0。

## 使用范围

通过 `--source-format poseidon-trusted-pytorch-v26 --call ./call.json` 选择结构化来源接口。Tensor 叶子数 1–4、调用树深度 8、节点数 256、调用 JSON 8 KiB；计算图容量沿用所选资源配置。默认 v18、latest=v21 保持。

本批覆盖上述 12 个静态上下文。更大模型、任意 Python、bootstrap 和 CUDA 由各自验收记录管理。

## 记录与绑定

- Checkout：`/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- 报告绑定：`36e13c0da7ed7e801dd9c2efb196651b0a7da8eaca54b393cb736044ccebad2e`。
- SDK：`b9c4890b3569a998bd7d30f8252469a29d448ed7b200e834a2d9c89e60013b10`。
- 原 r1392 方案、授权和调用账本保留；本批已结算。剩余额度仅属于原范围。
- 新产物约 `194.4` MiB；全局约 `29.49` GiB，低于 32 GiB。
- 后端库、依赖、参数、模型、reference、门限、x86 配置和网络保持；安装／下载／commit／push 为 0。
- 审计工具读回 JSON 时修正了输出键顺序和 tuple 数组化比对；两次失败日志保留。该修正发生在开发审计工具中，新增 API／编译／密态调用均为 0。

[机器记录](agent-structured-input-live-acceptance-r1393.json) · [批准范围](agent-structured-input-live-proposal-r1392.md) · [输入用法](operator-decomposition-and-python.md#pytorch-v26-结构化输入)

只读复核：

~~~bash
PYTHONPATH=scripts/baseline python3 -B scripts/baseline/benchmarks/tools/structured_input_live_acceptance_v1/verify_completed.py
~~~
