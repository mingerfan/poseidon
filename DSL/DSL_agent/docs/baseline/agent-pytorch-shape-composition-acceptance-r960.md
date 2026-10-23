# PyTorch 算术与静态形状接入审查

当前生产入口有一个共同缺口：Tensor 算术方法的结果缺少静态 shape 元数据。最小修补原型已验证，正式入口尚待接入。

| 验收 | 结果 |
|---|---:|
| 开发工具测试 | 8 passed，0 failed/errors/skipped |
| 合法 PyTorch 上下文 | 24 接入通过，9 接入失败 |
| 拒绝测试 | 3/3 符合预期 |
| 等价 JSON 独立参考 | 33 上下文 × 16 = 528 组 |
| 已接入模块与等价图比较 | 24 × 16 = 384 组 |
| 修补原型独立沙箱 | 9/9，144 组参考，最大误差 0 |

33 个合法任务覆盖加、减、乘广播后 reshape，包含三种写法和不同形状查询，是 3 个数学计算组。写法变体按入口任务计数；原 1,200 模型集保持。

## 原失败与原因

下面每种算术方法分别覆盖 `y.shape`、`y.size(0)`、`y.numel()`，共 9 项失败：

~~~python
y = x.add(public)   # sub、mul 同样受影响
return y.reshape(y.numel())
~~~

等价的 `x + public`、`torch.add(x, public)` 以及使用输入签名查询或固定 reshape 的写法通过。失败位于 `pytorch_static_trace.py` 的第一次 metadata tracing：方法尚未登记 shape 规则，后续 shape 查询立即拒绝，规范化尚未开始。

这 9 项属于 Agent 接入层缺口；编译器、Runtime 和密态数值均尚未运行。受控拒绝包括加密数据相关分支、原地加法和 `out=` 修改。

## 已准备的最小修补

开发目录内的 `metadata-policy.patch` 为 Agent 自有静态 tracer 增加显式 `signature-static-arithmetic-shape-v1` 策略。只登记 add/sub/subtract/mul/multiply 的 metadata 运算与公开索引兼容；原策略保留。`git apply --check` 通过。

修补在隔离开发副本中使原 9 个模型通过 tracing、规范化、数学／Torch／原模块参考比较；旧 index 策略仍拒绝。没有应用到生产源码，没有修改 Poseidon、Runtime、Dacapo、参数、模型或 reference。

生产接入下一步：新增显式可信输入版本，登记 metadata 策略和实际实现身份；复用现有隔离、规范化、绑定与公开 CLI。补策略、来源拒绝、旧格式兼容及原 9 个来源回归。原失败和历史绑定保留。

## 绑定与执行状态

- ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- r950 的 681 份运行源码、1,367 份案例／证据和 4 份驱动保持；648 后端文件、6 reference 文件及历史报告保持。
- 本批只增加开发工具、来源案例、修补原型和文档；Git diff check 通过。
- 新增证据约 0.95 MiB。API、安装、下载、SDK 重建、编译、Hecate tracing、密态执行、解密和 Git 发布均 0。
- 当前没有验收 worker。94 项冻结后端批次仍待批准，收据和后端目录均未创建。批准后使用 r950 已有入口；修补的生产接入安排在该批安全边界后。
- 本批不改变 bootstrap、密文分区、CUDA、原八个编译失败、真实 Agent 新功能、应用精度、容量或 ResNet20 状态。

审计工具首次未识别原地修改的准确错误文本；原 35 份捕获保留，复核只追加尚未运行的第 36 项。原型首次在模型捕获前因开发脚本调用了未绑定的环境辅助模块停止，原日志和源码保留；纠正该开发脚本后完成独立验收。两项工具错误与模型的 9 项接入失败分开记录。

[机器报告](agent-pytorch-shape-composition-acceptance-r960.json) · [输入用法](operator-decomposition-and-python.md#公开量与张量算术) · [待批准后端批次](agent-arithmetic-matrix-offline-proposal-r950.md)。
