# PyTorch 结构化输入接入

ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。本批完成 Agent 输入层适配；新增编译、密态与 API 调用均为 0。

## 范围

显式 `poseidon-trusted-pytorch-v26` 在 v25 上增加 `poseidon-pytorch-call-v1`。支持 Tensor、静态 tuple/list、字符串键字典的输入组合，位置参数、仅关键字参数及公开默认值。逻辑与物理接口继续使用声明顺序的命名 Tensor。输出保留字典、tuple/list 和 None。

最多 4 个 Tensor 叶子，每个仅绑定一次；调用树深度 8、256 节点、8 KiB。原元素、算子、rank、节点、权重、密文与深度上限保持。可变参数、加密数据条件、错误签名、重复／缺失输入和运行时修改冻结公开参数均拒绝。

CLI 增加 `--call`；应用重用程序包时可选择 `execute(..., sample_format="forward_call")`。原命名样本接口保持。

## 检查结果

| 检查 | passed | failed | skipped |
|---|---:|---:|---:|
| 单元与组件兼容 | 216 | 0 | 0 |
| 真实隔离来源捕获 | 12 | 0 | 0 |
| 应用来源接入 | 6 | 0 | 0 |
| 规则候选静态检查 | 15 | 0 | 0 |

12 个来源模型覆盖字典、嵌套容器、关键字、公开默认值各 3 个上下文。两套独立参考共 192 组／1,344 值；分解最大差异 `1.1102230246251565e-16`，原始与归一化 Torch 差异 0。6 个应用请求准备并取消成功，其中 3 个旧请求和哈希逐字节保持。48 项为新增检查。

新编译、密态执行、解密和新 Agent 生成均未运行。这些检查不能计为 Poseidon 或真实 Agent 通过。

## 修复

1. 将 FX 占位符按声明顺序绑定至命名 Tensor，保持不同形状输入及来源映射。
2. 校验并记录公开参数的精确值及 FX 专门化断言，移除仅与冻结参数对应的检查节点，数值节点保持。
3. 针对固定 Torch 2.0.1 嵌套默认值产生无效签名的问题，在 Agent 输入适配中记录原默认值并显式规范化占位符签名；Torch 和后端保持。
4. 修复公开状态检查遗漏 call_module 子模块参数的问题；真实 Linear 权重参与来源及完整性绑定。
5. 程序包执行按原 forward 签名绑定结构化样本，校验冻结公开值后进入原命名输入执行器。

旧失败日志和原绑定保留。SDK `b9c4890b3569a998bd7d30f8252469a29d448ed7b200e834a2d9c89e60013b10`；Poseidon、Runtime、Dacapo、依赖锁、参数、x86 和网络保持；无 commit／push。

## 下一批

[r1390 离线方案](agent-input-binding-offline-proposal-r1390.md)：12 个新上下文加 3 个旧版本兼容，最多 15 编译／24 密态调用。检查包重载、结构化新输入、重复执行及命名输入一致性。方案尚待批准。

[机器记录](agent-input-binding-component-checkpoint-r1389.json) · [用法](operator-decomposition-and-python.md#pytorch-v26-结构化输入预览)
