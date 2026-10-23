# 广播与 packing：当前真实 Agent 验收

原始可信本地 PyTorch 的 wide_tail_linear 已在 explicit-v19 下通过，一次自动修复后成功。

模型包含广播、1,025→3 Linear、按行均值和共享命名输出，共 2,052 个输入元素、3,078 个固定参数。原源码、模型、权重、布局、reference 和后端参数保持原样；生成程序来自当前真实 Agent，未人工替换。

## 实际结果

| 项目 | 结果 |
|---|---|
| 真实生成／HTTP | 2／2，全部结算，unknown 0 |
| 首轮 | 编译与密态执行完成，数值失败 |
| 自动修复后 | 数值通过 |
| 编译／密态调用 | 2／4，包含首轮数值失败 |
| 通过范围 | 4 组基础输入、3 组新输入、1 组重复输入，共 112 个输出值 |
| 最大绝对误差 | 1.521419468e-7 |
| 原始来源、程序包重载、新输入、重复执行、缓存 | 通过 |
| 冻结数值门限 | 1e-5 + 1e-4 × abs(reference) |

独立审计重新读取原始响应、实际编译产物、解密数组和来源记录；两套明文 reference 使用 1e-12 误差界。初次后置审计因缺少显式启动参数而停止，该记录保留；随后补做只读审计，新增 API、编译和密态调用均为 0。

SDK：eef0f250a457a24aa658c7921e6bc62e0387a9c7c8ea29f9f7cf47353354ad41。
审计绑定：f498376899b129815150e72da1c8be98b96371abe461a08d87acb0642ee8e9e4。

## 通用算子下一步：Tensor.gather

当前公开索引入口对应 index_select。已建立按元素 Tensor.gather 的独立草稿：静态公开 int64 索引规范化为已有 reshape／public_gather／reshape。它区分每行索引、重复索引、较小的非索引轴和负轴表示；原负索引仍拒绝。

27 项草稿检查通过，含 6 个固定 Torch 2.0.1、独立数学 reference 与现有核心图分解的差分上下文。覆盖向量、逐行、中轴、部分行、末轴和 rank32，并验证 dtype、越界、深度、循环数据及展开预算拒绝。

草稿保存在 scripts/baseline/benchmarks/tools/tensor_gather_component_v1。生产源码捕获尚未接入，尚未编译、密态执行或真实 Agent 生成，不能计为新算子已支持。

下一步接入版本化 PyTorch 捕获与来源校验，再验证三个不同上下文的 Poseidon 和真实 Agent 路径。继续复用已有分解与后端接口。

## 范围

本批仅证明所列大输入模型及固定测试输入的数值一致。算子和未见组合、容量、原模型应用精度、bootstrap、CUDA 实机和 ResNet20 继续推进，分别验收。

[机器可读结果](agent-binding-frames-acceptance-r1574.json) · [前一阶段](agent-binding-frames-progress-r1572.md)
