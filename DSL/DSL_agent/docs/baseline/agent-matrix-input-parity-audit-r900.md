# 矩阵模型跨入口审计

24 个静态矩阵模型均通过隔离 PyTorch 捕获和独立参考比较。相同数学计算直接提交 JSON 图时，9 个通过接入，15 个被当前分解规则拒绝。差异属于模型接入范围；本批没有编译或密态执行。

## 接入结果

JSON 使用 §poseidon-operator-graph-v1§；PyTorch 使用 §poseidon-trusted-pytorch-v7§。二者选用 §general-tensors-v1§ 和相同输入形状、公开权重、输出及参考。接入通过表示完成类型、形状、分解及绑定检查，后续编译和数值验收另计。

| 上下文 | 模型数 | 直接 JSON 接入 | PyTorch 规范化后接入 |
|---|---:|---:|---:|
| 双密文二维矩阵 | 3 | 3/3 | 3/3 |
| 双密文三维批次 | 3 | 3/3 | 3/3 |
| 批次广播 | 3 | 3/3 | 3/3 |
| 高阶批次，rank 4／8／32 | 3 | 0/3 | 3/3 |
| 左向量 | 3 | 0/3 | 3/3 |
| 右向量 | 3 | 0/3 | 3/3 |
| 右侧公开矩阵／向量 | 3 | 0/3 | 3/3 |
| 左侧公开矩阵／向量 | 3 | 0/3 | 3/3 |

公开权重采用 NPY 状态包，包含 6 个非重复权重数组。高阶例使用 rank 4、8、32 的静态批次；本集每例最多 32 个输入元素、16 个输出元素。它是补充接入对照集，不增加原冻结 Benchmark 的 1,200 个模型计数。

## 已确认

- 8 个边界测试 passed，failed／errors／skipped 均为 0。
- 24/24 隔离捕获通过；原始 nn.Module 和既有双参考检查通过。
- 同一批 16 组输入，声明式模型、规范化模型和分解图共 384 组独立比较通过；最大绝对差 `2.220446049250313e-16`，明文比较容差为 `1e-12 + 1e-12*abs(reference)`。
- 78 项完整性拒绝通过：72 项矩阵来源记录变造、6 项权重内容变造。
- 80 个来源／清单／模型／NPY 文件可按字节复现，两次生成一致。
- 运行环境为 ARM Ubuntu、Python 3.10.14、NumPy 1.25.2、Torch 2.0.1，CUDA／HIP 为 None。
- 待验收 r892 的 674 个运行源码、517 个绑定文件和 4 个驱动文件保持。648 个后端源码、6 个参考文件及受保护历史记录核验通过。
- 编译、Poseidon 密态执行、解密和新 Agent 生成均为 0；API、安装、下载、SDK 重建、参数及数值门限变更、Git 发布均为 0。

审计绑定：`0794b8365d231d688402e227767d00dd4a4b2c56db2be54584c59bd5b93d4d7f`。数据与工具使用新审计绑定，原 r898 记录保留，不重复相加测试数。

## 缺口与修改方向

直接 JSON 的 `matmul` 规则限制为 rank 2/3 的双密文输入；公开右操作数被类型检查拒绝，公开左操作数还受首操作数必须加密的规则限制。PyTorch v7 在输入层已完成向量升维、批次广播、公开权重变换及结果形状还原，因此对应 24 例都能准备现有核心图。

下一步在 Agent 层共用现有矩阵规划器，以显式版本或策略开放 JSON 的相同规范化范围，保留原接口与请求规则。范围包括 encrypted/plain 类型识别、来源映射、公共权重绑定、乘积张量预算和输出还原；继续复用已有 Linear、广播、乘法及归约。实施后验收非法形状、rank／容量边界和来源变造，再补受影响的真实后端与单独批准的 Agent 测试。

待 r892 的十二项密态窗口结束或撤回后再改共用运行源码。本批补充工具和案例独立保存在开发目录，r892 仍待批准。

## 复现

在已配置的项目 Python 环境中重建对照集：

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/baseline/benchmarks/tools/generate_matrix_entry_sources.py \
  --output "$results/matrix-entry-fixtures"
~~~

执行开发审计，输出目录须为新目录：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/audit_matrix_input_equivalence.py \
  --cases scripts/baseline/cases/matrix-entry-equivalence-v1 \
  --output "$results/matrix-entry-audit" --seconds 600
~~~

工具沿用锁定的 ARM Python 和独立 bubblewrap 捕获。明文数据不进入模型生成提示，编译及密态门限仍为 `1e-5 + 1e-4*abs(reference)`。

[机器报告](agent-matrix-input-parity-audit-r900.json) · [来源集](../../scripts/baseline/cases/matrix-entry-equivalence-v1/) · [当前矩阵输入](operator-decomposition-and-python.md#pytorch-矩阵运算) · [待验收容量排程](agent-capacity-neutral-offline-proposal-r892.md)
