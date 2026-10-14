# JSON 矩阵输入接入

当前提供显式 `poseidon-operator-graph-v2`，将静态向量、矩阵、多维批次广播及公开矩阵两侧乘法统一到既有分解和布局。标量矩阵输出使用一个元素的张量。旧 v1、PyTorch v7 和已有请求规则保持。

## 实现

Agent 新增 `operator_graph.py`，复用现有 PyTorch 矩阵规划器；生成模型接收规范化的计算图、公开权重及规则。程序包保留原图、分解图、节点来源和实现哈希。输入域检查按新格式处理标量后续组合。

本批保留原版本实现与捕获绑定，修改导入、能力判断、提交、来源核验及输入域分派。Poseidon、Runtime、Dacapo、依赖、参数、六份 reference／检查器源文件保持原哈希。

## 当前结果

| 检查 | 结果 |
|---|---|
| 单元、组件、参考与兼容测试 | 206 passed，0 failed/errors/skipped |
| JSON v2 公开导入与应用提交 | 36/36 |
| 独立参考输入组 | 576，复核同一输入不重复计数 |
| 旧请求与规则源码逐字兼容 | 12/12 |
| 旧 PyTorch 捕获源码身份 | 17/17 保持 |
| 新模型真实编译／密态／Agent 生成 | 尚未运行 |

36 例分十二个上下文组，每组三例：矩阵乘、批量密文、批次广播、高 rank、向量、公开右侧、公开左侧、公开批量权重，以及标量链、公开点积、共享矩阵链和显式近似域链。[模型清单](../../scripts/baseline/cases/operator-matrix-v2/catalog.json)。

早期工具将 `candidate_source` 的返回二元组误当源码，已在新工具中正确拆包。新格式的近似域组合错误经测试复现后修正，保留原失败日志。原八个编译失败还没有新密态结果，继续保留。

## 使用

~~~bash
python3 -B scripts/prepare_model.py \
  --graph scripts/baseline/cases/operator-matrix-v2/public_left_0.json \
  --resource-profile general-tensors-v1 \
  --write --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/json-matrix-import"
~~~

在已有固定环境中导入，随后使用 `app.submit_import(imported_directory, options=...)` 和现有生成、验收、程序包接口。容量图默认选择 v9 公共路由；显式配置继续参与绑定。

## 下一批

拟将原十二项与新增三十六项合为 48 项离线后端验收，检查真实 tracing、编译、RuntimePlan、Poseidon CPU、逐项数值和六份程序包复用。原未执行 r892 提案保留，但其共用源码绑定已过期；新提案按当前源码单独授权。门限保持 `1e-5 + 1e-4*abs(reference)`。

[机器报告](agent-json-matrix-static-acceptance-r913.json) · [输入用法](operator-decomposition-and-python.md#json-矩阵计算图)。
