# LayerNorm 源码校准

支持可信本地 PyTorch 的 nn.LayerNorm 和 torch.nn.functional.layer_norm。
准备请求使用 request-v2，选择原始 FX 节点及模块路径；模块的 normalized_shape、eps 与图哈希一起绑定。
函数写法的 normalized_shape 和 eps 使用静态字面量。

## 配置

LayerNorm 的倒平方根近似作用于方差加 eps。观察节点输入时使用：

    "projection": {"kind": "layer_norm_variance"}

该投影按 normalized_shape 指定的末尾轴计算总体方差（unbiased=False），再加原始 eps。
统计保持输入的 float32/float64 类型，原始 forward、权重和输出结构保持。
节点、模块路径及默认配置沿用既有优先级；LayerNorm 选择 function=rsqrt。

命令入口沿用：

    python3 scripts/agent_calibration.py inspect --request inspect.json --output /absolute/new-inventory --trust-local-code
    python3 scripts/agent_calibration.py observe --request observe.json --output /absolute/new-observations --trust-local-code
    python3 scripts/agent_calibration.py freeze --request freeze.json --output /absolute/new-freeze --trust-local-code

freeze 保存显式配置及已观察范围，随后使用配置后的 manifest 提交正常模型生成请求。
本批固定 Newton 倒平方根近似，区间 [0.0001,4]、2 次迭代；原函数精度另行验收。

## 本批结果

| 项目 | 结果 |
|---|---|
| 相关测试 | 145 passed，failed/error/skipped=0 |
| 原始源码、校准与双明文参考 | 6/6：模块/函数各3个上下文 |
| 规则候选完整执行 | 1/3通过，2项编译 scale 750 > 720 |
| 人工等价表达 | 原失败2项均通过真实密态和包复用 |
| 代表模型程序包 | 3包通过新输入及重复执行 |
| 真实 Agent | r1668 独立批次；状态查看其实际报告 |

模块和函数上下文覆盖 shape [2]、[2,2]、[1,2,2]；本批 LayerNorm 使用 elementwise_affine=False。
“输入仿射”上下文指进入 LayerNorm 前的缩放、平移，并非新增学习增益参数验收。
规则 scale 失败保留，采用 v10 布局及移除冗余掩码仍有相同失败。
开发探针仅合并相邻、可精确表示的二进制公开系数，模型、原参考、近似配置与后端保持；其程序单独归为人工等价表达。

成功代表集首次比较12组40值，max absolute error 1.3254385145e-8。
包复用比较12组40值，max absolute error 3.2724299625e-7。
门限为 abs(actual-reference) <= 1e-5+1e-4*abs(reference)。

离线全过程10次编译尝试（4次成功）、12次密态调用，API=0。
上述计数包含原失败和重复验证，不计为新增模型数。
原函数应用精度尚未认证；本批仅验证已冻结近似图及其 CKKS 执行。

## 记录

执行目录为 ARM work root 下 agent-layernorm-calibration-r1662、
agent-layernorm-calibration-poseidon-r1663/r1664/r1665 和 agent-layernorm-calibration-manual-r1666。
审计为 scripts/baseline/benchmarks/tools/layernorm_calibration_v1/audit-r1667-r2.json。
原审计中混合规则/人工来源的聚合标签已另建更正记录，原始运行数据保留。

SDK：316b4fcb795526fb96402050de45d6edadd294f76a3a7a25de3bf98d59a16084
审计绑定：d53654a07cd4cca4a54cb5ad863a23eacccfef9d88be6822e86493ec9370cd31

本批修改 Agent 自有源码准备、规则候选调度及测试。Poseidon、Runtime、Dacapo、参数、数值门限、网络和 x86 配置保持；安装、下载、提交、推送、PR 均为0。
