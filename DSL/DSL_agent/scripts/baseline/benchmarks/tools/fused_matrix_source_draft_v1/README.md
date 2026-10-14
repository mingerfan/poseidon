# 融合矩阵来源接入草案

在可信 PyTorch 来源入口中规范化 `addmm`、`baddbmm`、`addbmm`。
来源契约为 `poseidon-trusted-pytorch-v14`，代码在独立覆盖层中验证。
应用生产入口仍使用现有契约。

## 范围

| 写法 | 形状 |
|---|---|
| addmm | 两个二维矩阵 |
| baddbmm | 批数相同的两个三维张量 |
| addbmm | 三维批量矩阵相乘，再沿批轴求和 |

支持函数、Tensor 方法和关键字调用、结果形状内的偏置广播、公开有限
alpha/beta、公开矩阵权重与密文矩阵组合。采用现有 mm/bmm、sum、
multiply、add 规范化。来源保留原始 FX、规范化 FX、参数、步骤和权重绑定，
逐次写入按已有别名契约处理。

输入为固定形状的 float64 张量，公开系数绝对值至多 1,024。
beta=0 按 Torch 语义忽略偏置值。out 参数、融合运算自身的原地写法、
动态形状和超资源输入返回诊断。

## 来源与组件

来源包由显式 Python 文件清单、工厂参数、输入输出签名和无 pickle 权重组成。
来源 Python 在隔离捕获进程内执行。重载校验使用数据化 FX 记录，
核对步骤、拓扑顺序、参数、类型、shape、权重及后续规范化记录。
生成请求准备和提交沿用现有应用接口。

V12/V13 保持原有行为，新增写法显式使用 V14。

## 验证

现有测试分为来源捕获与独立明文比较、来源包重载、提交/取消接口、
数据化记录正反例，以及旧版本兼容与拒绝检查。

在已配置的 ARM Ubuntu checkout 中，以新结果名运行：

```sh
python3 -B scripts/baseline/benchmarks/tools/fused_matrix_source_draft_v1/run_isolated.py agent-fused-matrix-intake-rNEW source
python3 -B scripts/baseline/benchmarks/tools/fused_matrix_source_draft_v1/run_isolated.py agent-fused-matrix-intake-rNEW affected
```

将 `NEW` 替换为尚未使用的数字；两次运行分别使用新目录。
每轮保留源码快照、运行版本、隔离与资源记录。
`host_transport_checks.py` 检查实际来源记录，可在标准 Python 中运行。
`audit_acceptance.py` 汇总原始报告及定向补验，保留历史失败状态。

本草案已经验证来源与明文语义。物理布局、编译、Poseidon 密态执行、
编译程序包和真实 Agent 生成分别安排后续验收。
