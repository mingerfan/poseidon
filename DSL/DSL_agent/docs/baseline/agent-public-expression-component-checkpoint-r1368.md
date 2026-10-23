# PyTorch v23 公开张量表达式接入

显式来源版本 `poseidon-trusted-pytorch-v23` 已接入应用与程序包接口。当前完成来源捕获、独立明文参考及完整性检查；Poseidon 编译、密态和真实 Agent 验收分别等待后续批次。

## 接入范围

公开参数或 buffer 可通过 reshape/view、permute/transpose/t/T、静态基础切片、unsqueeze/squeeze/flatten、contiguous 和 bool→float32/float64 转换参与模型计算。输入形状查询仍来自静态签名。

变换只处理由已认证公开状态组成的闭合表达式。派生常量记录逻辑 dtype、物理 float64 数据、原始数组、来源节点、变换参数及 NPY 哈希。加密运算及其顺序保持，数值除法沿用非零、广播、幅值和倒数上限。

重载时从原始 NPY 重算派生常量并核验整条来源映射，过程只读取声明式记录和数组。首次可信 Python 捕获仍在既有独立沙箱执行。

新增版本与旧 v22、JSON 和 schema 1–5 分开登记。默认来源版本仍为 v18，latest 仍为 v21；资源配置、后端库及参数保持。

## 当前检查

| 检查 | Passed | Failed / Errors / Skipped |
|---|---:|---:|
| 数学与容量边界 | 31 | 0 / 0 / 0 |
| FX 来源与变换映射 | 22 | 0 / 0 / 0 |
| 显式静态策略与旧 v22 行为 | 14 | 0 / 0 / 0 |
| 整包来源、参考及重载 | 12 | 0 / 0 / 0 |
| 旧接口及隔离来源入口 | 167 | 0 / 0 / 0 |
| 下一批审批与冻结检查 | 7 | 0 / 0 / 0 |
| 合计 | 253 | 0 / 0 / 0 |

五个变换分区各有三个不同 shape／参数上下文。18 份来源配置完成 288 组 CPU float64 参考比较、1,712 个输出值及程序包来源重载。其中六份另行完成实际隔离源码捕获；它们是原配置的子集。

可选原生精度接口的 v22／v23 字典 None 输出完成两个真实隔离 worker 检查。百万元素公开数组和 rank32 只完成数学／静态容量边界检查。

本阶段编译、密态、Provider／HTTP 均为 0。上述明文和来源结果不计入密态或真实 Agent 成功率。

## 历史与当前绑定

r1364 的五项真实 Agent 批次已结算：5/5，通过时使用原绑定，6 次生成／6 次 HTTP。原报告、批准记录和生成来源保持原字节。当前输入适配使用新 SDK 与请求绑定，旧记录保留为该版本的历史证据。

当前 SDK：b41fe9599da34b14b013ce1aad5237969897132f0f0130643c795fc780841d5a  
阶段报告绑定：68ba70fc39c7fc02f01e6d4071aabe68222b9bb554bab7dd528c51d6705a5237  
四个阶段的后端身份均与 r1366 一致；本批变更为 Agent 自有模块、测试和文档。分支 lhy-agent-dsl、HEAD 7588e480 保持，已有修改保留。网络、依赖、库、参数、reference 与数值门限保持，commit/push 为 0。

首次夹具及配置字段失败、原始日志和源码版本均保留。范围检查按既有真实参数和公开数据上限执行；未修改后端以获得通过。

## 下一批

[r1369 离线范围](agent-public-expression-offline-proposal-r1369.md)：18 个原案例，固定规则候选；检查 Hecate、RuntimePlan／OperatorSpec／bundle、Poseidon CPU 数值及三个程序包的新输入和重复执行。它属于规则基线验收，真实 Agent 新生成另行冻结预算。

只读命令：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_expression_cpu_v1/run.py --plan
```

详细机器记录见同名 JSON。后续仍包括通用算子、布局容量、bootstrap、整模型和 CUDA 的独立里程碑。
