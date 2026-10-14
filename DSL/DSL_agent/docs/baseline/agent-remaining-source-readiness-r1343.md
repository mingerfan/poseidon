# 拆分输入与 MHA 当前来源准备 r1343

剩余六项 chunk/unbind 和两项 MHA 已完成当前 SDK 下的隔离来源捕获、
独立参考检查和生成请求冻结。付费生成尚未批准或启动。

## 范围

| 模型 | 来源版本 | 当前状态 |
|---|---|---|
| chunk_branches | v17 | 来源与参考通过 |
| chunk_projection | v17 | 来源与参考通过 |
| chunk_public_weight | v17 | 来源与参考通过 |
| unbind_fusion | v17 | 来源与参考通过 |
| unbind_mutation | v17 | 来源与参考通过 |
| unbind_sequence | v17 | 来源与参考通过 |
| self_post_linear_residual__declared_shallow_v1 | v16 | 来源与参考通过 |
| linear_into_separate_cross_none__declared_shallow_v1 | v16 | 来源与参考通过 |

采用当前 `explicit-v13`、公开权重描述符和数学角色规则、P4/w45，
资源配置 `general-tensors-v1`。MHA 的独立参考针对声明的近似计算；
原函数应用精度仍单独保留。

八项共约 129.26 秒。来源任务没有生成或 HTTP 记录。
六份拆分请求的公有上下文为 40965–50408 字节；
MHA 两份为 118470、119342 字节，均满足原 131072 字节上限。
压缩请求经原 codec 恢复后，与捕获请求和公开投影一致。

本轮**编译、密态执行、解密和真实 Agent 新生成均为 0**。
六模型的旧规则验收、MHA 的旧失败和未知 HTTP 保持原记录与绑定，
本轮来源通过不替代这些结果。

## 绑定与后续

SDK 身份保持 `ee05605763e8f26e106c8c48167afb7017cee7c6fa946093626dc9c35bf29995`。
模型、公开权重、来源文件、完整请求、公有上下文和源码身份见
[机器记录](agent-remaining-source-readiness-r1343.json)。

来源准备采用独立 300 秒／64 MiB 小批次，原失败窗口和旧付费额度没有复用。
实际执行先创建新的任务和账本，当前来源任务只用于离线准备。

r1341 的八份 einsum 请求、源码身份和调用预算保持原样，仍待本次批准。
现在共 16 份当前来源请求已准备：8 einsum＋6 拆分＋2 MHA。
后八项需要独立冻结付费范围；本轮未创建新付费方案，也未增加待答批准问题。

654 项后端／数学参考、两份历史付费请求和既有 SDK 文件核验一致。
安装、下载、后端／参数／reference／门限修改、commit 和 push 为 0。
通用输入与算子、容量、应用精度、bootstrap、密文交接、CUDA 和整网目标继续。

下一条只读命令：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/einsum_live_v1/run.py --plan
~~~
