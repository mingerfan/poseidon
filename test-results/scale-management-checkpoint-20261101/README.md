# Scale 与 Boot 管理检查点（2026-11-01）

保存当前已实现代码、验证证据和已撤销实验，随后再处理 scale 与 Boot 评估接口的拆分。
提交清单见 [commits.json](commits.json)，本次检查见 [validation.json](validation.json)。

## 当前实现

编译器已共用真实模数链的 scale/容量及 Boot 对齐计算，并在物理导出时合并兼容归约的 RS。
区域规划通过 `--native-scale-policy=regional --boot-placement=greedy` 开启，默认仍为 threshold。
区域 scale 搜索与 greedy Boot 目前共享同一 Planner 的存活状态；尚未提供可供 optimized
复用的独立算术评估接口。regional 与 optimized/depth-dp 的组合会明确拒绝，不能解释为已测得某个 Boot 数。
计算下限和 Boot 入口已拆开：当前 profile 允许计算到 Q5，Boot 入口仍是 Q6/s40，出口 Q31/s40。

## 完整 ResNet20 的已验证规划

s = log2(scale)，Qn 表示还剩 n 个模数。输入图和真实 Q50/P25 profile 保持一致。

| 方案 | 初始输入 | Boot | 最终输出 |
| --- | --- | ---: | --- |
| 当前 regional，不限定最终 scale | Q31 或 Q34 / s40 | 18 | Q6 / s约160 |
| 当前 regional，最终要求 s40 | Q31 或 Q34 / s40 | 19 | Q26 / s约40 |
| 历史手动阶段约束方案 | Q34 / s40 | 18 | Q5 / s约40 |

18次/s160 与 18次/s40 不是同一输出约定。放宽算术下限到 Q5，单独不足以让当前 regional
达到18次/s40。相同 Q34 输入下，当前自动方案第一次 Boot 前将 Q9 ModSwitch 到 Q6，丢掉3Q；
手动阶段约束方案先用这些 Q 做卷积再 Boot。比较图和物理入口审计见
[对比图与数据](../regional-level-comparison-20261031/comparison.json)。
以上为编译及 RuntimePlan 状态校验，不是整网密态准确率或 GPU 延迟测试。

## 验证与记录范围

本次本地通过20组 CTest（未构建的 frontend 测试除外）、15项 native 预算检查、18项 native
产物检查和9项手动 Boot 对齐检查。CTest 含22项 regional 算术计划回放。
188Server 此时不可达；此前在该机器完成的区域规划验证、真实模数验收和百万指令压力记录保留在原目录。

- [共享模数与对齐计算](../scale-state-step1-20261031/validation-summary.json)
- [归约 RS 合并](../reduction-rs-step2-20261031/validation-summary.json)
- [当前区域规划、完整模型与百万指令结果](../regional-scale-20261031/summary.json)
- [Q5 下限实验](../regional-arithmetic-q5-20261031/summary.json)
- [optimized 组合当前被拒绝的记录](../regional-optimized-check-20261031/result.json)
- [已回滚的局部 step3 实验](../local-scale-step3-20261031/validation-summary.json)
- [回滚后的乘法区域模拟](../multiplication-scale-simulation-20261031/summary.json)

`local-scale-step3-20261031` 中的补丁、源码与跑分只作历史证据，未重新加入当前编译器；
其中被撤回的跑分不能作为新区域算法的性能结论。当前区域算法百万条压力用例的最终记录为
35.11秒整段预编译、其中规划 pass 12.8459秒，峰值 RSS 1,338,444 KiB；
这不是任意 DAG 的复杂度保证。

生成的模型载荷、bytecode、物理计划二进制和本机审计可执行文件留在本地。
约3.292 GiB的 Q5 实验产物清单与 SHA256 见
[local-artifacts.json](../regional-arithmetic-q5-20261031/local-artifacts.json)，复现命令、审计源码、
状态轨迹和结果已提交。SSH 连接失败记录不能计作远端验收通过。
