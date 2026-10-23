# Bootstrap 误差定位 r235

已用独立数学计算解释 r234 有符号输入的主要偏差。没有新增密态运行。

## 结论

当前 Bootstrapper 的 EvalMod 完成多项式和两次倍角后，近似的是归一化正弦，而不是精确的模约减。对编码后的实系数 c，其主要映射为：

`c → R × A × sin(2πc/R)`

其中 R=32；固定多项式推导出的 A=0.15915569197480922，接近 1/(2π)。该函数在零点附近接近恒等映射，但对非零数据有偏差。不能用零输入通过推断非零消息精度。

独立实现使用 CKKS 奇次单位根定义进行编码与解码，不调用 Poseidon encoder、Bootstrapper 或 DSL 转换器。Clenshaw 多项式计算与现有 heap 的数学展开也单独比较。

| 项目 | 最大差异 |
|---|---:|
| CKKS 数学编码/解码往返 | 2.11e-14 |
| Clenshaw 与 heap 数学展开 | 2.05e-14 |
| 零进位小区间：多项式倍角与归一化正弦 | 1.88e-12 |
| 正弦模型相对原消息 | 1.00979e-4 |
| 实际密态结果相对正弦模型 | 4.15060e-6 |

正弦模型预测 16,384 个槽位全部超过原门限，与 r234 失败范围一致。实际密态结果与预测的平均绝对差为 7.16e-7。主要误差由这个数学近似解释；剩余差异仍包含多项式进位区间、CKKS 算术和尺度对齐误差，未逐一隔离。

## 后续试验

已有源码还包含另一条 EvalModPoly/arcsine 路径，但它不等于当前 BootstrapConfig 路径已验证，暂不切换算法。

最小方案保留 Q/P、安全级别及已构建库，将消息比率与配套输出倍数同时设为 128。独立数学预测原四组输入的最大误差为 5.05e-6；这不包含密态噪声，不能记为验收通过。

另一个已发现的接口限制是刷新输出 scale 约为 2^48，超过下一次刷新所需的 q0/R。零输入掩盖了这个问题。提案在第二次刷新前使用真实乘常数1及 rescale，将 scale 归一化到约 2^40；保留原消息，消耗一个 level，不采用元数据覆盖或解密再加密。探针增加输入 scale 门禁。

[冻结提案](poseidon-native-bootstrap-proposal-r235.json)及独立源码已准备，尚未构建或运行。需要新的参数试验批准后执行；不修改当前 Agent 或 Runtime。

## 验证与接续

- 新增独立数学审计工具及 5 项单元测试：5 passed，0 failed，0 skipped。
- 原 r234 失败及全部绑定保留。
- 新付费调用、依赖安装和下载均为 0；无提交或推送。
- [机器报告](poseidon-bootstrap-math-audit-r235.json)。
- 工具：`scripts/baseline/bootstrap_math_audit.py`。
- 原证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-bootstrap-experiment-r234/`。
