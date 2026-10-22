# R5b：修复常量去重的内存问题，重跑 r5_resnet20

R5 被阻塞的原因：`backends/base.py` 的 `Backend.const` 用 `tuple(data.tolist())` 作为去重键。每个长度为 16384 的常量会变成一个约 0.5 MB 的 Python 浮点元组，ResNet20 有 4992 个不同常量，在 4 GiB 的进程限制下报 MemoryError。另外，CountBackend 根本用不到常量的数据，却也存了一份。

Claude 已于 2026-10-20 用临时脚本验证过修复方案：整网 Count（63 阶，定义域取校准范围）用时 0.6 秒、峰值 200 MB；batch=50 的 Mock 一批用时 7.3 秒、峰值 1.1 GB。

## 1. 修改

1. `pt_agent/backends/base.py` 的 `Backend.const`：
   - 去重键改为 `hashlib.blake2b(data.tobytes(), digest_size=16).digest()`。这个摘要只用来去重，不是校验。
   - `_Plain` 的 data：只有 CountBackend 存 `None`，Mock 和 Emit 照常保存数组。在 `Backend` 上加一个类属性 `_keep_const_data = True`，CountBackend 把它设为 False。
   - CountBackend 中任何地方都不能读取 Plain 的 data。
2. `pt_agent/experiments/r5_resnet20.py`：网格计算的 batch 从 200 改为 50（1000 张校准图分 20 批）。其余步骤不变。

## 2. 测试

在 `pt_agent/tests/test_r5_resnet20.py` 末尾加一个测试：

- 读取 `/home/lhohy/pt_agent_results/r3-resnet20-20261020T131658/` 下的 graph 和 calibration.json；
- 所有 act 取 `("cheb", 63, 该张量的 min, 该张量的 max)`，用 `CountBackend(16384)` 组装整网；
- 断言：
  - counts = {add:6487, sub:171, neg:0, mul_cc:304, mul_cp:4800, rot:2624, boot:0}；
  - `ops_total=14386`，`distinct_consts=4992`，`len(rot_steps)=182`，`max_depth=193`；
  - 所有 Plain 的 data 都是 None；
  - 用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 读到的峰值不超过 1 GiB。

R2 的测试必须仍然全部通过，因为 distinct_consts 的语义没变。

## 3. 运行与完成标准

- R1–R5 的测试全部通过。
- 重跑 `python3 -m pt_agent.experiments.r5_resnet20`，完成 R5 spec §8 的全部 5 步。
- 汇报内容与 R5 §11 相同：精确模式的误差、counts 和 max_depth、6 组网格结果、Emit 常量大小、编译结果或服务器作业。
- 编译阶段如果因为内存或时间超限失败，照实记录，并保留服务器作业，不重试。
- STATUS.md 中 R5 一行改为完成；R5b 新增一行。
