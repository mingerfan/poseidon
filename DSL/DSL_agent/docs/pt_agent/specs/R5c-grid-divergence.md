# R5c：网格中发散的配置照实记录，跑完 r5_resnet20

R5b 的阻塞已由 Claude 于 2026-10-20 诊断完毕，结论如下：

- deg=63、margin=1.0 时，校准图 300–349 这一批中，第 17 个 act（`add_6`）有 1 个元素轻微越出定义域（|u|=1.016）。
- 高阶 Chebyshev 多项式在 [-1,1] 之外按指数增长，误差逐层放大：到 `add_7` 已有 514 个元素越界、|u|=32，再下一层达到 3e109，最终溢出。
- 同一批图在 (63, 1.25) 和 (127, 1.25) 两种配置下，全部 19 个 act 的 |u| 都不超过 0.85，不会越界。

这是实验结论，**不是代码缺陷**。所以网格不要因为某组发散而中止，要把它作为一个结果记录下来。

## 1. 修改：只改 `pt_agent/experiments/r5_resnet20.py`

1. 网格中每组配置在逐批计算时，如果 Mock 抛出 `BackendError`，就停止这一组，记录 `diverged=True`、`images_done`（此前已完整算完的张数）和 `diverged_batch_start`（出错那一批的起始索引）。agree、top1、rmse 只按已完成的图计算并照常写出，然后**继续下一组**。
2. 每组都通过 build 的 observer 额外记录 `max_u`：所有 act 输入经仿射映射后的 |u| 最大值，只统计有效槽。同时记录 `max_u_act`，即取到这个最大值的张量名。
3. 第 4 步 Emit 选配置时，只从 `diverged=False` 的组中选 agree 最高的；同分时取 deg 较小的。如果全部发散，跳过 Emit，并记录 `emit={"skipped": "all configurations diverged"}`。
4. 重新运行 `python3 -m pt_agent.experiments.r5_resnet20`，从头跑完全部 5 步。之前的结果目录保留，不覆盖。

## 2. 测试

`tests/test_r5_resnet20.py` 中加一个测试：用一个只有 2 张图、手工把某个 act 的定义域缩到 `(min*0.5, max*0.5)` 的配置，调用实验里负责单组网格的函数（如果它还不是独立函数，就先把它抽出来），断言返回的 `diverged=True`，且不抛异常。R1–R5 的测试依然全部通过。

## 3. 完成标准与汇报

- r5_resnet20 跑完，`complete=true`。
- 汇报 6 组网格的 deg、margin、diverged、images_done、agree、top1、logit_rmse、max_u 和 max_u_act，以及 Emit 常量大小和编译结果（或服务器作业路径）。
- STATUS.md 中 R5、R5b 两行改为完成；新增 R5c 一行；删除"阻塞"栏里与 R5、R5b 有关的条目。
