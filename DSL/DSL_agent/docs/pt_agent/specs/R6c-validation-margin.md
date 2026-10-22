# R6c：用验证集选择近似定义域，测试集如实统计发散的图

## 0. 背景（Claude 于 2026-10-20 的诊断）

- R6 第二次运行（deg=127、margin=1.25）在测试集第 50–99 张图这一批出现非有限值，原因是激活输入越出了定义域，与 R5c 的诊断同属一类。
- 定义域由 1000 张校准图（训练集）的 min/max 乘以 margin 得到。Claude 在训练集中另取 10000 张**未参与校准**的图（索引 10000–19999 中不属于校准集的部分），只用精确网络做前向，统计结果是：要覆盖所有激活值，至少需要 **1.33 倍**余量。其中 bn1 需要 1.330、add_1 需要 1.320、add_3 需要 1.283。所以 1.25 本来就不够，再叠加近似误差带来的偏移，个别图一定会越界。
- 原则：**测试集只用于最终报告，不用于选参数。** margin 和 deg 由验证集决定，验证集取自训练集，与校准集不相交。

## 1. 修改

路径相对 `DSL/DSL_agent/scripts/`：

### 1.1 `pt_agent/data/cifar.py`

新增函数：

```python
def validation_set(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """训练集索引从 10000 起递增，跳过校准集中出现过的索引，取前 n 个。
    返回 (归一化后的图, 标签)，读取方式与 calibration_set 相同。"""
```

### 1.2 `pt_agent/run.py`：新增 `validate` stage，修改 `plan` 和 `test`

**stage 的顺序**：`validate → plan → check → mock → count → emit → compile → test`。

**validate**：

1. 先用精确网络统计"需要的余量"：在 `validation_set(2000)` 上执行 `reference_forward(keep=True)`。对每个 act 的输入张量 a，计算每张图的 `max(v.max()/hi_cal, v.min()/lo_cal)`，取所有图上的最大值，得到 `exact_needed_margin[a]`；总的需要余量 = 各 a 中的最大值。其中 lo_cal、hi_cal 是 calibration 中该张量的 min 和 max。
2. 候选组合：deg ∈ {63, 127}，margin ∈ {1.4, 1.5, 1.75}，共 6 组。每组都执行：
   - 用 `rule.plan(graph, calibration, deg, margin)` 生成程序 → parse → lower；
   - 在 `MockBackend(16384, batch=50)` 上用验证集的 2000 张图跑近似模式；
   - 遇到某批抛 BackendError，就记为该组 `diverged=True`、`images_done`，停止这一组并继续下一组；
   - 用 observer 记录 `max_u` 和 `max_u_act`；
   - 记录 `agree`（argmax 与精确参考一致的张数）、`logit_rmse` 和 `seconds`。
3. **选择规则**：在 `diverged=False` 且 `max_u ≤ 0.95` 的组中，选 agree 最高的；agree 相同时取 deg 较小的，再相同时取 margin 较小的。没有符合条件的组时，validate 报错退出，并在 REPORT 中写明。
4. 结果写入 REPORT：`validation={exact_needed_margin_overall, exact_needed_margin_by_act（前 5 项）, grid=[…], selected={deg, margin}}`。

**plan**：如果本次运行包含 validate，就用 `selected` 中的 deg 和 margin，此时命令行的 `--deg`、`--margin` 必须为空，否则报错。不包含 validate 时，沿用 R6 的行为。

**test**（仍然需要 `--final`）：

- 按 batch=50 计算。某一批抛 BackendError 时，把这一批改为逐张（batch=1）重算：
  - 能算出结果的图正常计入；
  - 单张仍然抛错的图记为发散，**算作分类错误**。
- 这一步只是为了把发散的图逐张定位出来、照实统计，不是重试或修复，所以 deg、margin、定义域一律不改。
- 记录 `top1_test`（分母为 10000，发散的图算错）、`n_diverged`、`diverged_indices`（测试集索引），以及 `m1_item3_pass = top1_test >= 90.73`。

## 2. 测试（nixrun，`test_r6c_*.py`）

1. `validation_set(2000)`：与校准集的索引交集为空，所有索引都 ≥10000，返回 2000 张。
2. 选择规则做成纯函数 `select(grid_rows)`，用手工构造的输入覆盖以下情况：
   - 有发散的组被排除；
   - max_u > 0.95 的组被排除；
   - agree 相同时取较小的 deg；
   - 全部不合格时抛错。
3. test 阶段逐张统计的逻辑抽成函数 `evaluate_batches(run_batch, images, labels, batch)`。给它一个假的 `run_batch`，让它对包含第 7 号图的批次抛 BackendError、单独算第 7 号图时也抛错，验证结果为：n_diverged=1，`diverged_indices=[7]`，第 7 号图计为错误，其余图正常统计。
4. R1–R6 的测试依然全部通过。

## 3. 运行与完成标准

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.run --model resnet20 --planner rule \
    --stages validate,plan,check,mock,count,emit,compile,test --final
```

- 汇报中贴出：
  - exact_needed_margin_overall 及前 5 个 act；
  - validate 的 6 组结果（diverged、images_done、agree、max_u、max_u_act）和选中的组；
  - 校准集上的 agree；
  - counts 和 max_depth；
  - 编译结果（耗时、峰值内存、boots）；
  - top1_test、n_diverged、m1_item3_pass。
- 测试阶段即使有发散的图也要跑完，照实报告。
- STATUS.md：R6 改为完成，新增 R6c 一行；M1 第 3 条按 `m1_item3_pass` 填写，第 4 条填写编译结果；删除"阻塞"栏中与 R6 测试中断有关的条目。

## 4. 不要做

- 不用测试集选任何参数；不读取 R6 失败那批测试图的诊断信息来做选择。
- 不修改 kernel、lowering、规划器的算法，也不修改校准集。
- 编译失败不重试。
