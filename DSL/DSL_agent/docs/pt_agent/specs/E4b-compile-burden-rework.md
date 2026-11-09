# E4b：E4 返工（2026-11-05，Claude 审查 E4 后）

E4 的驱动与 spec 一致，但 spec 本身有两处疏漏，导致专家程序没有得到有效编译数据：
1. 上游 `poly/MPCB.py:4` 导入 `einops`，trace 环境里没有它。ResNet、MobileNet、SqueezeNet、AlexNet、VGG16 在导入阶段退出。
2. 团队移植后的 `--dacapo` 流水线总在最后加 `EmitRuntimePlan`（构建源码 `src/tools/optimizer.cpp:566-626`、`:340-420`），
   缺少 RuntimePlan 元数据参数时报错退出。MLP 的两次编译都在这一步退出，此前的放置 pass 已全部跑完。

本轮只改这两处，其余全部沿用 E4 spec（限制、顺序、跳过规则、残留进程清理、服务器作业、计数规则、汇总字段）。API 0，Nix 独占，串行。

结果目录：`/home/lhohy/pt_agent_results/E4b-compile-burden-<开跑日期>/`。把 E4 的 `run.sh`、`driver.py`、`launcher.py`、`collect.py`
复制过来，只做下面列出的修改。E4 目录原样保留，作为历史。

## 1　einops：沿用 B 线已安装的依赖

- 路径：`pt_agent.paths.PLATFORM_DIR / 'deps/poly-python/einops-0.6.1'`（本机为 `/home/lhohy/poseidon-work/platforms/aarch64-linux/deps/poly-python/einops-0.6.1`）。
  这是 B 线 2026-09-21 按 `scripts/baseline/poly_dependencies.py` 装好的 einops 0.6.1，不需要下载或安装。
  用户 2026-11-05 已批准本轮沿用这份依赖（"批准"）。
- `trace_inside` 和 `server_job` 的 PYTHONPATH 在上游 poly 之后追加这个目录。
- 开跑前检查 `einops/__init__.py` 存在，不存在就停止并写进阻塞。不调用 `poly_dependencies.verify()`，不加任何哈希检查。
- 上游 `requirements.txt` 写的是 einops 0.7.0。MPCB 只用到 `einops.rearrange` 和 `einops.repeat`，两版接口相同；E4b.md 里注明版本差异。

## 2　编译命令：补齐 RuntimePlan 元数据，打开 DaCapo 的调试快照

`expert_argv(model, placement)` 改为：

```python
plan_args = [a for a in toolchain.hecate_command(dacapo=True) if a.startswith('--runtime-plan-')]
argv = [str(toolchain.host_paths()['hecate_opt']), '--dacapo', f'--boot-placement={placement}',
        f'--ckks-config={DACAPO_UPSTREAM}/profiled_HEAAN_CPU.json', '--waterline=40', '--mlir-timing',
        '--enable-debug-printer', *plan_args,
        str(work / f'traced/{model}.mlir'), '-o', str(work / placement / 'lowered.mlir')]
```

参数作用：
- `plan_args` 与本项目其他编译相同（json 格式、files 包、target/context/operator-spec 元数据）。它们只作用于最后的 EmitRuntimePlan，
  不影响放置 pass。
- `--enable-debug-printer` 是 DaCapo README 中 `hbt` 的用法。它在放置完成、转换到 CKKS 之前，把 Earth IR 快照写到
  `work/P/lowered.earth.mlir`（optimizer.cpp 的 `createLocationSnapshotPass`）。
- 输出名改为 `lowered.mlir`，这样能直接用 `toolchain.plan_stats` 读取计划，`toolchain.prune` 也能清理快照和包。

## 3　记录字段（COMPILE.json 新增）

- `boots_snapshot`：按 E4 §1.4 的计数规则数 `lowered.earth.mlir` 里的 `earth.bootstrap`。这是主要的自举数，即使计划导出失败也能得到。
- `plan`：导出成功时，记录 `toolchain.plan_stats(work/P, function=<名称>)` 的结果。`<名称>` 取自唯一的
  `lowered._hecate_*.runtime-plan.json`，计划里的 boot 数记为 `boots_plan`。两个自举数不一致时都保留，并在 notes 里注明，不停止运行。
- 从 `--mlir-timing` 报告中解析三个时间：
  - `export_s`：EmitRuntimePlan 一行的墙钟时间；
  - `placement_s`：放置 pass 的墙钟时间之和。optimized 为 BypassDetection、CandidateSelection、DaCapoPlanner、BootstrapPlacement、
    ProactiveRescaling；depth-dp 为 GreedyBootstrapPlacement；
  - `optimize_s`：`compile_s − export_s`（计时报告缺失时为 null）。
- 结果分类沿用 E4。另增 `export_failed`：放置 pass 已出现在计时报告里，但 EmitRuntimePlan 报错退出。此时 `placement_s` 和
  `boots_snapshot` 照常记录。

清理：计数和 `plan_stats` 完成后，先执行 `toolchain.prune(work / P)`（删除 lowered.* 和包），再执行 E4 的 50 MiB 规则。

## 4　磁盘

本机剩余约 14 GB，trusted_run 要求至少 10 GiB，大模型的快照和明文包可能放不下。
- 每次编译前记录剩余空间。
- trusted_run 因磁盘不足报错时，该模型记为 `disk`，导出服务器作业（同 E4 §1.5，命令按本 spec），然后继续下一个模型。

## 5　不重做的部分

- B 部分（6 个 PT-DSL 程序的规模）直接引用 `E4-compile-burden-20261105/ptdsl/*/SIZE.json`，不重新 trace。
- collect 的 M7、M7b、M2、DaCapo 论文行与 E4 相同。表头增加 placement_s、export_s、boots_snapshot、boots_plan 四列。
  POINTS.csv 增加 placement_s 列。

## 6　不要做

- 不安装、下载或复制任何依赖；einops 只按 §1 加进路径。
- 不改上游脚本、poly、前端、编译器或 pt_agent 库；不提高限制；失败不重试。
- 不调用 API，不运行服务器作业，不提交、不推送。

## 7　完成标准

- 6 个专家程序各有 TRACE.json 和两份 COMPILE.json（或写明原因的跳过记录）。完成的编译要有 `boots_snapshot`，导出成功的还要有 `plan`。
- E4b.json、E4b.md、POINTS.csv 齐全；registry 中 round 为 `E4b`；最后一次残留进程检查为 0。
- STATUS.md 新增 E4b 一行，"与 spec 不一致之处"逐条列出。
