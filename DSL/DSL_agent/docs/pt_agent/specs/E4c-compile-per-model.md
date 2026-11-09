# E4c：逐个模型 trace 和编译（E4b 磁盘接续，2026-11-05，Claude 审查 E4b 后）

E4b 的记录（`/home/lhohy/pt_agent_results/E4b-compile-burden-20261105/`）：
- MLP、ResNet、MobileNet trace 成功，Earth op 分别为 457 / 10,089 / 45,673，输入 IR 分别为 1,401,825 / 1,570,843,452 /
  8,077,579,096 字节。上游 tracer 把权重以文本常量写进 IR，所以文件很大。
- E4 spec 规定 6 个模型全部 trace 完再编译。ResNet 和 MobileNet 的 IR 共 9.65 GB 留在盘上，剩余空间降到 4.47 GB，
  低于 trusted_run 的 10 GiB 下限，后 3 个 trace 和全部编译都在启动前被拒绝。原因在 spec 规定的顺序。
- E4b 新加的编译参数、快照计数和计时解析还没有在真实编译上运行过。
- 驱动的 RSS 采样线程调用 `ps`，Nix 环境里没有这个命令，线程一启动就退出（E4 和 E4b 的 compile 日志里都有这条
  FileNotFoundError）。此前没有编译超时，所以结果没有受影响。

本轮改三处：执行顺序、编译时的文本输出、RSS 采样。其余沿用 E4b spec 及它引用的 E4 spec。API 0，Nix 独占，串行。

结果目录：`/home/lhohy/pt_agent_results/E4c-compile-burden-<开跑日期>/`。复制 E4b 的 `run.sh`、`driver.py`、`launcher.py`、
`collect.py`，只做下面列出的修改。E4、E4b 目录原样保留。registry 的 round 为 `E4c`，服务器作业目录名为 `E4c-<模型>`。

## 1　顺序：一个模型做完再做下一个

- 顺序：MLP、ResNet、SqueezeNet、AlexNet、VGG16。每个模型依次执行 trace → depth-dp → optimized → 清理，然后才开始下一个。
- 清理：每次编译在计数和 `plan_stats` 之后，先执行 `toolchain.prune(work / P)`，再按 50 MiB 规则清理。一个模型的两次编译
  都结束后（包括跳过），按 50 MiB 规则删除它的 traced IR。
- MobileNet 本机不 trace，也不编译。E4b 已测得它的输入 IR 为 8,077,579,096 字节，trace 后剩余空间约 6 GB，编译会在
  启动前被拒绝。它的 TRACE.json 引用 E4b 的记录（earth_ops 45,673，注明来源路径），两次编译记为 `skipped`，reason 写
  "E4b 实测输入 IR 8,077,579,096 字节，本机剩余空间不够编译"，并按本 spec 的命令导出服务器作业。
- SqueezeNet、AlexNet、VGG16 的 IR 大小未知，trace 受 8 GiB 单文件限制。trace 因文件大小失败，或 trace 后剩余空间低于
  10 GiB 时，按 E4b §4 记录（trace 失败或编译 `disk`），导出服务器作业，删除残留 IR，继续下一个模型。
- E4 的超时跳过规则保留：某个模型的 optimized 超时后，trace 得到的 Earth op 更多的模型跳过 optimized。
- 每次 trace 和编译前记录剩余空间（沿用 E4b）。

## 2　编译命令：省略大常量的文本打印

`expert_argv` 在 `--enable-debug-printer` 之后加 `--mlir-elide-elementsattrs-if-larger=16`。

- hecate-opt 注册了 MLIR 的打印选项（构建源码 `src/tools/optimizer.cpp:139` `registerAsmPrinterCLOptions()`）。
  DaCapo 流水线的两个快照（`lowered.earth.mlir`、`lowered.ckks.mlir`，optimizer.cpp:608-622）用 `OpPrintingFlags()` 构造
  打印设置，MLIR 18 的这个构造函数读取命令行打印选项；最终 `-o` 输出也用这套选项。
- 这个选项只影响文本打印：元素多于 16 个的常量打印成 `dense_resource<__elided__>`。RuntimePlan 的明文负载由
  EmitRuntimePlan 从内存中的属性写出（`EmitRuntimePlan.cpp` 的 `encodePayload`），所以计划、`plan_stats` 和 op 计数都不受影响。
- 原因：不加这个选项时，ResNet 的两个快照和输出各约 1.6 GB 文本，打印时间会计入 `compile_s`，也占用磁盘。
- COMPILE.json 记录清理前的 `snapshot_bytes`（lowered.earth.mlir）和 `output_bytes`（lowered.mlir）。MLP 的快照里应当有
  `__elided__`；没有时在 notes 里写明，继续运行。

## 3　RSS 采样改为读 /proc

- `Sampler.sample` 遍历 `/proc/<pid>`：`comm` 等于 `hecate-opt`，并且 `cmdline`（NUL 分隔）包含工作目录的进程，
  从 `status` 的 `VmRSS` 读取内存。读取途中进程退出就跳过它。
- 采样线程出现异常时，把异常文本写进 COMPILE.json 的 `sampler_error`，编译照常进行。
- 超时记录中的 RSS 下界来自这些样本（沿用 E4）。

## 4　汇总

- collect 产出 `E4c.json`、`E4c.md`、`POINTS.csv`，行的组成与 E4b 相同。专家程序行来自 E4c，MobileNet 的 trace 来自 E4b。
  表头增加 `snapshot_bytes`。
- E4c.md 注明三点：编译时省略了大常量的文本打印；MobileNet 只导出了服务器作业；einops 版本差异（同 E4b）。

## 5　不要做

- 不删除 E4、E4b 或其他历史结果来腾出空间。
- 其余同 E4b §6：不安装、下载或复制依赖；不改上游脚本、poly、前端、编译器或 pt_agent 库；不提高限制；失败不重试；
  不调用 API，不运行服务器作业，不提交、不推送。

## 6　完成标准

- MLP、ResNet 各有 TRACE.json 和两份 COMPILE.json。完成的编译有 `boots_snapshot`、`placement_s`、`snapshot_bytes`，导出成功的
  还有 `plan`。
- SqueezeNet、AlexNet、VGG16 各有 TRACE.json，以及两份 COMPILE.json 或写明原因的跳过记录。MobileNet 有引用记录、两份跳过记录
  和服务器作业。
- E4c.json、E4c.md、POINTS.csv 齐全；registry 中 round 为 `E4c`；最后一次残留进程检查为 0。
- STATUS.md 新增 E4c 一行，"与 spec 不一致之处"逐条列出。
