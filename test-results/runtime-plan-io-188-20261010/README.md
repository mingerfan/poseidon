# RuntimePlan I/O 优化验收

2026-10-10，188Server / HPU-001。实施依据为 [I/O 优化方案](../../docs/poseidon-runtime-plan-io-optimization.zh-CN.md)。

阶段 A 使用 ckks-runtime `0ead354`，Release、GCC 11.4、单线程、nice 10。片段从原有四卡 Qwen V2 execution 数组提取完整记录，没有重新跟踪或编译模型。输入来源 SHA-256 为 `e7b0f54fb91ca0a606b021ba9dd7b7ac8f47e345aa8fedfdd440933575e15547`；各片段摘要在 [samples.jsonl](samples.jsonl)。

| 完整指令条数 | 严格单遍 SAX DOM 解析中位数 | 三次范围 |
| --- | ---: | ---: |
| 100,000 | 0.343 秒 | 0.341–0.344 秒 |
| 200,000 | 0.722 秒 | 0.689–0.722 秒 |
| 400,000 | 1.386 秒 | 1.383–1.390 秒 |
| 800,000 | 2.777 秒 | 2.759–2.782 秒 |

数据翻倍的比值为 2.10、1.92、2.00。上述只计解析，读取、哈希和销毁独立记录在 [stage-a-scaling.jsonl](stage-a-scaling.jsonl)。未清共享服务器缓存，不标为冷启动。方案中的原回调路径 400,000 条为单次 61.886 秒，测量次数与当前三次中位数不同。

阶段 B 使用 DaCapo `7177b46`：默认紧凑 JSON，以流写出并关闭、检查错误后 rename 发布入口；`pretty=true` 可用于诊断，`report-io=true` 报告导出时间及字节数。此阶段尚保留 plan DOM 与 manifest 字符串。

本地 ckks-runtime 全部 5 组 CTest 通过；DaCapo I/O、导出流水线、内存与明文 streaming 共 4 组 CTest 通过。新增检查覆盖转义重复键、容器、截断、尾部垃圾、数值溢出，以及紧凑/诊断输出逐项等价、原始 manifest 摘要、输出确定性和发布失败时保留旧入口。

## 阶段 C

ckks-runtime `4d403a8` 以 64 KiB 输入缓冲区执行单遍 SAX，完成一个记录即转换为 typed plan。临时 DOM 只覆盖一个描述符、指令或 manifest 条目；仍保存完整 typed plan。manifest 直接建立内容长度索引，不保留全文、条目数组或重复的集合。SHA-256 支持增量更新，保持原始字节摘要。

DaCapo `52f071f` 按公共布局中的 ID、ordinal 和阶段逐条写出，manifest 逐项写入并增量哈希。常量仍按 DenseElementsAttr 和内容摘要去重。输出缓冲区为 64 KiB；只在有界临时记录中构造 JSON，未保存完整 plan/manifest DOM 或输出字符串。`--runtime-plan-pretty` 和 `--runtime-plan-report-io` 也可从编译器顶层入口传递。

产物先写入同一文件系统的 staging 文件/目录，关闭并检查全部文件后发布 bundle，最后 rename 发布 plan。复用已有 bundle 要求其原始 manifest 摘要与每个 blob 均匹配；已存在的不同 bundle 应使用新 prefix。默认写入模式不支持多个进程共用一个 prefix。失败清理和损坏的既有 blob 均有测试。

### 完整四卡 V2 元数据加载，O3 首次测量

| 项目 | 测量 |
| --- | ---: |
| 原始 JSON 字节数 | 8,427,772,779 |
| 读取 | 3.11 秒 |
| 解析与 typed plan 构建 | 224.70 秒 |
| 增量 SHA-256 | 50.32 秒 |
| 加载合计 | 278.13 秒 |
| 销毁 | 1.09 秒 |
| 加载峰值 RSS | 5.52 GiB |
| values / initialization / execution | 10,669,278 / 3,281,673 / 18,054,435 |
| typed 数组有效元素 / capacity 字节 | 4,011,276,432 / 6,945,767,424 |

原始结果见 [stage-c-gpu4-load.json](stage-c-gpu4-load.json)，源摘要与方案中的四卡文件相同。GCC 11.4、`-O3 -DNDEBUG`、单线程、nice 10，地址空间上限 32 GiB。数组 capacity 是虚拟容量，不等于实际驻留页；嵌套输入、字符串和索引未包含在数组字节指标内。16 GiB 的元数据加载目标已达到，180 秒的时间目标尚未达到。

方案的两遍解析对照为 425.17 秒、52.26 GiB，使用 `-O2 -DNDEBUG`；不能把编译选项不同的总时间差全部归因于解析器改动。上述测量不包含 Verifier、bundle 索引、执行任务或 GPU 初始化，后续分别报告。

本地 ckks-runtime 5/5、DaCapo 9/9、Poseidon CPU API 测试通过。逐项比对真实 V1 样例的全部 target、values、指令属性、输入输出和 bundle 引用；V2/V3 版本约束、任意根字段顺序、跨 64 KiB 边界、尾部空白摘要、截断、尾部垃圾、I/O 错误、manifest 重复内容、转义重复键和错误长度均有回归检查。真实 GPU MLP 采用已有无 Boot 校准计划，fixture 和 mock 均在既有绝对/相对容差内（最大绝对误差 0.52479，最大相对误差 0.01235）；[结果](gpu-mlp.json)和[日志](gpu-mlp.log)单独记录，不表示全量 Qwen GPU 执行通过。

### 完整四卡 V3，O2 加载与全局校验

[stage-c-v3-load-verify-o2.json](stage-c-v3-load-verify-o2.json) 使用原有完整 V3 文件，GCC 11.4、`-O2 -DNDEBUG`、单线程、nice 10、32 GiB 地址空间限制。源 SHA-256 为 `bd6b45b7f06e04456b2cf4665e7c975d9bdd6b5a4c864f072ef3833a54939d9a`，包含在线 Encode 和 Fence。此次测量与常量恢复工具同时运行，服务器也有其他负载；缓存未控制。

| 项目 | 测量 |
| --- | ---: |
| 原始 JSON 字节数 | 8,756,201,335 |
| values / execution（initialization 为空） | 11,064,667 / 22,149,239 |
| 读取 / 解析与构建 / SHA-256 | 1.72 / 216.63 / 56.67 秒 |
| 元数据加载合计 | 275.03 秒 |
| 元数据加载峰值 RSS | 5.69 GiB |
| 完整 PlanVerifier | 33.52 秒 |
| bundle manifest 摘要校验与索引 | 7.99 秒 |
| 包含校验后的进程峰值 RSS | 8.73 GiB |
| 计划销毁 | 1.21 秒 |

全局定义、使用、Release/reuse/Fence、物理元数据、能力和摘要检查均通过，没有关闭摘要验证。bundle 阶段只建 manifest 索引；不读取全部 blob、不构造 RNS 明文。上述合计仍不包含 runtime 的描述符索引、使用次数表、任务闭包、rank/device 资源绑定或 GPU 初始化与执行，不能标作完整启动时间。

### 全量导出的输入边界

原有四卡 `qwen24.mlir` 是诊断产物：部分 Encode 已被后续 `.cst` 导出改成整数索引。直接重新导出在第一处这样的 Encode 明确失败，所有 plan/bundle staging 文件被清理；[失败记录](diagnostic-export-failed-process.json)与[日志](diagnostic-export-failed.log)保留。该次 90.50 秒的 EmitRuntimePlan 只是失败前的工作，不能算作完成导出性能。

实验工具 [restore-scheduled-payloads.cpp](restore-scheduled-payloads.cpp) 从原计划提取 Encode 输出 ID 与 payload，核对完整计划和 manifest 摘要，再逐项校验 blob 内容摘要并恢复 DenseElementsAttr。其他 IR 字节逐行复制，不重建 ID、调度、Release/reuse 或 boot 放置。这是实验输入准备工具，runtime 仍拒绝 `.cst` 索引。已用 splat 和非 splat float64 的小型计划核对恢复前后 plan、manifest 和内容摘要。

[恢复结果](restore-scheduled-payloads.json)记录了 1,019,527 个 Encode，计划提取 181.23 秒、常量恢复 256.41 秒、峰值 RSS 492.8 MiB。恢复后的 IR 为 40,864,182,764 字节，其中非 splat 常量采用 float64 原始字节的十六进制 DenseElementsAttr。该文件大于原诊断 IR；它的读取、解析时间和内存应计入实验进程开销，分别与 EmitRuntimePlan 时间、进入导出阶段时的 RSS 报告。

导出命令与 128 GiB 地址空间限制在 [run-export.py](run-export.py)，完整记录比对、同编译选项加载和独立 blob 摘要检查在 [run-validation.py](run-validation.py)。后者以导出器关闭所有文件并完成发布后打印的完整报告为入口，检查 writer 与 reader 的计划字节数、摘要和记录数量一致。完整编译器的诊断 IR 输出和销毁可能仍在继续；`-o /dev/null` 只避免诊断文件落盘，仍有 IR 序列化成本。全进程时间单独报告。

记录比对、加载测量和独立 blob 检查三组工作并行。原 JSON 和紧凑 JSON 的 O2 加载依次进行，但与其他验收负载重叠，属于共享负载下的单次观测，不能视为受控速度对照。独立 blob 检查只保留 manifest DOM 和单个 64 KiB 数据块，不计入 reader 的性能结果。版本、二进制摘要、编译选项和缓存条件见 [provenance.json](provenance.json)。

### 完整四卡 V2 流式导出

[导出分项](stage-c-gpu4-export.json)覆盖 10,669,278 个 values、21,336,108 条指令，以及 784,750 个唯一 blob。计划为 4,891,577,376 字节，manifest 为 82,675,385 字节，blob 原始载荷仍为 14,701,969,408 字节。计划的新源摘要为 `b65320a51e773161ea5d395c6f93704afd77abad87b8f4f4418a91602c457404`。

| 项目 | 测量 |
| --- | ---: |
| EmitRuntimePlan 完整导出 | 423.56 秒 |
| 布局 / 编号 | 16.13 秒 |
| 常量转换 / SHA-256 / 写入 | 97.84 / 80.62 / 47.09 秒 |
| plan 记录序列化 | 72.59 秒 |
| plan SHA-256 / 写入 | 26.37 / 4.17 秒 |
| manifest | 1.61 秒 |
| 进入导出 / 完成导出 RSS | 60.93 / 61.84 GiB |
| 报告时进程峰值 RSS | 61.90 GiB |

`build_seconds` 为包含布局、常量处理和记录写出的 421.86 秒；`plan_serialize_seconds` 包含序列化触发的缓冲区哈希和写入。这些分项存在包含关系，不能相加为总时间。RSS 包含完整 MLIR 与常量；导出起止增加约 0.92 GiB，不能把整个进程的 61.90 GiB 标成 plan DOM 或纯 writer 内存。

原有编译记录的 EmitRuntimePlan 为 537.06 秒；本次为同一调度与计划语义恢复常量后的独立导出。DenseElementsAttr 的恢复表示与原编译不同，缓存和共享服务器负载未控制，因此这两次观测不能单独证明 JSON 改动带来的速度比例。

[全进程报告](stage-c-gpu4-export-process.json)为 931.17 秒、峰值 RSS 65.22 GiB、退出码 0。它包括 40.86 GB 恢复 IR 的读取与解析、pass 验证、诊断 IR 序列化和清理；该时间不能与原编译的 JSON 导出子阶段比较。[完整日志](stage-c-gpu4-export.log)保留导出器的原始计时。新旧 manifest 的 784,750 个内容摘要、长度及根元数据逐项一致，见 [manifest 比对](stage-c-manifest-equivalence.json)。

## 复现

```bash
cmake -S third_party/ckks-runtime -B build-io -DCMAKE_BUILD_TYPE=Release
cmake --build build-io --target runtime_plan_io_benchmark -j 4
python3 scripts/extract_runtime_plan_io_samples.py OLD_PRETTY_PLAN DATA_DIR/samples
for n in 100000 200000 400000 800000; do
  for run in 1 2 3; do
    build-io/runtime_plan_io_benchmark --dom DATA_DIR/samples/execution-${n}.json
  done
done
build-io/runtime_plan_io_benchmark --plan PLAN_JSON OPERATOR_SPEC_JSON
```

片段工具有意接受基线 `indent=2` 导出布局，遇到其他格式直接报错。大文件与构建目录位于远端 `/home/xuming/poseidon-runtime-io-20261010/`，本目录只保存小型结果。
