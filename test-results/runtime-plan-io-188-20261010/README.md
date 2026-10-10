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

### 全量导出的输入边界

原有四卡 `qwen24.mlir` 是诊断产物：部分 Encode 已被后续 `.cst` 导出改成整数索引。直接重新导出在第一处这样的 Encode 明确失败，所有 plan/bundle staging 文件被清理；[失败记录](diagnostic-export-failed-process.json)与[日志](diagnostic-export-failed.log)保留。该次 90.50 秒的 EmitRuntimePlan 只是失败前的工作，不能算作完成导出性能。

实验工具 [restore-scheduled-payloads.cpp](restore-scheduled-payloads.cpp) 从原计划提取 Encode 输出 ID 与 payload，核对完整计划和 manifest 摘要，再逐项校验 blob 内容摘要并恢复 DenseElementsAttr。其他 IR 字节逐行复制，不重建 ID、调度、Release/reuse 或 boot 放置。这是实验输入准备工具，runtime 仍拒绝 `.cst` 索引。已用 splat 和非 splat float64 的小型计划核对恢复前后 plan、manifest 和内容摘要。

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
