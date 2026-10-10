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
