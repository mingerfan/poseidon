# 正式二进制 RuntimePlan：编译器、runtime 与完整 Qwen 加载

2026-10-11。正式入口已接入，不再仅限实验工具。DaCapo 默认生成二进制指令和 manifest；需要查看指令时显式选择 JSON。runtime 整文件读入后并行构建 typed plan，默认 4 线程，不采用流式流水线。

## 完整 Qwen，188Server / HPU-001

复用上一轮已逐字段核验的实验二进制，将容器标识改为正式格式，并切换到二进制 manifest 引用。**没有重新解析 8.76 GB JSON、没有重编译完整 Qwen、没有重新扫描 blob 哈希**。正式读取入口 `RuntimePlanReader`、`PlanVerifier` 和 `PlaintextBundleLoader` 由 `runtime_plan_io_benchmark` 调用。

- 11,064,667 个描述符、22,149,239 条 V3 execution 指令；init/final 为空。
- 指令文件 1,469,778,284 字节；manifest 37,668,171 字节、784,750 个索引项。
- 4/8 线程各测 3 次，1 线程测 1 次；32 GiB 地址空间预算，nice 10，单进程 180 秒超时。
- 共享服务器缓存未清空，不能当作严格冷读结果。所有新计划文件加载 `hash_seconds=0`，未计算整文件 SHA；普通索引测试没有读取权重 payload。
- 新旧二进制的所有描述符、指令、输入输出、目标字段一致。bundle 的存储格式引用按设计变化，ID 和版本仍相同；见 [字段比较](field-comparison.json)。

| 路径 | 指令加载 | PlanVerifier | manifest 索引 | 三项合计 |
| --- | ---: | ---: | ---: | ---: |
| 原 JSON，历史一次测量 | 275.03 s | 33.52 s | 7.99 s | 316.55 s |
| 正式二进制，1 线程 | 10.48 s | 6.85 s | 1.15 s | 18.47 s |
| 正式二进制，4 线程，中位数 | 6.55 s | 7.62 s | 1.18 s | 15.35 s |
| 正式二进制，8 线程，中位数 | 5.76 s | 7.80 s | 1.18 s | 14.74 s |

合计的中位数按每次完整结果计算，因此不一定等于各阶段中位数之和。4 线程合计范围 15.18–17.68 秒；8 线程 14.67–15.69 秒。4 线程相对历史三项合计快约 20.6 倍，包含格式、去掉全文件摘要及上一轮 Verifier 优化的共同收益，不能归因于多线程一项。

4 线程加载内部约 0.94 秒读取、3.65 秒扫描/数组初始化、1.84 秒并行解码；8 线程解码约 1.04 秒。单线程同一内存游标解码约 5.83 秒。计划仍完整驻留，临时文件缓冲约 1.37 GiB，构建后释放。普通加载进程峰值约 6.87 GiB。

### 同时预载全部 raw 权重

额外一次 4 线程测试启用 16 GiB raw 预算，将既有 `data.bin` 的 **14,701,969,408 字节** 顺序读入内存，不哈希、不解码 CKKS/RNS。

- raw pack 预载：10.29 秒，约 1.43 GB/s。
- 指令加载 + Verifier + manifest + 全量 raw pack 驻留：**25.70 秒**。
- 进程峰值：19.60 GiB。

上述合计不包括 runtime 的描述符/任务索引构建、密钥生成、CKKS 编码、GPU 上传和计算。没有宣称完整 Qwen GPU 执行通过。原 JSON 历史基线见 [V3 原始结果](../runtime-plan-io-188-20261010/stage-c-v3-load-verify-o2.json)。

## 编译器与执行验证

[编译器 MLP 对照](compiler-mlp.json)使用同一已调度 V3 IR（619 个描述符、1,239 条指令），分别直接导出 JSON 与二进制。所有指令字段和所有 manifest 字段一致；二进制计划 89,992 字节、manifest 4,955 字节，JSON 计划 259,705 字节、manifest 12,340 字节。重复导出字节一致；临时文件无法打开时保留原入口并清理 staging。

- [DaCapo 10/10 回归测试](compiler-ctest.log)：包含默认二进制头部、metadata、版本和失败发布测试。
- [runtime 7/7 测试](runtime-ctest.log)：旧 JSON、生命周期、Release/reuse/Fence、并行解码、错误版本/截断/尾部垃圾/非法索引范围，以及 binary V3 的实际 mock 结果。
- [Poseidon CPU API 及 2/4 rank MPI 测试](cpu-api-ctest.log)：3 个 CTest 入口全部通过。
- [真实 GPU MLP](gpu-mlp.json)与[日志](gpu-mlp.log)：本地 RTX 4060 Laptop GPU，编译器直接生成的 binary V3 和 binary manifest，16 MiB raw 预算，现有 fixture/mock 容差内；122 次 transfer，零 Boot，最终最大误差约 0.524794。输入来自既有校准 MLP，与本轮完整 Qwen 的 metadata benchmark 分开。

编译器在首次导出常量时仍计算 content ID，用于去重；这没有增加一次验证性 payload 扫描。binary 加载不重算指令、manifest 或 payload 的整文件摘要。OperatorSpec 保留小文件摘要；MPI 只比较小 metadata 的身份摘要，不保证逐指令字节相等。格式说明见 [RuntimePlan binary V1](../../third_party/ckks-runtime/docs/runtime-plan/binary-v1.md)。

## 使用

```sh
# 默认生成二进制；也可显式指定
hecate-opt ... --runtime-plan-format=binary
# 需要读指令时生成 JSON
hecate-opt ... --runtime-plan-format=json --runtime-plan-pretty=true
# 所有正式执行入口均可接收 .runtime-plan.bin 或 .runtime-plan.json
CKKS_RUNTIME_PLAN_READ_THREADS=8 poseidon_runtime_gpu_mlp_e2e PLAN.bin SPEC BUNDLE FIXTURE MOCK REPORT
```

`emit-runtime-plan` pass 使用 `plan-format=binary|json`；二进制要求 pack、禁用 pretty。两种模式使用不同输出 prefix。容器头：8 字节 `CKKSPL01` / `CKKSMF01`、u32 容器版本、u32 metadata 长度、metadata JSON；没有字面的 `METADATA` 标识，位置和长度由头部明确指定。所有数值字段为显式 little-endian。

复现脚本：[`benchmark_runtime_binary_production.py`](../../scripts/benchmark_runtime_binary_production.py) 和 [`test_runtime_binary_export.py`](../../scripts/test_runtime_binary_export.py)。远端产物保留在 `/home/xuming/poseidon-runtime-io-20261010/production-binary-20261011/`；14.70 GB raw pack 通过软链接复用，仓库只保存结果和日志。
