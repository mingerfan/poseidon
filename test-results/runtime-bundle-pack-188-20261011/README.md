# Bundle pack 与内存驻留验收

2026-10-11，在 188Server / HPU-001 上复用上一轮完整四卡 Qwen 的紧凑 V2 计划和 V1 bundle。不重编译模型、不重新加载全量计划、不逐 blob 重算哈希。

实现提交：ckks-runtime `720c740`（读取器、合并工具、预算与测试），DaCapo `c7344fb`（直接 pack 导出与兼容选项）。两者分别 push 到 `integration/source`、`integration/multi`。

原始 JSON 的 31.03 MB/s 是 8,427,772,779 字节除以 271.59 秒的端到端加载结果。读取、解析与构建、SHA-256 分别为 1.82、213.89、55.88 秒，见[原始 O2 结果](../runtime-plan-io-188-20261010/stage-c-original-load-o2.json)。读取只占约 0.7%；JSON 解析和对象分配占约 78.8%。这个速度不能用于估计 `.bin` 的读取带宽。

## 真实产物合并

[转换结果](pack-conversion.json)与[进度](pack-conversion.log)：

| 项目 | 结果 |
| --- | ---: |
| 原始唯一 blob 数 | 784,750 |
| 原始载荷 / pack 字节数 | 14,701,969,408 |
| 合并后 bundle 文件数 | 2：manifest.json、data.bin |
| 合并载荷、生成 manifest | 46.86 秒 |
| 复制 4.89 GB 计划并替换 manifest 引用 | 46.89 秒 |
| 重新计算 blob 哈希数 | 0 |

合并工具按原 manifest 顺序复制已有文件，检查长度，生成 offset。只对 manifest 元数据计算摘要。新计划由原计划逐字节复制，仅修改同长度的 manifest 引用；原产物保留。

## 一次 raw 驻留读取

新 C++ reader 使用 GCC 11.4、Release、`-O2 -DNDEBUG`、nice 10、16 GiB raw 预算，读取存储 V2 pack。[测量结果](pack-resident-read.json)：

| 项目 | 结果 |
| --- | ---: |
| raw 预载量 | 14.70 GB / 13.69 GiB |
| raw 预载时间 | 9.61 秒 |
| raw 预载速度 | 1.53 GB/s，十进制 |
| manifest 解析、摘要与索引、范围检查 | 3.49 秒 |
| open 总计 | 13.09 秒 |
| 进程峰值 RSS | 13.84 GiB |
| 解码 / 校验的 blob 数 | 0 / 0 |

raw 时间包含内存分配、顺序读取和关闭文件。此测量紧接合并之后，共享服务器缓存未控制，不标作冷磁盘速度。它证明 raw pack 能以大块连续读取进入内存，不代表完整 RuntimePlan 启动、逐 blob float64 解码、RNS Encode 或 GPU 执行时间，也不是与旧小文件 reader 的受控速度对照。

默认 reader 保持一个 pack 文件句柄，通过 offset 读取；驻留模式一次加载 raw pack，之后从内存范围取数据。实际执行按需读取某个 blob 时仍检查内容和有限 float64 值，预载阶段不做全量 blob 哈希扫描。

## 逐文件读取与 pack 的一次对照

随后用同一个小型 C++ 程序比较两种布局，均将同一份 14,701,969,408 字节载荷读入未初始化的连续 raw 缓冲。索引准备在计时外；计时包含缓冲分配、路径和文件长度访问、open/read/EOF 检查/close。逐文件模式按原 manifest 顺序访问 784,750 个 `.bin`，与转换后 pack 中的内容顺序一致；pack 模式以 8 MiB 块连续读取。两种模式均不解码、不计算 blob 哈希。

| 模式 | raw 读取时间 | 十进制吞吐 |
| --- | ---: | ---: |
| 逐文件加载 | 18.97 秒 | 775 MB/s |
| pack 连续加载 | 10.56 秒 | 1,392 MB/s |

本次 pack 吞吐为逐文件的 **1.80 倍**，读取时间减少 **44.3%**；两者进程峰值 RSS 均约 13.75 GiB。[逐文件结果](raw-read-files.json)、[pack 结果](raw-read-pack.json)、[C++ 程序](raw-io-comparison.cpp)和[运行脚本](run-raw-comparison.py)保留供复现。命令在远端工作目录执行 `python3 run-raw-comparison.py`，编译采用 `g++ -std=c++17 -O2 -DNDEBUG`，两个测量依次以 nice 10 运行，先 files 后 pack，仅各一次。

这是共享服务器、缓存未控制的单次 raw I/O 对照；不是完整旧/新 runtime 的启动对照。逐文件模式也将数据放入同样大小的 resident 缓冲，以隔离文件组织的影响；真实旧 reader 的单 blob 临时分配、解码、摘要与 Encode 未纳入测量。缓存和磁盘状态变化时比例会变化，不能将 1.80 倍外推为全模型或推理加速。驻留后的访问无需磁盘读取，但仍有按需解码和 Encode 成本。

## 命令与产物位置

远端工作目录 `/home/xuming/poseidon-runtime-io-20261010`：

```bash
python3 src/tools/pack_plaintext_bundle.py \
  artifacts/gpu4/qwen24._hecate_qwen25_24layer.bundle \
  artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.bundle \
  --plan artifacts/gpu4/qwen24._hecate_qwen25_24layer.runtime-plan.json \
  --output-plan artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.runtime-plan.json \
  --progress

build-o2/runtime_bundle_io_benchmark \
  artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.bundle \
  runtime-plan-2514-_hecate_qwen25_24layer-plaintexts 1 \
  sha256:2162179dbadf6e5bd977827943a15d33e80f9f24c1b9dde6b54cb1d5b6f17296 \
  17179869184
```

合并工具拒绝已有输出，复现需换新路径。reader 和工具使用本次提交代码；远端构建只编译 runtime core 和 bundle 实验工具，不启用 GPU 或模型编译依赖。

## 小型回归

ckks-runtime 6/6 CTest 通过，涵盖旧/新布局 slot 等价、共享 reader 并发 offset 读取、raw 驻留、V3 在线 Encode/Fence、预算与范围错误、长度错误，以及转换失败清理和已有输出保护。导出端检查 pack/旧布局等价、确定性、去重、损坏既有 pack 拒绝和失败清理。CPU/GPU 执行入口支持 raw 驻留预算环境变量。

Poseidon CPU API 7 项测试通过，CPU/GPU MLP 执行入口重新构建通过。真实 GPU MLP 使用已有无 Boot 校准 V1 计划，将 100 个 blob 合并为 1,280,000 字节 pack，显式提供 16 MiB raw 预算；fixture 和 mock 均在既有容差内，见[结果](gpu-mlp.json)和[日志](gpu-mlp.log)。命令采用 `POSEIDON_GPU_MLP_ALLOW_NO_BOOT=1`、`POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary`，模型、spec、fixture 来自 `test-results/e2e-new-profile-e63ec18/`。该 V1 计划完成 eager load 后释放 raw 缓冲，因此报告中的长期 resident 字节数为零；V3 持续驻留与 Fence 已由小型 runtime 测试覆盖。此结果不表示完整 Qwen GPU 执行通过。
