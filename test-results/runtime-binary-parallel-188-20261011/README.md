# 完整 Qwen：去掉指令文件 SHA、优化 Verifier、并行构建 RuntimePlan

2026-10-11，188Server / HPU-001，Xeon E5-2650 v4，GCC 11.4，`-O2 -DNDEBUG`，nice 10。直接复用已有的 1,469,778,349 字节 V3 二进制文件，包含 11,064,667 个描述符、22,149,239 条指令。

按用户要求，二进制计划读取不再计算整文件 SHA-256。没有重新解析原始 8.76 GB JSON、转换计划、编译模型或访问权重。所有新加载测量均为 `hash_seconds=0`、`source_hashing=false`、`blob_payloads_read=0`。

**Verifier 从 32.28 秒降到 6.98 秒，快约 4.6 倍；8 线程加载为 5.84–5.97 秒，加载加校验为 13.40–15.24 秒。** 原二进制包含 SHA 的加载加旧 Verifier 为 54.43 秒。

## 测量结果

| 读取方式 | 加载 | Verifier | 加载加校验 |
| --- | ---: | ---: | ---: |
| 上轮流式、有 SHA、旧 Verifier，单次 | 22.147 秒 | 32.279 秒 | 54.427 秒 |
| 本轮流式、无 SHA、新 Verifier，单次 | 13.210 秒 | 6.977 秒 | 20.187 秒 |
| 游标，1 线程，两次范围 | 10.725–10.812 秒 | 6.963–7.003 秒 | 17.688–17.814 秒 |
| 游标，2 线程，两次范围 | 8.169–8.216 秒 | 7.306–7.314 秒 | 15.483–15.521 秒 |
| 游标，4 线程，两次范围 | 6.709–6.717 秒 | 7.456–7.501 秒 | 14.166–14.219 秒 |
| 游标，8 线程，两次范围 | 5.841–5.965 秒 | 7.554–9.275 秒 | 13.395–15.240 秒 |

原始结果见[loads.json](loads.json)、[完整字段比较](field-comparison.json)、[输入配置](inputs.json)，[summary.json](summary.json)由[summarize.py](summarize.py)生成。旧数据来自[上轮完整二进制记录](../runtime-binary-qwen-188-20261011/binary-load.json)。上轮与本轮缓存和机器负载未控制，不能将多个独立改动的收益全归给多线程。

同一游标路径内，填充对象阶段的两次中位数从 1 线程 6.002 秒降至 2 线程 3.431 秒、4 线程 1.938 秒、8 线程 1.098 秒，约快 5.5 倍；整体加载由 10.768 秒降至 5.903 秒，约快 1.8 倍。仍有约 3.74 秒扫描/字符串表/数组初始化，以及约 0.92 秒输入缓冲读取，限制整体倍率。

8 线程第二轮 Verifier 为 9.275 秒，另一次 7.554 秒；没有把最快总耗时当作稳定承诺。4 线程总耗时约 14.2 秒且本轮更稳定，8 线程加载更快；目前两档均值得保留。

流式无 SHA 加载峰值 RSS 5.507 GiB，含新 Verifier 6.273 GiB；原旧 Verifier 为 8.734 GiB。游标各档加载峰值为 6.874–6.876 GiB，包含临时的 1.37 GiB 文件缓冲。全字段比较同时持有两份计划及输入缓冲，峰值 12.297 GiB，不能当作单计划内存。

## 代码与方法

- ckks-runtime `2b7cc3a` 优化生产 PlanVerifier：连续状态数组替代多份值状态哈希表，先确认 ID 按序连续才直接索引；稀疏 uint64 或乱序 ID 使用一个预分配索引。模数预算改为前缀和，错误文字只在失败时构造，计算输入改用固定小数组，单目标 Transfer 不再分配 set 节点。全部定义/使用、Release/reuse、设备、算子及最终输出检查继续执行。
- ckks-runtime `0f42346` 加入不计算摘要的实验读取和多线程解码。旧 `PLNEXP01` 字节格式不变，已有产物直接使用；不依赖新编译器。实验工具的 `plan-binary` 是关闭 SHA 的原流式读取，`plan-binary-parallel` 将整个文件读入内存，用游标扫描记录边界，以每块 16,384 条记录分配线程，直接填入互不重叠的现有 RuntimePlan 对象。共享字符串表只读，字段校验复用原解码器，异常原样传播。
- 因为格式没有块目录，扫描、字符串表读取和顶层数组初始化仍串行。此阶段记作 `scan_allocate_seconds`，填充对象记作 `decode_seconds`。原始输入缓冲在读取返回时释放；峰值 RSS 包含缓冲，不代表长期多出一份驻留文件。
- [run-experiment.py](run-experiment.py)先比较完整流式与 8 线程读取结果，再测一次流式基线；游标读取按 1/2/4/8、8/4/2/1 顺序各运行两次。每个进程限制 32 GiB 地址空间、180 秒墙钟时间。共享服务器缓存未控制，不是冷启动测试。

Verifier 是生产代码改动，二进制及并行入口仍是实验工具；生产 JSON 读取入口未切换格式，其默认 source digest 行为未改变。本次结果仅覆盖计划加载和 Verifier，不包含任务构建、设备初始化、manifest 或 GPU/CKKS 执行。

## 正确性

完整流式与 8 线程读取逐字段一致，没有做记录哈希比较或重新生成 JSON。全部格式版本、目标、bundle 引用、输入输出、11,064,667 个描述符和 22,149,239 条指令及属性被比较。每次加载均通过新 Verifier，能力/密钥数量仍是 4 / 42,566。

本地 7/7 CTest 通过，包括既有 Release/reuse、非法计划和运行测试。新 Verifier 用例覆盖按序连续 ID、乱序描述符、稀疏 ID、uint64 最大值及重复 ID；能力和密钥需求保持一致。二进制小型 V1/V3 样例单线程/4 线程读回一致，截断、尾部垃圾、未知 magic、非法描述符枚举和指令 tag 均被拒绝；线程内错误正常返回调用方。

## 复现

```bash
python3 run-no-sha-parallel.py /home/xuming/poseidon-runtime-io-20261010 \
  --operator-spec /home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/profiles/operator-spec.json

# 直接测某一线程数
build-o2/runtime_binary_io_experiment plan-binary-parallel \
  full-qwen-binary-corrected-20261011/qwen24.v3.plan.bin \
  /home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/profiles/operator-spec.json 8
```

远端结果目录为 `/home/xuming/poseidon-runtime-io-20261010/no-sha-parallel-20261011/`。脚本拒绝已有目录，重复实验需更换目录。该目录保存计时可执行文件及源码快照 `*-measured*`；没有计算这些工具的 SHA。仓库保存小型原始结果和汇总脚本。
