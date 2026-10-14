# RuntimePlan 与 manifest 二进制小实验

2026-10-11，188Server / HPU-001，Xeon E5-2650 v4、GCC 11.4、`-O2 -DNDEBUG`、单线程、nice 10。每种输入/格式运行三次，JSON 与二进制按 J/B、B/J、J/B 交替。转换在测量前完成；共享服务器缓存未控制，结果不标作冷启动。

实验工具实现为 ckks-runtime `c0f83ff`，push 到 `integration/source`；此目录保存 Poseidon 的样本准备、测量与汇总记录。

**自定义二进制明显改善元数据加载：约 40 万条指令的加载快 10.1 倍，真实 manifest 加载快 4.2 倍。** 计时保留原始元数据 SHA-256、typed 对象构建和严格范围检查。整个实验不读取、不解码、不校验权重 blob。

后续已直接转换并读回完整 V3 Qwen，22,149,239 条指令及 11,064,667 个描述符全部字段一致，加载 278.42 → 22.15 秒，见[完整测量记录](../runtime-binary-qwen-188-20261011/README.md)。本文件保留小样本实验当时的结果及 2 GiB 数组限制；后续工具已提高单数组上限到 8 GiB，生产格式仍未切换。

## 输入范围

- 计划源为已有的 `build-plaintext-models/1gpu-stream._hecate_MLP.runtime-plan.json`，V3，619 个描述符、1,239 条指令。其中有 121 条在线 Encode、375 条 Compute、122 条 Transfer、616 条 Release、5 条 Fence，2 处 reuse_input。复制 81/324 份独立计算图，重新映射 ValueId、TransferId 和全局 ordinal，得到 100,359/401,436 条指令和 50,139/200,556 个描述符。JSON 与二进制使用完全相同的每份输入，均通过同一 PlanVerifier。它们是重复 MLP 的规模样本，**不是完整 Qwen 计划**。
- manifest 使用上一轮真实四卡 Qwen pack 的全部 784,750 个条目，原始文件为 98,562,074 字节。只处理索引，不访问 14.70 GB 的 `data.bin`。
- 文件大小为十进制 MB，RSS 为 MiB。

输入记录见 [inputs.jsonl](inputs.jsonl)，转换记录见 [conversions.json](conversions.json)，三次原始结果见 [loads.jsonl](loads.jsonl)，中位数及范围见 [summary.json](summary.json)。

## 中位数结果

| 输入 | JSON 大小 | 二进制大小 | JSON 加载 | 二进制加载 | 加载倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100,359 条指令、50,139 个描述符 | 21.61 MB | 6.64 MB | 1.008 秒 | 0.102 秒 | 9.9 倍 |
| 401,436 条指令、200,556 个描述符 | 87.44 MB | 26.53 MB | 4.031 秒 | 0.398 秒 | 10.1 倍 |
| 784,750 项真实 manifest | 98.56 MB | 37.67 MB | 3.460 秒 | 0.825 秒 | 4.2 倍 |

40 万条计划体积减少 69.7%，manifest 减少 61.8%。两档计划条数相差四倍，JSON 加载中位数约增加四倍，二进制约增加 3.9 倍；两者均没有出现原有回调解析的平方级扫描。

### 40 万条计划分项

| 阶段/指标 | JSON | 二进制 |
| --- | ---: | ---: |
| 文件读取 | 0.020 秒 | 0.006 秒 |
| 元数据 SHA-256 | 0.564 秒 | 0.171 秒 |
| 解析与 typed 构建 | 3.445 秒 | 0.221 秒 |
| 同一 PlanVerifier | 0.472 秒 | 0.472 秒 |
| 每次 load + verify 之和的中位数 | 4.502 秒 | 0.897 秒 |
| 加载阶段进程峰值 RSS 中位数 | 113.99 MiB | 102.38 MiB |
| 含 Verifier 的进程峰值 RSS | 约 168 MiB | 约 168 MiB |

解析与构建分项快约 15.6 倍；计入 SHA-256 后，加载快约 10.1 倍。较小文件同时减少了元数据摘要成本。二进制没有改变 typed plan 结构，也没有改变 Verifier 的工作，所以长期内存和校验耗时下降有限。load + verify 不包含 OperatorSpec 文件加载、runtime 索引和任务构建、设备初始化或执行。

每个字段单独取三次中位数，分项之和不必严格等于加载中位数。加载时间范围：10 万 JSON 1.005–1.032 秒、二进制 0.102–0.125 秒；40 万 JSON 4.017–4.040 秒、二进制 0.394–0.425 秒；manifest JSON 3.439–3.469 秒、二进制 0.818–0.832 秒。

## 编码为什么更快

计划采用显式小端整数 ID、操作码、属性和共享字符串表；每个 ValueDesc 的编码为 35 字节。正文没有重复 JSON 字段名，也不再解析十进制 ID 或构造逐记录 JSON DOM。数组有数量前缀，可以提前分配准确容量，直接填入现有 RuntimePlan。

manifest 条目固定为 48 字节：32 字节 content 摘要、8 字节 offset、8 字节长度。读回仍建立与 JSON 相同 key/value 类型的哈希索引，并检查重复内容、范围、重叠和覆盖；已知条数允许提前 reserve。少量根元数据保留 JSON，以复用现有检查，不属于主要开销。

这是用于判断是否值得工程化的原型，完整编码和限制见 ckks-runtime 的 [实验说明](../../third_party/ckks-runtime/docs/experiments/binary-plan-io.md)。没有将 C++ struct/variant 内存直接写入文件，没有新增序列化依赖。

## 正确性与边界

小型测试将二进制读回结果投影为 JSON，与源数据逐字段比较：六份 V1 fixture 覆盖 inline/bundle、Compute 属性、Transfer/Replicate、Boot；真实 V3 MLP 覆盖 Release/reuse/在线 Encode/Fence。七个样本读回一致，截断、尾部垃圾和未知 magic 均被拒绝，见 [self-test.json](self-test.json)。实验 CTest 通过。40 万条样本的两个格式均通过同一 PlanVerifier，能力和密钥集合数量一致。

同一读回测试在远端最终源码上通过，见 [self-test-188.json](self-test-188.json)。`local-*.json` 为本地 Ryzen AI 9 HX 370 上早期的小型 V3 功能检查，不参与上表中的 188Server 统计。

真实 manifest 的全部内容标识、offset、长度和根元数据读回一致，见转换记录中的 `entries_equal=true`。该检查不触及 blob，也没有计算 blob 哈希。

没有直接测量完整 2,000 多万条 Qwen 指令，没有执行 GPU/CKKS 运算，也没有将约 10 倍加载倍率外推为完整模型启动或推理速度。manifest 结果是全量真实索引的实测。

生产 DaCapo/Runtime/Poseidon 入口没有切换到这个实验格式。实验每个 typed 数组上限 2 GiB；稳定版本、节目录、摘要绑定、可配置总内存预算和完整不可信输入审计尚未完成。实验 plan 仍引用原 JSON manifest，二进制 manifest 单独测量，未绑定成新的生产 bundle 入口。

一次性转换 40 万条 JSON 先加载约 4.12 秒，再从已加载 typed plan 写二进制约 0.20 秒。这个成本不能算成编译器导出耗时，也没有测量流式 JSON 导出对照。若正式采用，DaCapo 可直接写二进制，避免每次先解析 JSON 再转换。

## 复现与代码

远端工作目录：`/home/xuming/poseidon-runtime-io-20261010`。小型工具通过 CMake 的 `runtime_binary_io_experiment` 目标构建；复制输入的脚本为 [prepare_binary_io_experiment.py](../../scripts/prepare_binary_io_experiment.py)。[run-experiment.py](run-experiment.py)执行一次准备/转换和三次交替测量，[summarize.py](summarize.py)计算中位数及范围。

```bash
# 在远端 root 下；原输入不会被修改，重复实验指定新的样本目录
python3 binary-io-experiment/run-experiment.py /home/xuming/poseidon-runtime-io-20261010 \
  --data-dir /home/xuming/poseidon-runtime-io-20261010/binary-io-experiment/data-repeat

# 本地汇总小型结果
python3 test-results/runtime-binary-io-20261011/summarize.py
```

准备脚本拒绝已有样本输出，重复实验需换新 data 目录。远端保存 JSON/二进制样本及原始 manifest；仓库只保存小型源码、测试和结果。计时二进制及其源码快照位于远端 `binary-io-experiment/*-measured*`；随后仅为测试入口增加可选 V3 样例参数和整理格式，读取/编码逻辑相同。

计时二进制 SHA-256：`972e073efb40f6d242648fc2dff6cbc0e5576cf1593051dade274b6c7bf5e8b6`。计时 codec 源码 SHA-256：`5e8423fa6326f7796fb15f6e74cdd653a5c138ea5261229dab730b7ec8d2c940`。这些是小型工具的版本标识，与权重 blob 哈希校验无关。
