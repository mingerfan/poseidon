# 完整 Qwen JSON 转二进制与读回实验

2026-10-11，188Server / HPU-001，Xeon E5-2650 v4，GCC 11.4，`-O2 -DNDEBUG`，单线程、nice 10。这次直接使用既有完整 V3 Qwen JSON，没有复制 MLP 样本、重新编译模型或处理权重载荷。

**完整计划加载从 278.42 秒降到 22.15 秒，快 12.6 倍；计入同一 PlanVerifier 从 312.00 秒降到 54.43 秒，快 5.7 倍。** 全部 11,064,667 个描述符和 22,149,239 条指令逐字段比较一致。

后续按用户要求关闭二进制文件 SHA，优化生产 Verifier，并加入实验多线程读取：8 线程加载 5.84–5.97 秒、加载加校验 13.40–15.24 秒，见[后续记录](../runtime-binary-parallel-188-20261011/README.md)。本文件保留上轮含 SHA 的原始对照数据。

## 单次完整测量结果

| 项目 | JSON | 二进制 |
| --- | ---: | ---: |
| 文件字节数 | 8,756,201,335 | 1,469,778,349 |
| 文件大小，十进制 GB | 8.756 | 1.470 |
| 文件读取 | 1.904 秒 | 0.352 秒 |
| 原始计划元数据 SHA-256 | 55.897 秒 | 9.433 秒 |
| 解析及 RuntimePlan 构建 | 220.622 秒 | 12.362 秒 |
| 加载合计 | 278.424 秒 | 22.147 秒 |
| 同一 PlanVerifier | 33.580 秒 | 32.279 秒 |
| 加载加 PlanVerifier | 312.003 秒 | 54.427 秒 |
| 加载阶段进程峰值 RSS | 5.689 GiB | 5.507 GiB |
| 含 PlanVerifier 的进程峰值 RSS | 8.734 GiB | 8.734 GiB |

文件减少 83.2%，解析与构建快 17.8 倍。JSON 源是之前已有的带缩进完整 V3 文件，因此体积收益同时包含去掉空白、字段名和文本 ID；没有另测完整紧凑 V3 JSON。两种格式均解码为同一 RuntimePlan，PlanVerifier 时间和长期内存相近。两者产生的能力/密钥数量均为 4 / 42,566。

JSON 基线来自成功转换进程的首次加载；二进制列来自另起的独立进程。共享服务器缓存未控制，两种格式各以此完整基线运行一次，结果是实测观测，不是中位数或冷启动带宽。

原始数据见[转换和读回](conversion-roundtrip.json)、[独立二进制加载](binary-load.json)、[汇总](summary.json)和[输入配置](inputs.json)。

## 一次性转换与正确性

JSON 加载完成后，二进制写出用了 12.386 秒；首次读回 21.065 秒，逐字段对比 1.003 秒，`all_fields_equal=true`。纯 JSON 加载加写出约 290.81 秒；包括源 PlanVerifier、读回与比较共 346.46 秒。这个一次性转换成本与以后直接读取二进制的 22.15 秒分开报告，不是编译器生成耗时。

读回比较阶段峰值 RSS 为 11.329 GiB，含保留的原 JSON RuntimePlan 及 allocator 保留量，不能与表中的单计划加载 RSS 比较。没有重新生成多 GB JSON 或进行记录哈希对比。

七份小型 fixture 往返通过，见[self-test-188.json](self-test-188.json)；真实 V3 MLP 的新命令检查见[mlp-smoke-188.json](mlp-smoke-188.json)。完整源计划和独立二进制计划均通过同一 PlanVerifier。

## 输入与方法

源文件为 `/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-plaintext-streaming/qwen24._hecate_qwen25_24layer.runtime-plan.json`，8,756,201,335 字节，11,064,667 个 ValueDesc、22,149,239 条 execution 指令，initialization/finalization 为空。

实验工具版本为 ckks-runtime `9dbd4e5`。使用此前 `PLNEXP01` 实验编码，仅将单 typed 数组限制从 2 GiB 提高到 8 GiB，支持完整 execution 数组。文件编码没有变化，生产 reader/导出器/执行入口没有切换格式。

[run-experiment.py](run-experiment.py)依次执行：

1. `convert-plan-check` 用严格 JSON reader 加载一次完整 JSON，记录加载基线，并执行原有 PlanVerifier。
2. 将该 RuntimePlan 写成二进制，再读回同一 typed 结构，逐字段比较根元数据、全部描述符、分阶段指令及属性、输入输出、bundle 引用。inline double 按位比较；比较不重新计算内容摘要，不构造完整 JSON DOM。
3. 另起进程 `plan-binary`，测量二进制加载和同一 PlanVerifier，取得单计划进程的加载/校验内存。

每个进程限制 32 GiB 地址空间、900 秒墙钟时间。成功运行中 JSON 完整加载一次，没有清空共享服务器页缓存。二进制测量在转换及读回之后，缓存条件未控制，不能标为冷启动。

第一次错误地沿用小型 MLP 的 OperatorSpec，完整 JSON 加载 274.47 秒后，Verifier 因 id/version 不匹配而退出，二进制尚未写出。见[失败日志](failed-first-attempt.log)。修正为原完整 Qwen 编译/验收使用的[OperatorSpec](operator-spec.json)后重新运行；因此本次作业累计完整 JSON 加载两次。性能表使用成功运行自己的 JSON 基线。脚本现要求显式指定 OperatorSpec，失败和成功目录各自保留。

加载计时包含正常 reader 的文件读取、原始计划元数据 SHA-256、解析及 RuntimePlan 构建；PlanVerifier 另计。权重 blob、manifest、runtime 任务构建、设备初始化、GPU/CKKS 执行均不在本次范围内。**权重载荷读取及哈希校验次数为零。**

读回逐字段对比期间同时保留 JSON 和二进制的 RuntimePlan，该阶段 RSS 不能用作单计划内存比较。另起进程的结果用于最终性能表。

## 复现

工具通过 `CKKS_RUNTIME_BUILD_EXPERIMENTS=ON` 的 `runtime_binary_io_experiment` 目标构建。七份 V1/V3 小样例先通过往返、字段比较及损坏输入检查，再运行完整文件；真实 V3 MLP 还通过 `convert-plan-check` 冒烟检查。实验 CTest 通过。

```bash
python3 run-full-qwen-binary.py /home/xuming/poseidon-runtime-io-20261010 \
  /home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-plaintext-streaming/qwen24._hecate_qwen25_24layer.runtime-plan.json \
  --operator-spec /home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/profiles/operator-spec.json \
  --work-dir /home/xuming/poseidon-runtime-io-20261010/full-qwen-binary-corrected-20261011
```

结果及二进制保存在远端 `full-qwen-binary-corrected-20261011/`，脚本拒绝覆盖已有 `qwen24.v3.plan.bin`。重复实验用 `--work-dir` 指定新目录。仓库只保存脚本、工具源码和小型结果。

该目录同时保存计时可执行文件及 codec/tool 源码快照 `*-measured*`。实验不包含完整 Qwen 执行，加载加速不能当作 GPU/CKKS 推理加速。生产入口的格式接入仍需独立实现。
