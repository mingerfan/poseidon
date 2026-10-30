# Qwen2.5 0.5B 两层 Decode DSL — R121

这是 **round_121_paper_batch4_full_ckks 完整 CPU CKKS 验收通过**的 DSL 源码与验证证据包。模型来自 `round_120_down256_short`；两个独立 Decoder Block、两个已知输入 token，包含最终 RMSNorm、完整词表 LM head 和两层 KV 输出。介绍以本 README 为准；冻结目录中 qwen24 等名称和旧说明保留来源，不表示已验证 24 层。

## 结构和输入输出

- hidden=896，intermediate=4864，14 个 Q head、2 个 KV head，head_dim=64，词表151936；每密文32768槽，环度65536。
- 包含 RMSNorm、QKV、RoPE、因果 Attention（第二 token 包含 Softmax 近似）、O 投影、残差、SwiGLU 和 LM head。首 token 只有一个可见 key，可消除不用的 Q/Softmax 计算。
- 输入是两个 embedding 密文：每个前896槽有效，其余为零。客户端明文查 embedding 后加密；权重是明文常量。
- 输出仍为密文：5 个分块 logits（最后一块20864槽有效）和2个层 KV（各前512槽按 K0,V0,K1,V1 排列）。客户端解密后采样。模型末端不对 logits 做 Softmax。
- 本入口只返回末 token 的 logits 及两层 KV；编译器可裁剪不影响这些输出的分支。完整验收指这些两层模型输出全部正确，不表示每个 token 的所有无用分支也都保留执行。
- 测试使用合成权重、两个已知 token；这不是未知未来 token 的提前生成，不证明真实预训练权重、长序列、完整24层或 GPU 精度。

## 完整实测与对比

| 指标 | 已发布 R071 | 上一完整基线 R112 | 本版 R121 |
|---|---:|---:|---:|
| 自举次数 | 154 | 74 | 74 |
| Runtime 秒 | 8114.139 | 3318.470 | 2740.815 |
| 峰值 RSS GiB | 28.988235 | 27.075573 | 27.079472 |
| 全保留计划系数 GiB（静态） | 285.345703 | 152.402832 | 141.609375 |
| 释放图峰值 MiB（静态） | 203.500 | 288.000 | 290.500 |

本版实部最大误差 0.000432636673021，虚部 0.000459204414763，无效尾部 0.000397642710881，均低于统一门槛0.001。每列为不同密钥、不同时间的单次完整测试，不能当作隔离的线程或单算子性能归因。

完整计划全保留系数 141.609375 GiB，释放图峰值 290.500 MiB。它们不含全部密钥、自举内部工作区和分配器开销，均不是实测 RSS；实测采用释放中间值模式。详见 [数据量、level、槽位与内存分析](MODEL_DATA_ANALYSIS.md)、[完整验收](evidence/FINAL_ANALYSIS.md) 和 [与 R112 的内存/level 对比](evidence/MEMORY_LEVEL_COMPARISON.md)。

## 优化与编译器

包含宽矩阵/MLP 打包、已知 token 的分组计算、QKV tile128、首 query 的独立可裁剪路径和干净 KV 输出；配对路径的 MLP down tile=256。终端多余刷新消除、release_ready 主机调度及64MiB明文 Encode 根缓存是额外的 CPU 计划后处理，不能只上传算子后声称保留全部内存收益。

本版使用隔离的实验 Hecate/Dacapo 编译器，修正 normalize 在 factor==waterline 时的自举段预算，**不是宣称仓库官方最新版**。通过的编译器 SHA256：`12705f29938821d48ae3db3def7767af7a2af6507d40fb0c7e9bfdb98918de68`。补丁、183份源码哈希和12组对照回归证据见 `compiler_reference/`、`evidence/selected-short-compiler-*`。需使用匹配源码和运行库；不能对任意新版盲目套补丁。

CPU 使用 R078 selective pool reclaim 后端、R080 native NAF 旋转 API、32线程；真实48阶自举，输入L1→输出L5、scale43。运行 API 不持有私钥、没有解密再加密。完整测试不保留短测即时诊断副本。

## 使用

Python3.10+、NumPy、匹配构建的 Hecate 前端与 hecate-opt；Runtime 和 Poseidon 自举库需由目标机器提供。此包不含权重数组、密钥或编译后二进制库。

```bash
python -B verify_package.py
python -m pip install -r requirements.txt
# 下列生成完整尺寸合成权重，约1.2GB；可替换为符合 schema 的自有权重。
python -B prepare_fixture.py --output-dir /data/qwen-fixture
python -B trace_model.py --weights /data/qwen-fixture/weights --output-dir /data/qwen-traced
python -B compile_model.py --hecate-opt /path/to/hecate-opt --input /data/qwen-traced/ACTUAL.mlirbc --output-dir /data/qwen-compiled
python -B optimize_cpu_plan.py --input /data/qwen-compiled/ACTUAL.runtime-plan.json --output /data/qwen-plan.json --report /data/qwen-plan-audit.json
/path/to/verify_runtime_plan /data/qwen-plan.json frozen/model/package/config/operator-spec.json
```

ACTUAL 文件名以 Hecate 输出为准；每次使用新目录。`trace_model.py --dry-run`、`compile_model.py --dry-run` 可只查看命令。合成生成器的新结果应自行做完整明文/CKKS验收，不能继承历史结果哈希。独立完整 Mock 后再运行 CPU native probe：

```bash
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE /path/to/native-probe /data/qwen-plan.json frozen/model/package/config/operator-spec.json /path/to/constant-bundle /data/qwen-fixture/runtime-fixture.json /path/to/cpu_bootstrap_reference.json /data/result.json
```

`frozen/tools/native_probe.cpp` 是已验收完整探针原文件。构建需匹配 Runtime API、Poseidon 头文件/库、native Boot 回调与参数；本包不是独立二进制发行版。常量 bundle 必须随计划保留，不能只复制 JSON。

GPU 侧复用 DSL，按 GPU 合约重编译，`compile_model.py` 支持 `--compiler-config`、`--operator-spec`、`--device-count` 和 `--boot-profile`。`optimize_cpu_plan.py` 只接受单机 CPU 计划；GPU 的调度、释放、缓存和自举需由 GPU 工具链接入并重新验证。CPU 的精度、次数、RSS 和耗时均不是 GPU 保证。

## 文件与来源

`frozen/model/` 原字节保留模型，`frozen/tools/` 保留 CPU 后处理和探针，`compiler_reference/` 保存编译器补丁与绑定，`evidence/` 保存实际验收及分析。`source-binding.json` 记录来源，`manifest.json` 校验交付包。`delivery-config.json` 和 `trace_model.py` 固定真正通过的开关，避免继承旧追踪入口默认值。
