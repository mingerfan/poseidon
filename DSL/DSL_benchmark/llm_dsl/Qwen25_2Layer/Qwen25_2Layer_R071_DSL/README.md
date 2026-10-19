# Qwen2.5 0.5B 两层 Decode DSL — Round 071

这是 **071 完整 CPU CKKS 测试通过版本**的源码交付包：060 冻结 DSL、070 并行 CPU 后端、32 线程。包含两个独立 Decoder Block、两个 token 顺序 decode、最后一个 token 的完整词表 logits，以及两个层的 KV 缓存。

## 模型与验证范围

| 项目 | 内容 |
|---|---|
| 网络层数 / token 数 | 2 个真实尺寸 Decoder Block / 2 token（位置 0、1） |
| hidden / intermediate | 896 / 4864 |
| Q heads / KV heads / head dimension | 14 / 2 / 64 |
| 词表 | 151936 |
| 包含计算 | RMSNorm、Q/K/V、RoPE、因果 Attention（含 Softmax 近似）、输出投影、SwiGLU/MLP、残差、最终 RMSNorm、LM head |
| 非线性策略 | `degree10p2_pair13_silu_canonical_fold2` |
| 加密槽位 | 每密文 32768 个，CKKS 多项式度数 65536 |
| 已验证 | 合成权重、两层两 token、CPU 原生真实自举完整执行 |
| 尚未验证 | 该包 GPU 执行、预训练权重精度、完整 24 层、长序列 |

单个可见 token 的 Attention 无需实际执行 Softmax；第二个 token 使用两位置的近似 Attention。只输出末 token logits 时，编译器可能裁剪不影响最终输出及 KV 的计算。因此“两层两 token”不意味着每个 token 的所有无用分支都保留执行。

## 输入与输出

- **输入 2 个密文**：客户端先明文查 embedding 表，再将两个 896 维向量分别放在密文前 896 槽，其余槽清零并加密。不是把加密 token ID 直接交给模型查询词表。
- 模型权重是明文常量，计算时按需要编码；本包不包含预训练权重和密钥。
- **输出 7 个密文**：前 5 个拼接为 151936 维原始 logits（末块有效 20864 槽）；后 2 个分别是各层 KV，每个前 512 槽按 `K0,V0,K1,V1` 排列，每段 128 维。
- 服务器计算输出仍为密文；客户端用私钥解密，再选 token。末端不对 logits 做 Softmax。评估程序中的解密仅用于误差检查。

## 已完成的 071 测试

| 指标 | 实测 |
|---|---:|
| 原生真实自举 | 154 / 154 |
| Runtime 计算耗时 | 8114.139 秒（2.254 小时） |
| worker 全流程 | 8209.263 秒 |
| CPU 峰值 RSS | 31125880832 字节（28.988235 GiB） |
| 最大实部误差 | 0.000701730252421 |
| 最大虚部 | 0.000680512914213 |
| 最大无效尾部误差 | 0.000657849893599 |
| 验收阈值 | 0.001 |

逐输出结果、计时范围及内存分析见 [完整报告](evidence/FINAL_ANALYSIS.md)，槽位布局见 [槽位报告](evidence/SLOT_UTILIZATION_REPORT.md)。这些是历史 CPU 实测，不是此次打包重跑的结果，也不是 GPU 显存/耗时结论。

## 内存与自举优化

1. 全槽位 LM head、批量 GQA 和每层一个打包 KV；最终只保留 7 个输出。
2. 官方编译器插入真实 `native Boot`；CPU 合约输入 L5，自举 L1→L5、scale=43，使用旧 48 阶 CPU 核心，无 decrypt-reencrypt。
3. 官方显式 `Release` + 本地已验证的 host DFS 排序 + **64 MiB 编码明文根缓存**，缓存淘汰后需要时重新编码，算术和常量不变。
4. `FinalOnly` 本身不等于释放内存，必须执行 Release。071 没有启用后续 078 的空闲池主动回收，也不是 081/082 候选版。

图数据释放峰值系数约 **203.5 MiB**，不包含计算密钥和自举工作区。全保留明密文系数约 **285.346 GiB**，只是静态载荷，不是实测 RSS；28.988 GiB 实测对应释放模式。64 MiB 只限制 Encode 根缓存，不是整个进程内存。

## 目录说明

```text
trace_071.py                 固定 071 策略和两 token 的 DSL 追踪入口
compile_071.py               官方自动自举编译入口，显式 Release、二进制 IR
optimize_cpu_plan.py         CPU 专用 DFS + 64 MiB 缓存后处理
prepare_fixture_071.py       合成权重、输入和独立明文参考生成
verify_package.py           SHA256、Python 语法和历史验收证据检查
frozen/model/               按历史 manifest 原字节复制的模型与算子
frozen/tools/               原字节保留的计划排序、CPU/Mock 探针与小模型验证源码
cpu_backend_reference/      070 后端补丁、基底绑定及使用限制
evidence/                   071 完整精度、内存、level、slot 及源代码绑定证据
```

`frozen/model/package` 内的 `qwen24`、`compile_qwen24.py` 是继承的模块名；当前配置明确为 **2 层**。其中旧 README、provenance 和接口布尔标志属于历史来源记录，以本 README、071 completion-verification 和实际输出为准。不要直接用旧 `trace_combined_model.py` 的默认策略；请使用 `trace_071.py`。

## 运行环境与命令

Python 3.10+、NumPy，及兼容的 Hecate/Dacapo 前端和编译器。CPU 测试锁定 Dacapo `39c1fb255adddbfb9d7180f9d1b8535a31241bff`、ckks-runtime `f7e9115eaec22e44c8e12ff9f53528e421691688`；不是声称它们是当前最新分支。Hecate 需来自匹配构建，按构建说明设置 Python 路径及 `HECATE_BUILD`。执行 RuntimePlan 还需要匹配的 Poseidon Runtime、自举 API 和参数。

以下均在本包目录执行。路径按目标机器修改，每次使用新的输出目录。

```bash
python -B verify_package.py
python -m pip install -r requirements.txt
# 仅在需要生成数据时执行：完整尺寸合成权重约 1.2 GB。
python -B prepare_fixture_071.py --output-dir /data/qwen071-fixture
python -B trace_071.py --weights /data/qwen071-fixture/weights --output-dir /data/qwen071-traced
# --input 指向 trace 输出的实际 MLIR/bytecode 文件。
python -B compile_071.py --hecate-opt /path/to/hecate-opt --input /data/qwen071-traced/model_entry.mlir --output-dir /data/qwen071-compiled
```

`model_entry.mlir` 只是示例文件名，请以 Hecate 实际导出为准；二进制 bytecode 也可使用 `.mlir` 后缀。`compile_071.py --dry-run` 可以检查命令和合约而不执行编译。

### CPU 内存优化与执行

```bash
python -B optimize_cpu_plan.py --input /data/qwen071-compiled/ACTUAL.runtime-plan.json --output /data/qwen071-cache64.json --report /data/qwen071-memory-audit.json
/path/to/verify_runtime_plan /data/qwen071-cache64.json frozen/model/package/config/operator-spec.json
# 在已配置 native Boot 回调、匹配核心和运行库并构建 native_probe.cpp 后：
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE /path/to/native-probe /data/qwen071-cache64.json frozen/model/package/config/operator-spec.json /path/to/constant-bundle /data/qwen071-fixture/runtime-fixture.json /path/to/cpu_bootstrap_reference.json /data/qwen071-result.json
```

`constant-bundle` 是编译产物中 manifest 所在目录；优化后的计划继续引用原 bundle，不能仅拷贝 plan JSON。先通过 PlanVerifier 和完整 Mock，再做 CKKS。探针源码与 CLI 已保留，但外部 CPU 核心、隔离回调、头文件、静态库和 bootstrap 参数不在本包内；详见 [CPU 后端说明](cpu_backend_reference/README.md)。本包是可供集成的 DSL 源码与证据包，不是包含运行库的一键二进制发行版。

### GPU 接入

重用 `trace_071.py` 和冻结算子；编译时提供 GPU 对应的 **compiler config、operator spec、device count**，例如在上面 compile 命令中增加 `--compiler-config /path/gpu-compiler.json --operator-spec /path/gpu-spec.json --device-count 1 --boot-profile PROFILE_ID`。不要直接执行 CPU 专用 `optimize_cpu_plan.py` 于 GPU 计划，它显式要求 host placement。

GPU 后端需要提供同等数学语义的真实自举，并支持相应 level/scale、旋转、Release 和输出布局；GPU 的自举次数、内存、速度与误差需重新测量，不能把 CPU 的 154 次或精度作为 GPU 保证。若新工具链接口变化，先按该分支匹配合约和前端，再做逐层/完整输出对照。

## 来源与完整性

`source-binding.json` 逐文件记录本地 071 冻结来源及 SHA256；`manifest.json` 覆盖交付文件。原始结果 SHA256 与 completion-verification 绑定；原 CPU 执行计划 SHA256 为 `f671cbd15f88ace2d0f97745fcfe8bb533d3fb3eccda2c1ab09d401cd986a4cf`。重新生成、不同编译器或不同常量产生的计划必须重新验证，不能继承这个哈希的验收结论。

此次交付不包含权重数组、大型编译中间文件、私钥、连接凭据或服务器控制脚本。后续两层版本请在上级 `Qwen25_2Layer/` 中新增独立版本目录并更新索引。
