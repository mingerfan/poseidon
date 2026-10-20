# Qwen2.5 0.5B 24 层 DSL：对应已发布的 071 两层版本

本工程从 **071 已上传两层包**扩展而来，保持 `degree10p2_pair13_silu_canonical_fold2` 非线性策略、Linear/LM head、Attention 打包和 CPU 内存优化算法。24 个 Block 使用独立权重和独立 KV，默认两个 token 顺序 decode。

**这是 24 层源码交付版，不是 24 层完整 CKKS 已验证版。** 当前检查包括来源哈希、结构、入口和小尺寸 24 层四条算术路径；没有执行真实宽度 24 层编译或 CKKS/GPU 长测。071 的 154 次自举、2.254 小时及 28.988 GiB 内存均只属于原两层结果，不能套到这里。

## 结构和输入输出

| 项目 | 配置 |
|---|---|
| Block 数 | 24 |
| hidden / intermediate | 896 / 4864 |
| Q / KV heads，head dimension | 14 / 2，64 |
| 词表 | 151936 |
| CKKS 槽位 / polynomial degree | 32768 / 65536 |
| 目前交付序列范围 | 最多 2 个可见 token；配置的位置上限 512 不代表已支持 512-token 推理 |
| decode-pair / 两 token prefill | 2 个输入密文，29 个输出密文 |
| 第二 token 外部缓存 decode | 1 个 embedding + 24 个 KV 输入密文，共 25 个输入；29 个输出 |

客户端先查明文 embedding 表，再加密每个 token 的 896 维向量（前 896 槽有效，尾部清零）。权重作为明文常量编码。输出前 5 个密文为最后一个 token 的 151936 维原始 logits；后 24 个为各层 KV，两个 token 时每个有效 512 槽，按 `K0,V0,K1,V1` 排列，各段 128 维。输入输出都是密文；解密与 token 选择由客户端处理，末端不额外对 logits 做 Softmax。

每个 Block 包含 RMSNorm、Q/K/V、RoPE、因果 Attention（含 Softmax 近似）、O 投影、残差、RMSNorm、SwiGLU/MLP、残差，最后包含 final RMSNorm 与完整 LM head。一个可见 token 不需要执行 Softmax。只请求末 token logits 时，编译器可裁剪不影响输出和 KV 的算术。

## 从 071 迁移了什么

- 所有 `qwen24/kernels` 算术文件、`packed_attention.py`、DFS/明文缓存算法与 071 原字节一致；详见 `source-binding.json`。
- 将模型检查和入口从 2 层扩展到 24 层；生成 291 个具名权重张量的 schema，24 个 profile 项、24 组独立权重和 KV。原两层 profile 完全相同，因此复制同一数值配置到 24 层；这不等于完成 24 层预训练模型校准。
- 保留 RMS degree10p2、pair13 Softmax、canonical_fold2 SiLU、原 packed-row Linear 与全槽 LM head。
- 保留官方 native Boot 编译合约、显式 Release、CPU DFS + 64 MiB Encode 根缓存；实际 24 层自举数量、level 分布和峰值须编译/运行后统计。
- 此版本不采用后续 079/082 的 wide_baby16、078 主动池回收或 080 native_naf 候选。CPU 32 线程来自执行后端设置，不是 DSL 自身保证。

## 使用方法

环境：Python 3.10+、NumPy、匹配的 Hecate/Dacapo 前端和编译器，以及目标 CPU/GPU Runtime。071 历史锁定 Dacapo `39c1fb255adddbfb9d7180f9d1b8535a31241bff`、ckks-runtime `f7e9115eaec22e44c8e12ff9f53528e421691688`。目标分支若使用其他版本，需重新验证兼容性，不要把新分支名字当作同一工具链。

```bash
python -B verify_package.py
python -B trace_model.py --describe
python -B trace_model.py --mode decode --describe
# 下条仅做小尺寸 24 层验证，需要 NumPy，不需要 Hecate 或 CKKS：
python -B validation/check_small_model.py --output-dir /path/to/new-small-check
# 以下是显式重计算入口，打包过程中没有运行：
python -B prepare_fixture.py --output-dir /data/new-qwen24-fixture
python -B trace_model.py --weights /data/new-qwen24-fixture/weights --output-dir /data/new-qwen24-trace
python -B compile_model.py --hecate-opt /path/to/hecate-opt --input /path/to/actual-traced.mlir --output-dir /data/new-qwen24-compiled
```

`--mode` 支持 `decode-pair`（默认）、`prefill`（2 token）、`prefill-one` 和 `decode`（位置 1，已有位置 0 的缓存）。`--describe` 不导入 Hecate，不执行模型。完整尺寸 fixture 生成约 2.52 GB float32 合成权重，并做独立 dense 明文参考；请在服务器上显式执行，不能把它当作快速检查。NPY 目录或 NPZ 的具名权重格式见 `source/model/package/qwen24/weights.py` 和 `config/weight_schema.json`，不会自动转换 Hugging Face 权重。

## CPU 计划的内存优化

```bash
python -B optimize_cpu_plan.py --input /path/to/ACTUAL.runtime-plan.json --output /data/new-qwen24-cache64.json --report /data/new-qwen24-memory.json
/path/to/verify_runtime_plan /data/new-qwen24-cache64.json source/model/package/config/operator-spec.json
```

后处理固定两输入、29 输出的 decode-pair/prefill 计划；外部缓存 decode 不能直接使用此入口。原常量 bundle 必须与优化后的计划一起保留。采用 Runtime `FinalOnly` 并执行 Release，64 MiB 是明文编码根缓存上限，不是进程内存上限。先检查完整 Mock、非线性输入区间和内存准入，再运行 CKKS。

CPU 合约用真实 native Boot，L1→L5、scale43；需要已接通的 CPU 自举回调及兼容核心，API 不应获取私钥。此包不含外部运行库、构建产物、自举密钥和权重。071 CPU 后端并行参考见同仓库 `../Qwen25_2Layer/Qwen25_2Layer_R071_DSL/cpu_backend_reference/`。

## GPU 接入和验证边界

使用同一 DSL，编译时提供 GPU 对应 `--compiler-config`、`--operator-spec`、`--device-count` 和必要的 `--boot-profile`。编译器将重新规划，GPU 自举数量和 level 不能预先写成 071 的数字。CPU 专用 DFS/cache 后处理要求 host placement，不能直接套到 GPU plan。

24 层的噪声与近似误差会累积；两层 CPU 通过不保证 24 层或 GPU 通过。小尺寸检查只证明被测浮点路径、24 次独立 Block 调用和缓存布局正确，不等于真实宽度编译、CKKS 精度或 GPU 显存验证。证据见 `evidence/validation.json`；`evidence/parent_071/` 明确只是父版本历史证据。

当前源码配置的 RMS 方差域 [0.5,2]、SiLU 域 [-2,2]、两位置 Attention 的 profile 只用于既定合成数据验收。真实模型需要逐层校准，不能直接把预训练权重投入后宣称精度有效。

## 与已有 24 层包的关系

本目录新增保存，原 `Qwen25_24Layer_DSL` 不覆盖。本版本是 **R071 的对应 24 层迁移**；另一个历史 24 层包的自举计数、测试中止情况和内存数字不作为本版本的结果。
