# Qwen2.5 0.5B：两层、两 token Decode DSL（自动自举编译）

本包交付当前上海 CPU 后台测试对应的 **两个独立 Decoder Block、两个 token 顺序 decode** 源码，以及 Dacapo 自动插入真实自举的编译配置。源码从已编译、已完成全图 MockVec 对照的冻结实验直接取回，并逐文件核验 SHA-256。

**完整编译和明文模拟通过；完整两层 CKKS 精度仍在后台验证，GPU 精度尚未验证。** 权重是可重复生成的合成权重。本包不是完整 24 层预训练模型。

## 1. 模型内容和接口

```text
两个 token 的密文 embedding
  → 每个 token 顺序通过两个 Decoder Block，复用第一个 token 的密文 KV
  → 最后 token 的最终 RMSNorm → 完整 LM head
  → 最后 token 的全部 logits + 两个 block 的全部 KV

每个 Decoder Block：
RMSNorm → Q/K/V → RoPE → 因果 GQA（含 Softmax）→ O 投影 → 残差
        → RMSNorm → Gate/Up → SiLU 与逐元素乘法 → Down → 残差
```

| 项目 | 配置 |
|---|---|
| Block / token 数 | 2 / 2，位置 0、1 |
| hidden / FFN | 896 / 4864 |
| Query / KV heads / head dimension | 14 / 2 / 64 |
| 槽位数 | 32768，N=65536 |
| 输入 | 两个密文，前 896 槽为 embedding；测试尾部 896、897 槽为 0.3、-0.2 |
| 权重 | 明文；Linear 和 LM head 使用优化的打包实现 |
| 输出 | 157 个密文：149 个 logits 分块，随后 8 个 KV 分块 |
| logits 顺序 | 词表从 0 开始，每块 1024 槽，末块 384 槽，共 151936 维 |
| KV 顺序 | block 0 再 block 1；每层依次 K0、K1、V0、V1，每块有效前 128 槽 |
| 当前 CPU 计划的输入 / 输出 level | L5 / L2，scale_log2=43 |

精确接口由 `model_entry.signature('full')` 定义，追踪时输出 `signature.json`。本入口将两个 decode 步展开在同一图内，不是跨两次外部调用传递 KV 的服务接口。输入是已准备的 embedding，不接收加密 token ID；计算输出保持密文，误差检查时才解密。

只要求最后 token 的 logits，因此编译器可裁掉第一个 token 在末个 block 中不影响所需输出的算术，同时保留 KV 路径。

## 2. 文件入口

- `trace_decode.py`：调用冻结入口，生成 Hecate 原生 MLIR bytecode。
- `compile_decode.py`：使用 `--dacapo --boot-placement=depth-dp`，配置 native Boot、明文 bundle 和全中间值保留。
- `prepare_fixture.py`：生成当前两层合成权重、输入与独立原函数参考。
- `source/Qwen25_2Layer_Decode_Test/model_entry.py`：两个 block、两个 token 的实际模型组合与输出接口。
- `source/Qwen_DSL_Memory_Study/packed_linear.py`、`packed_lm_head.py`：当前优化算子。
- `source/Qwen25_24Layer_DSL/qwen24/`：共享的模型、基础算子、非线性、权重读取和参考实现。
- `config/`：当前 CPU 编译器配置与 OperatorSpec，按服务器原字节保存。
- `evidence/`：完整编译、MockVec、level 和容量记录。
- `source-binding.json`、`manifest.json`：服务器来源与包文件哈希。

`source` 下保留原目录名，是为了使服务器冻结源码及其相对导入逐字节一致。实际入口将 `num_layers` 设为 2，并取前两层近似配置。使用本包根目录入口；共享目录中的 24 层工具和说明不改变本包的两层测试范围。

## 3. 依赖与生成数据

Python >=3.10、NumPy；追踪另需已构建的 Hecate Python/native frontend。原生编译使用团队现有的 `hecate-opt`。

已验证的版本：

| 组件 | Commit |
|---|---|
| Dacapo | `39c1fb255adddbfb9d7180f9d1b8535a31241bff` |
| ckks-runtime | `f7e9115eaec22e44c8e12ff9f53528e421691688` |

应使用 Poseidon 主分支锁定的子模块版本；原生前端与编译器保持匹配。上海构建仅有四处 `llvm::dyn_cast` 的 GCC/LLVM18 命名空间兼容修正，未替换官方 depth-dp 规划逻辑。

在本包根目录运行：

```bash
python -B verify_manifest.py
python -B prepare_fixture.py --output-dir fixture
```

生成器使用种子 20261009、逐权重名称稳定派生随机序列，生成两个不同 block。包括 embedding 与 LM head 的权重数组合计 1208379904 字节，约 1.13 GiB；Git 包不包含大权重，也不下载模型。`fixture/reference.npz` 使用原始 sqrt、SiLU、Softmax 数学函数作为参考；`input_slots.npy` 的无效尾部与当前 CKKS 探针一致。

## 4. 追踪与带自举编译

先配置已构建的 Hecate，例如将 `HECATE` 指向 Dacapo 源码、`HECATE_BUILD` 指向相应构建目录，并配置该环境需要的 Python 和动态库搜索路径。随后执行：

```bash
python -B trace_decode.py --weights fixture/weights --scope full --output-dir traced
python -B compile_decode.py --hecate-opt /path/to/build/bin/hecate-opt \
  --input traced/model_entry.mlirbc --output-dir compiled_cpu
```

所有输出目录须为新目录。实际编译命令和日志保存在输出目录中；生成 RuntimePlan、CKKS IR 和明文 bundle。加 `--dry-run` 可检查命令而不执行编译。

**自举由编译器插入，Python DSL 表达模型算术。** 当前配置启用 native Boot，计划中的全部 604 次自举均为 L1 → L5、scale_log2=43。无需在每个 Python 算子里手写 bootstrap。

默认配置是已验证编译的 **CPU 合约**。其中 OperatorSpec 的 `status=placeholder`、Boot `latency_us=0` 是接口占位，不是实测性能；声明支持 Boot 也不能代替后端实现。当前上海执行通过隔离回调绑定真实 CPU 自举，无 decrypt-reencrypt；本包交付 DSL 和编译配置，不包含该 CPU 测试程序或运行库补丁。

### GPU 端

GPU 团队应使用与其实现一致的编译配置、OperatorSpec、native 自举 profile 和实际 device count，重新编译同一份 DSL。例如：

```bash
python -B compile_decode.py --hecate-opt /path/to/build/bin/hecate-opt \
  --input traced/model_entry.mlirbc --output-dir compiled_gpu \
  --compiler-config /path/to/gpu-compiler-config.json \
  --operator-spec /path/to/gpu-operator-spec.json \
  --boot-profile YOUR_NATIVE_BOOT_PROFILE --device-count 1
```

脚本从 OperatorSpec 读取 target/context/spec ID 并计算 SHA-256；两份 GPU 合约需一起提供。GPU 自举接口和精度需由对应后端验证，不能将默认 CPU 模数、scale 和 OperatorSpec 直接改名当作 GPU 配置。若 GPU 改变槽位容量，则需要同步修改打包并重新追踪。

## 5. 当前已验证结果

| 检查 | 当前结果 |
|---|---|
| 全图原生追踪、Dacapo 编译、PlanVerifier | 通过 |
| 全图原生 MockVec 对照独立原函数参考 | 通过；最大误差 `4.471264475336412e-8`，尾部 0 |
| 编译计划 | 99117 个值，604 次 native Boot，157 个输出 |
| 全部明文系数载荷 | 16739467264 字节，约 15.590 GiB |
| 全保留明密文系数载荷 | 379144634368 字节，约 353.106 GiB；非实测 RSS/显存 |
| 完整两层 CPU CKKS | 2026-10-13 发布时仍在后台验证，尚无最终精度结果 |
| GPU CKKS | 尚未验证 |

当前 RuntimePlan SHA-256：

```text
b43f7b8b307353923d0b42ff993ce8b908d09e0d9eebd180e1fa0c2e8d4fd96f
```

编译采用全中间值保留，不插入 release 或输入复用；运行时还要采用匹配的 `AllValuesAfterRun` 策略才能复现该统计口径。实际运行需要额外容纳评估密钥、自举工作区及分配开销。近似参数目前对应合成输入范围，更换为预训练权重或更长序列后需要重新校准与验证。
