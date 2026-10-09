# Qwen2.5-0.5B：已通过 CPU CKKS 的 2 token 完整 decoder block DSL

本包交付 **单个真实尺寸 decoder block** 的 Hecate Python DSL，用于对接团队已有 GPU 编译/运行环境。
核心算术源码与 2026-09-22 上海服务器通过的版本逐文件 SHA-256 一致。

**验证范围：1 个 block、2 token、合成权重、prefill 和连续两步 decode。不是 24 层全模型，也没有包含 embedding、最终 RMSNorm、LM head 或文本生成。**
四个内核文件中的 `qwen_model.py` 保留完整模型类定义，但本包入口仅调用索引 0 的 `_decoder_block`，配置 `num_layers=1`。

## 1. 计算内容与已知结果

```text
密文 hidden
 → RMSNorm → Q/K/V 投影 → RoPE → 因果 GQA（包含近似 Softmax）→ O 投影
 → 加输入残差
 → RMSNorm → gate/up 投影 → SiLU(gate) × up → down 投影
 → 加第二条残差 → 密文 block hidden 输出
```

参数固定为 hidden=896、FFN=4864、Q heads=14、KV heads=2、head_dim=64、RoPE theta=1e6、RMS epsilon=1e-6。
输入是位置 **7、8** 的两个隐藏向量，不是 token ID 或文本。位置不能直接改成 0、1 而继续引用本包的参考输出。

| 路径 | 历史 CPU CKKS 最终最大误差 | 真实自举次数 |
| --- | ---: | ---: |
| prefill | 4.0365166993e-4 | 103 |
| decode-pair | 4.5542584525e-4 | 103 |

阈值为 1e-3；两路解密后实部输出最大差异 5.6914333426e-4。原 CPU 测试耗时约 9.98 小时，包含大量诊断解密，不能作为 GPU 性能基线。
历史误差统计使用复数槽值的绝对误差；保存的 `cpu_*_decrypted_real.npy` 只有实部，所以用这些 NPY 重算误差会略小，不会与上表完全相等。

本次新入口以 NumPy 槽运算重新验证真实尺寸：两路相对原函数误差均为 **1.5206731208e-8**，两路差异为 0，输出尾部为 0。
这验证了新入口接线和冻结算术，不是本次新增 GPU 或原生 Hecate 编译通过的结论。

## 2. 文件说明

| 路径 | 用途 |
| --- | --- |
| `trace_qwen_block.py` | 两种模式的 `hc.func` 跟踪和 `hc.save` 导出入口 |
| `qwen2token/block.py` | 单 block 调用、密文 KV 复用、输出拼接 |
| `qwen2token/kernels/` | 四个逐字节冻结的算术源文件；`expr.py` 适配现有 Hecate Expr/Plain |
| `config/model.json` | 冻结的 Qwen 尺寸 |
| `config/approximations.json` | 服务器通过版本的近似系数与区间，直接保存而不重新拟合 |
| `config/contract.json` | 槽数、输入输出和验收约定 |
| `config/cpu_bootstrap_reference.json` | 原 CPU 模数链和自举参数，供 GPU 后端对照 |
| `prepare_fixture.py` | 用原种子及生成顺序重建公开权重、输入及独立原函数参考 |
| `data/` | 小型冻结输入和原函数参考输出 |
| `check_gpu_outputs.py` | 对照 GPU 解密后的 NPY 输出，检查原函数误差、尾部和两路一致性 |
| `verification/` | 原 CPU 结果摘要、本次 NumPy 验证结果和可重跑脚本 |
| `provenance/` | 源码绑定及原始 CPU 构图/自举调度脚本归档 |
| `manifest.json` / `verify_manifest.py` | 包内文件 SHA-256 清单及校验工具 |

`provenance/original_tools` 是原始脚本归档，依赖旧工程的目录结构，不能从本包直接作为运行入口。
本包无需覆盖对方已有的 `hecate` Python 模块；私有命名空间 `qwen2token.kernels` 只导入其 `hecate.expr.Expr/Plain`。

## 3. 准备相同测试数据

在解压后的目录执行。准备数据和明文检查只需 Python >=3.10、NumPy；Hecate 跟踪另需对方现有的 Hecate 原生库及配套依赖。

```bash
python verify_manifest.py
python prepare_fixture.py --output-dir fixture
```

生成：

- `fixture/weights.npz`：公开 float64 合成权重，约 **119 MB**；含一个用于保持原随机数顺序的 final_norm 权重，此权重不进入 block 计算。
- `fixture/input_hidden.npy`：`[2,896]` 的明文隐藏向量。
- `fixture/input_slots.npy`：`[2,32768]`，每行对应一个待加密密文。
- `fixture/reference_hidden.npy`：`[2,896]` 独立原函数参考，含真正的 sqrt、SiLU、Softmax 数学计算。
- `fixture/fixture_manifest.json`：种子、数组哈希和范围检查。

随机种子为 20260922，严格先生成权重再生成输入。输入逐行归一化到均方值 1。生成器会与包内冻结输入和参考对照，不匹配时失败。
为保持包体积较小，没有把大权重放进 ZIP；不会下载真实模型权重。

## 4. 导出两条 DSL

先按对方已有工程配置好 Hecate（包括 `HECATE`、原生库路径和 Python 环境），再从本包根目录运行：

```bash
python trace_qwen_block.py --mode prefill --weights fixture/weights.npz --output-dir traced_prefill
python trace_qwen_block.py --mode decode-pair --weights fixture/weights.npz --output-dir traced_decode
```

两条命令分别启动进程；每个进程只注册自己的函数，避免 Hecate 全局函数列表混入另一条图。
入口名为 `_hecate_qwen_block_2tokens`，Python 函数名 `qwen_block_2tokens`。`hc.save` 保存原生前端产物，每个输出目录另含 `signature.json`。
生成目录要求不存在；已有结果不会被自动覆盖或清理。

### 两种模式的确切含义

- `prefill`：一次输入位置 7、8 的两个密文，执行因果 Attention。
- `decode-pair`：同一个静态图里先计算位置 7、生成密文 K/V，再计算位置 8 并直接复用该 K/V；最后拼接两个 token 的 block 输出。这与原来通过 CPU 测试的 decode 图一致。
- **本入口不是外部 KV cache 服务接口**：没有让 GPU 客户端在两次调用间传入/取出缓存，也没有单独认证跨调用缓存的序列化、层级或 scale 对齐。

## 5. GPU 输入输出约定

| 项目 | 约定 |
| --- | --- |
| 槽数 | 32768；原 CPU 多项式环维数 N=65536 |
| 输入 | 2 个密文，依次加密 `input_slots.npy` 第 0、1 行 |
| 有效输入 | 各自前 896 个槽位 |
| 输入尾部 | 896、897 号槽为 0.3、-0.2，其余为 0，复现原诊断的脏尾部探针 |
| 权重 | 公开编译期常量，不加密 |
| 输出 | 1 个密文，共 32768 槽 |
| 有效输出 | 0..895 为位置 7，896..1791 为位置 8 |
| 输出尾部 | 1792..32767 应接近 0 |
| KV | 全程保持密文表达式，不使用解密后重加密更新 |

解密发生在测试端，用于误差检查。模型计算输出本身仍是密文。本包不是隔离客户端私钥的部署服务。

GPU 运行后，分别保存两路完整解密槽值为 NPY，推荐 complex128；形状为 `[32768]` 或 `[1,32768]`：

```bash
python check_gpu_outputs.py --prefill gpu_prefill.npy --decode gpu_decode.npy
```

检查阈值均为 1e-3：两路对原函数参考的最大绝对误差、各自尾部误差、两路输出差异。
若只能导出实部，工具会明确标记未检查虚部；若只能导出 `[1792]` 或 `[2,896]` 有效值，可加 `--prefix-only`，但该结果不能代表尾部通过。
工具只检查提供的数组，GPU 执行来源需要对方保存实际运行日志、编译配置和代码版本。

## 6. 模数链与自举：后端需要处理的部分

此 DSL 使用冻结的加法、乘法和旋转算术，**没有在 Hecate 源码中显式插入 bootstrap**。
历史 CPU 测试在导出诊断 DAG 时使用独立公开调度器插入真实自举，每条路径 103 次，初始 level=8、自举恢复 level=5，最终 level=0。
这些数字属于那条 CPU 诊断调度，不要求 GPU 必须恰好执行 103 次。

GPU 编译/运行端必须提供可行的 scale/rescale/modswitch/重线性化与真实自举策略。只得到 MLIR 或图、或只完成编译，不能宣布完整 GPU 密文推理通过。
`config/cpu_bootstrap_reference.json` 与原调度源码供后端比较，不是可以直接喂给任意 GPU 编译器的配置文件，也不应把 CPU 实现的 level 编号直接套到不同后端。
原 CPU 测试启用了库内 `tc128` 参数检查；迁移后端需继续使用有效的安全参数与相应评估密钥。

## 7. 近似配置与范围

- 两次 RMSNorm：12 阶 Chebyshev 倒平方根，方差加 eps 的区间 `[0.5,2.0]`。
- SiLU：12 阶 Chebyshev，输入区间 `[-2,2]`，采用 blocked 求值。
- Softmax：减均值后的分数界为 0.5，exp 8 阶、倒数迭代 6 次，最大行长为 2。

系数直接来自已通过的服务器记录。不要使用旧 Attention/RMSNorm 单层源码包里的另一组配置替换它们。
当前权重和输入保证落在本组检查区间内；换成真实模型权重、任意文本的中间张量或增加 token 数，需要重新校准并验证。

## 8. 可选本地明文复核

```bash
python verification/validate_numpy.py --fixture-dir fixture --output-dir numpy_check
```

使用实际两个 tracing 入口和真实尺寸权重，在 NumPy 模拟的加/乘/旋转上检查接线、算术及零尾部。不会模拟 CKKS 噪声，也不会加载原生 Hecate 或调用 GPU。

本包是既有 Qwen block 的源码交付，不是 9 月 28 日新 Agent 框架已经验收的程序包。该 Agent 当前的 256 元素等限制仍需另行扩展。
