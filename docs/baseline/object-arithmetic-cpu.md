# 对象数组逐元素算术：正式契约与 CPU 密态正反例

后续更新：v20 十例定向真实 Agent 批次与批量接线现已完成，首次8/10、修复后10/10，
13次API调用。详见[独立付费验收报告](object-arithmetic-agent.md)。本文以下保留
人工golden阶段的原始记录，不将当时的零API记录改写为Agent测试。

## 状态

本轮补上之前明确缺失的数组级 +、-、* 及 whole-array/slice +=、-=、*=，
并正式接入 request v20 / AST v19 / construction metadata schema 14。
单例命令为 `run_candidate.py --object-arithmetic ...`。旧 v19 请求、
旧 AST v18 以及上一轮 30 例/117 项冻结清单不变。

本轮没有付费 API 调用。下面是人工 golden 的真实 Dacapo/SEAL CPU 证据，
不是 Agent 生成结果。新 v20 仍需定向真实 Agent 验收和批量模式接线；
完整项目 DSL 目标仍未完成。

## 上下游位置与语义依据

- 上游对象存储：`object_arrays.py` 中 ObjectArray 保存符号 Expr/Plain、
  Empty 和公开值；它不是一个密文的 slot 数组。
- 构图求值：`function_construction.normalize(..., object_arithmetic=True)`
  先按公开对象数组 shape 广播，再逐单元调用既有标量同态算术展开器。
- 下游：展开后的 flat Hecate AST 仍由既有 validator 检查，经真实 Hecate、
  Earth/CKKS lowering、HEVM/CST、SEAL CPU 执行；没有新增后端。
- 实际上游 `third_party/dacapo/python/poly/poly/Func.py` 的 HE_ReLU/HE_SiLU
  使用标量加法和乘法组合；当传入对象数组时需要该类逐元素语义。但本轮
  **没有完整验证这些 helpers**：其多项式配置和 bootstrap 仍有独立门禁。

本轮显式保留：

1. NumPy 广播对齐的是公开存储维度。例如 shape (2,1) 的密文对象数组与
   shape (2,) 的公开系数相乘，得到 (2,2) 个存储单元，不是改变每个密文的 packing。
2. 普通二元算术创建新存储；原地赋值修改共享存储，不扩大左侧 shape。
3. 重叠切片在执行原地运算前快照左右操作数；同一轮操作不使用刚更新的单元。
4. target/index 求值一次且先于 RHS；取出的切片是 live view，会看到 RHS 内的写入。
5. 0-D object ndarray 的普通二元运算返回标量；原地操作仍返回原数组对象。
6. 对象数组 + Expr 走 NumPy 逐元素派发；Expr + 含 Expr 的对象数组走上游
   resolveType -> Plain 的整体 float64 转换并失败，不发明对称广播。
7. Empty - Expr 返回 Expr 本身，绝不是负号；未初始化 None 不参与算术。
8. rank <=4、对象存储 <=128、展开 <=256 个密文算术语句等既有资源边界保持。
   广播结果尺寸在分配结果之前检查；128x1 与 1x128 不会生成 16384 单元。

直接 Plain 与数组的歧义路径仍明确拒绝，公开数值数组应显式构造；
对象数组 unary/division/power/matmul/reduction 等未在本轮启用。
这些是完整目标中的剩余项，不从“全部语义”的验收范围删除。

实现不调用含候选对象的 NumPy ufunc，也不执行候选 Python：
只有内部验证过的存储用于广播/索引，单元运算回调由可信解释器提供。

## 测试与失败诊断

14 个新增离线测试覆盖数值 NumPy 广播形状矩阵（包括 0-D/空维度）、
重叠切片、别名、shape 不能扩大、分配前资源检查、实际 Expr/Plain 派发、
Empty 符号、旧契约拒绝，以及新 request/CLI/provider 序列化。
另有 1 个显式启用的真实证据测试，检查下面六次运行。

初轮 57 项中一个测试失败：原有记录型 Plain 替身没有真实 float64 转换，
因此未拒绝 Expr + object-array。这不是后端通过或实现成功的证据。
测试改为加载可信仓库中实际 Plain 构造器，仅 C ABI 为记录替身，随后通过。
不把替身测试称作真实密态执行。

最终 25 个相关模块共 **344 项：325 通过、19 条件跳过、0 失败**。
其中新真实证据测试通过，不在跳过项中。
当前代码重新审计上一轮付费记录，仍为 **30/30、117/117**；
没有重新调用 API 或修改旧报告。

## 六次真实密态执行

四个正确程序和两个有意写错的反例，每例 4 组输入、16 个输出：
合计 **24 组密态输入、96 个标量比较**。全部预期匹配。
错误反例也真实完成编译、加密执行和解密，最后才由数值比较拒绝。

| 程序 | 意义 | 最大绝对误差 | 预期/结果 |
|---|---|---:|---|
| broadcast | 二维对象广播公开系数 | 1.7522261641644832e-8 | 正例通过 |
| alias | 普通算术新存储 + 原地更新共享引用 | 1.6644579492464118e-8 | 正例通过 |
| overlap | 重叠切片 += 读取旧值 | 5.825572246820343e-9 | 正例通过 |
| empty | 对象数组 Empty -= Expr | 1.717563069547623e-8 | 正例通过 |
| wrong_alias | 用 copy 代替共享引用 | 0.5000000281929857 | 数值拒绝 |
| wrong_empty | 将 Empty - x 当成 -x | 3.0000000110321086 | 数值拒绝 |

reference 为固定四元素仿射 `y=1.5*x+0.375`；不声称已覆盖所有网络/shape。
门限保持 `atol=1e-5, rtol=1e-4`。参数保持 degree=32768、
modulus_bits=[60]*14、SEAL tc128 检查通过；没有降低安全参数或替换 bootstrap。
余弦相似度可能因浮点舍入出现 1+2e-16，原始数据保留，不以此替代逐项门限。

总报告：

`/home/lhy/poseidon-work/results/object-arithmetic-goldens-8rmo8tme/report.json`

SHA-256：

`58032259c03af43a77195554acfe2dd18de711bd3c78b858ee15b4831ea1028e`

六个单例目录分别为：

- `candidate-replay-4kwgjl8o`
- `candidate-replay-qjnz0y15`
- `candidate-replay-r5bgjxeq`
- `candidate-replay-_t60klgy`
- `candidate-replay-ovfx_z6y`
- `candidate-replay-1iwqzph_`

均位于 `/home/lhy/poseidon-work/results`。报告保留不可变请求/权重/输入、
源码哈希、真实 trace、Earth/CKKS、HEVM/CST、解密数组及逐项误差。
离线证据测试逐个核对哈希、重新 normalize、核对真实 trace payload、
重算 reference、artifact gate 和数值比较；不依赖“passed”字符串作唯一证据。

各轮自动清理合计 **4,073,880,060 bytes（约 3.79 GiB）** 测试密钥；
旧随机密钥不可恢复，但可重新生成。保留编译产物与数值证据，API key 未改动。

## 复现命令

真实密态正反例（不付费；会重新生成并在每例结束后清理测试密钥）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 1950s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_object_arithmetic_goldens.py
```

在现有固定 Nix/Python 环境中离线复核：

```bash
cd '/mnt/d/Code Space/Poseidon/scripts/baseline'
export POSEIDON_OBJECT_ARITHMETIC_REPORT=/home/lhy/poseidon-work/results/object-arithmetic-goldens-8rmo8tme/report.json
LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 \
  /home/lhy/poseidon-work/venvs/hecate-2.0.1-cpu/bin/python \
  -m unittest test_object_arithmetic -v
```

没有安装依赖、sudo、修改驱动、运行 GPU、多 GPU或完整 ResNet。
所有修改保留本地，未 commit/push。

## 下一门禁

- 新契约的批量与逐项真实 Agent 生成覆盖；不沿用旧 v19 117 项成功结论。
- 继续核验未实现对象算术、Plain/array 派发、一般常量/packing 和真实 helper。
- 完整 bootstrap/upscale 及 Poseidon GPU 验证仍是独立未完成项。
