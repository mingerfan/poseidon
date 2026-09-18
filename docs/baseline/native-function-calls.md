# 原生 @hc.func 调用：实现、边界与真实 CPU 证据

## 结论与范围

已完成原生装饰函数调用的本地实现及人工验证：18项真实native边界测试通过；
13类直接表达/函数调用对照全部trace、编译通过；38组真实SEAL执行、544个
输出值通过独立PyTorch float64差分。最新代码重编译所得全部golden HEVM/CST
与实际执行过的对应产物字节一致。不是新增Agent付费测试或完整DSL完成声明。

Agent的多装饰函数入口仍关闭；原有普通Python构造helper保持可用。
本轮没有安装einops或其他依赖，没有付费API调用、GPU执行或bootstrap。
已有付费批次的精确外发/预算授权门禁仍保留，不将自动goal继续视为批准。

## 系统位置及实现

调用路径：人工Hecate Python `Func.__call__` → ctypes `createCall` → 已trace的
Earth单基本块IR静态内联 → 现有Earth/CKKS passes → HEVM/CST → stock SEAL_HEVM。

固定Dacapo commit：`4616402710f39df3e5f5bd7930a6c036025aaac3`。
上游Python原有`Func.__call__`调用`createCall`，但C++文件未实现该入口。
另外原`createFunc`不使用`inputTys`区分c/p。对应位置：

- `third_party/dacapo/python/hecate/hecate/expr.py`：ABI声明、Func输入/输出/状态。
- `third_party/dacapo/tools/frontend.cpp`：类型签名、插入点保存恢复、IRMapping及clone。

本地实现让callee Python只trace一次，调用处复制已验证IR；不是增加HEVM运行时
call指令，也不是把调用代码交给不受限exec。操作数与返回值通过IRMapping重绑定。
输入声明区分cipher/plain；调用检查参数数量、类型、值归属当前基本块。
结果保留标量、list、tuple、object ndarray的形状及C顺序，包括0-D和非连续view。
普通可迭代对象、任意控制流IR、递归装饰调用及kwargs调用尚未承诺支持。

`p`的本轮密态证据是helper形参绑定公开常量后内联到加密入口；不是证明
HEVM具有通用的独立明文入口ABI。数组shape在Python侧重构，HEVM仍是扁平结果序列。
代码规模随调用点内联增长，不能据此声明性能优化。

## 边界缺陷及修复

| 已复现问题 | 修复 |
|---|---|
| tracing外调用最终报错之前先执行helper及创建值 | 在参数转值和callee执行前检查活动trace |
| tracing失败后其他函数仍使用残留native上下文 | 整个上下文标记失败，后续eval/save拒绝，要求新进程 |
| 另一函数的Expr可传入当前函数，生成跨块引用 | native接口验证参数的parent block属于当前caller |
| helper异常被caller捕获后仍能完成损坏的IR | caller返回前检查嵌套失败标记；仍失败且清空Python活动栈 |

没有伪装事务回滚：失败后native插入点/IR可能残留，整个上下文不再可用。
前三项失败前报告 `native-call-boundaries-uiuros2m/report.json`：17项中14通过、3失败，
SHA256 `fec6701bb835a793565fd1e5bff1c00a0a76125bcf8f885e4d7ac01c4903325c`。
第四项单独unittest也在修复前以`ValueError not raised`失败（本轮控制台记录）。

最终18项每项在新进程调用真实libHecateFrontend，不使用mock ABI：
`native-call-boundaries-r13mb3o_/report.json`，
SHA256 `79ff802c00aeaa94bbe98339a015db4bc625d308c2eb463d67085a77a993abb1`。
包括多结果形状、fresh返回、trace-once、空结果、零输入、非法参数/返回、
直接及互相递归、失败状态、外部调用、跨函数值与C入口非法句柄/数量/容量。
这不是任意恶意C指针的内存安全验证；C接口仍为受信frontend内部接口。

## 编译与真实密态差分

13类：scalar、pair、nested、forward、two_inputs、public_argument、identity、
repeated、array、zero_dim、zero_input、empty_helper、plain_return。

旧对照报告 `native-function-calls-72gvr1e9/report.json` 的status为failed，
原因仅为当时工具把“产物非逐字节相同”当作失败：26次trace/compile均为exit 0，
12对byte-identical，array一对因常量和寄存器分配顺序不同而hash不同。
保留该报告，未修改历史结果。更新后的probe区分`compiled_nonidentical`和编译失败，
不会因为不同hash就判定语义错误，也不会因为编译通过就判定密态正确。

真实执行报告：
`/home/lhy/poseidon-work/results/seal-cpu-golden-native-calls-lqf5erw1/report.json`

SHA256 `8c5fc4479cd1c3432c5d6b5411b0275adbd2d90fcfaee2658a693ca0bcbb1ba6`。

- backend：existing upstream SEAL_HEVM CPU；13类、direct/call各执行。
- 固定输入：零值、有符号、种子4201/4202随机、范围边界；每组4个输入批次。
- 每个输入密文重复4-slot数据；pair/array按4个slot分别验证全部返回密文。
- 38次worker执行、152个输入批次、544个比较值，全部通过。
- 独立reference不调用helper、不使用解密中间结果；例如repeated参考用`2*x+.5`。
- 固定门限 `abs(actual-reference)<=1e-5+1e-4*abs(reference)`。
- 最大绝对误差 `2.8777190186346502e-8`；加权MAE `3.7831663588717155e-9`。
- 原有tc128参数：N=32768、16384 slots、14个60-bit模数，水线40。
- no bootstrap，无新backend、无安全参数或误差门限调整。
- 共用本次新生成的一套随机密钥；实验结束后按现有保留策略清理297283048 bytes
  （约283.5 MiB）。原随机密钥不可恢复，可重新生成；API凭据未读写。
- 请求/模型API完全未参与。IR、HEVM/CST、输入/reference、解密值、metadata和日志保留。

最后的异常路径修复后再次编译全部13类：
`native-function-calls-3a59ga67/report.json`，status=`compiled_nonidentical`，
SHA256 `d41d9525dd0f661e84bb299058bbc1948199c87eadddc1eee90ce63cab12f29a`。
独立回归将其每份golden产物与上述已执行版本逐字节核对，全部一致。
没有重复执行FHE来冒充更多独立模型；CPU runner随后仅把锁目录与既有
`cache/agent-native-slots`统一，不更改计算、参数或产物。

## 兼容性、复验和Git

当前源码下bootstrap容器6种trace/IR验证保持通过，仍不代表bootstrap真实执行：
`bootstrap-frontend-7pfuvja0/report.json`。
加减乘三种augassign与普通形式的编译对照保持byte-identical：
`frontend-augmented-q8mpv6dv/report.json`。
上述路径的共同前缀均为 `/home/lhy/poseidon-work/results/`。

最终回归75项全部通过、0skip：test_native_function_calls、
test_native_function_evidence、test_bootstrap_frontend、test_frontend_augmented_ops、
test_hecate_contract、test_dsl_semantic_inventory、test_object_unary、
test_object_unary_exercises、test_object_unary_paid_evidence、test_result_retention。
显式启用了18项真实native测试、CPU证据、当前编译证据、bootstrap/augassign
before/after证据和已有v22付费证据。历史付费证据只离线审计。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 950s python3 scripts/baseline/probe_native_call_boundaries.py
timeout -k 3s 950s python3 scripts/baseline/probe_native_calls.py
# 以下会重新进行真实CPU加密实验，不调用API；替换为上一步新的probe目录。
timeout -k 5s 1600s python3 scripts/baseline/run_native_call_goldens.py \
  /home/lhy/poseidon-work/results/native-function-calls-3a59ga67
```

主仓库独立补丁：`scripts/baseline/patches/dacapo-native-function-calls.patch`。
只含原生调用改动，不混入先前bootstrap、augassign和明文减法修复。
已在临时GIT_INDEX_FILE中read-tree固定commit并`git apply --cached --check`通过，
在当前子模块`git apply --reverse --check`也通过；均未应用补丁或改动实际索引。
主仓库及子模块`git diff --check`通过。仅本地修改，不commit/push/切换分支。

## 仍未完成

1. 设计并实现版本化Agent多装饰函数契约、符号/类型检查、受限构造和诊断反馈。
2. 新的真实付费Agent案例；人工golden不能计入Agent parse/compile/correctness率。
3. 更广泛参数绑定、packing/类型组合、上游helpers、真实bootstrap等语义。

这一步提供原生调用确定性基线，不缩小“完整支持DSL”的目标，也不声称目标完成。
