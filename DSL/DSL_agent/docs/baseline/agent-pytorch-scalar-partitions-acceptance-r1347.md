# PyTorch 标量分块接入验收

ARM Ubuntu 的 Agent 接入层新增显式 `poseidon-trusted-pytorch-v20`。一维 `unbind` 返回共享标量视图，可参与算术、`stack`、容器输出及有序写入。默认来源清单仍为 v18，旧接口和 x86 配置保留。

本批使用规则候选，通过 Poseidon CPU 编译、密态执行、解密和独立 reference 比较；真实 Agent 新生成、API 和 HTTP 调用均为 0。

| 检查 | 结果 |
|---|---:|
| 新语义与拒绝测试 | 39 passed |
| 兼容回归，含 v19 | 186 passed |
| 来源清单 CLI | 59 passed |
| 最终 failed / errors / skipped | 0 / 0 / 0 |
| 来源捕获／编译／密态调用 | 8 / 8 / 16 |
| 密态输入组／输出值 | 64 / 160 |
| 最大绝对误差 | 8.912422788665e-8 |
| 程序包重载及新输入复用 | 8 / 8 |

门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。每例验收四组基础输入，再重载程序包执行四组新输入；复用未追加生成或编译。

## 验收模型

| 模型 | 来源版本 | 重点 |
|---|---|---|
| scalar_function_merge | v20 | 函数关键字调用、标量算术 |
| scalar_shared_write | v20 | 标量视图写入、源张量与多输出 |
| scalar_source_snapshot | v20 | clone 快照及源张量后续写入 |
| scalar_repeated_stack | v20 | 静态负索引、重复端口与 stack |
| scalar_singleton_view | v20 | 单元素输入的标量视图输出 |
| scalar_public_weights | v20 | 独立 NPY 权重、公开标量分块 |
| diagonal_output_write | v19 | 对角线共享视图兼容 |
| legacy_mlp | v18 | Linear→square→Linear 兼容 |

静态测试另覆盖 tuple/list/dict 输出、detach、嵌套 chunk/unbind、标量 einsum、公开权重不可修改、动态端口拒绝及传输篡改。静态部分端口、负索引和重复端口由已有静态枚举支持，本轮增加兼容回归，未作为新增算子计数。

## 接入层修复

- 单元素纯视图输出保留显式 identity reshape，满足现有非空计算图契约。
- v20 显式登记公开 arithmetic view 检查，重算 slice、reshape、permute 和 public gather 的类型与来源；替换公开根内容或操作数的反例拒绝。
- 六个新增上下文复用现有标量载体、别名和基本索引 lowering；Dacapo、Runtime、Poseidon、reference 算法、安全参数和数值门限均未改。

最初的两项错误、修复中间记录及诊断保留。未使用简单模型替换失败案例，最终验证仍包含原单元素与公开权重模型。

## 绑定与范围

开发路径：`/home/lhohy/Code/Poseidon`；分支：`lhy-agent-dsl`；HEAD：`7588e48093631eaf8385506e3db058e7ca3cfe7e`。

SDK：`26505d8c3650845db95f9bf19e21a99d4109984ea5b02a410bfc48c404953e71`。8 个程序包均保留 Earth/CKKS IR、RuntimePlan、OperatorSpec、plaintext bundle 及数值验收记录。终态审计核验 1,180 份冻结源码、654 份后端／数学参考、两份历史请求；r1341 未执行方案字节保持，当前已失效，未使用其额度。

本批新增产物 56,308,501 字节，所有结果 27,976,294,131 字节；900 秒窗口内完成，密态串行，编译≤2、链接≤1。安装、下载、参数变更、commit 和 push 为 0。

上述结果只覆盖表内代表模型。新构造的真实 Agent 生成与修复、所有 shape/dtype、较大图、应用精度、bootstrap、密文分区交接、CUDA 实机及整网验收继续按原计划推进。

[使用说明](operator-decomposition-and-python.md#pytorch-v20-标量-unbind) · [机器记录](agent-pytorch-scalar-partitions-acceptance-r1347.json)

完整证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-scalar-partition-v20-source-acceptance-v1`。
