# 混合 packed／逐单元布局验收（r1311）

三个原低阶 Attention 模型已通过规则候选的真实 Poseidon CPU 编译、数值验证、程序包重载及重复执行。此前两个 `std::bad_alloc` 样例在原 4 GiB 沙箱限制下通过；后端库、参数、模型、公开权重、请求和 reference 均保持。

## 方法与范围

公开 Linear 直接使用冻结的 diagonal／bias 绑定；形状变换维护逻辑索引。在需要密文跨元素计算时按需提取并复制单元，进入后段 Linear 或输出时重新按 C-order 打包。缓存以不可变表达式、slot 和周期定位；没有按模型名称分派，也未从不透明 pv 向量反推缺失标量。

三个候选未使用缺失权重字面量兜底。本批仍是 `deterministic_rule_baseline`，不是新真实 Agent 生成。MHA v16 和混合 lowerer 均留在独立 staging；生产 exporter 的顺序修复见 [r1310](agent-formal-provenance-export-results-r1310.md)。

| 原样例 | 真实资格／新输入／重复执行 | 编译产物已知 payload 下界：旧→新 |
|---|---|---|
| Self Attention＋后段 Linear＋残差 | 4＋3＋1 组，128 值通过 | 3,863,740,416 → 2,269,511,680 bytes |
| 独立 QKV Cross Attention＋投影输出 | 4＋3＋1 组，128 值通过 | 4,064,018,432 → 3,125,673,984 bytes |
| 布尔 padding mask＋残差＋尾块 | 4＋3＋1 组，80 值通过 | 1,321,467,904 → 949,223,424 bytes |

合计 **24 组执行／336 值**，包含 3 组重复输入；最大绝对误差 **1.1185962366100366e-7**，满足原门限 `1e-5 + 1e-4*abs(reference)`。对照的是原声明的低阶近似计算，原 Softmax／应用准确率未在本批认证。

已知 payload 为当前 Runtime 保留值的存储下界；不包含完整评估密钥、分配器和工作区，不能作为测得峰值。实际密态成功由执行与解密记录证明。本批未测量峰值内存。

## 控制与保留的失败

- 混合策略控制：28 passed，0 failed/error/skipped；涵盖原三例、通用 Linear／重排／部分归约／广播／split 多输出、带污染尾槽的后段 Linear、绑定与来源拒绝。
- 输出树控制：3 passed，0 failed/error/skipped；按声明复核嵌套 None、tuple/list/dict、scalar。
- 首轮控制保留：16 passed／9 errors，公共 packing 记录的 set 无法编码；没有进入编译或密态执行。修复仅将精确 schema 内的 set[str] 排序记录，实际 packing 未变。
- 第二轮原报告保留：2 项完整工作流通过；Cross 在已完成密态／数值比较后，因脚本假定存在 weights 输出而 KeyError。实际树为 result 与 aux=[None, projected]。
- 后续独立绑定复核原三组新输入，并对三份包各执行一组重复输入，完整树与数值均通过。未再次编译。
- 原 driver 单次包执行后的 repeated_execution=True 不作为证据；后续报告记录了真正的第二次包执行。未来 driver 已修正该字段，并显式冻结独立 AST 解释器。

原报告、失败日志与执行时源码快照均保留；未把旧失败直接改写为通过。

## 使用与复核

[机器报告](agent-mixed-packed-scalar-results-r1311.json)

~~~bash
python3 -B scripts/baseline/benchmarks/tools/mixed_packed_scalar_rule_v1/inspect_current.py
~~~

该命令只校验当前源码、历史快照、保护表、资格产物、程序包及执行证据。本批采用固定 LLVM/MLIR、Poseidon CPU w45 与原 Q13/P1/tc128；编译≤2／链接≤1，密态串行。654 项后端、reference 与安全保护源码保持，x86 配置保持。本批 API／HTTP、安装、下载、后端及参数修改、commit 和 push 均为 0。

## 后续

先完成混合布局的组件接线和 MHA v16 生产切换范围审查，再冻结生成指导与真实 Agent 案例。其他周期、其他 Linear 策略、卷积／池化／BatchNorm、混合公开／密文拼接和更深模型不属于本草案新增验收范围；已有对应生产能力另按原报告使用。Bootstrap、通用算子扩展、大模型容量和 CUDA 继续按原计划分别推进。
