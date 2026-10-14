# Poseidon DSL 构造与协议验收

冻结清单的 **209 个构造分区、627 个原任务全部获得 Poseidon CPU 数值验收证据**。每个分区保留三个不同拓扑的原上下文。Dacapo、Runtime、Poseidon 库、安全参数、模型和 reference 均保持原样。

## 结果

| 范围 | 结果 |
|---|---|
| native 构造 | 129/129；43 分区 |
| public 构造 | 498/498；166 分区 |
| 双参考检查 | 10,032 组 |
| 真实密态比较 | 2,508 组输入、8,144 个输出值 |
| 最大绝对误差 | 8.31713211e-07 |
| 静态正反例 | 486/486，0 failed、0 skipped |
| 编译产物协议 | rescale、modswitch、relinearization、参数、rotation keys 共 5 项，各 3 个已执行上下文 |
| 冻结 helper 原任务 | 60 项历史 Poseidon 绑定核验；15 个可执行分区，3 个 bootstrap 分区阻塞 |
| 组件及程序包测试 | 76 passed、0 failed、3 项历史证据 skipped |
| 应用及包重载 | MLP、RMSNorm、Attention、native-return 数值等级及 MLP 编译等级，5/5；重载零生成/编译、缓存零 Provider 调用 |
| 本批真实 Agent | 未运行，付费调用 0 |

门限为 `1e-5 + 1e-4*abs(reference)`。627 项为规则候选的定向任务，其中 9 项是空返回或空参数展开的结构验收，不宣称数值贡献。其余核查真实 tracing、返回依赖和有限干预。606 项为 schema 3，21 项保留 r247 的 schema 2 来源，各来源绑定原样保留。

r258 新增 450 项耗时 1,585.48 秒，全部通过。486 项静态检查包含 48 个拒绝上下文、153 个数学反例、209 个合成 trace 反例、76 个 helper 省略调用反例，单独计分。编译协议观察产物指令、实际参数与最终输出，并绑定 Runtime 源码；未声称逐指令运行跟踪。

## 编译入口与剩余限制

原 9 项失败由未使用的独立 helper 定义触发，已通过 private 可见性和现有 `symbol-dce` 解决；原始 IR、helper tracing 和函数体保留。详见 [r257](poseidon-dsl-migration-r257.md)。

另有三个补充 `Poly_Default` 原案例在新入口下仍失败：

| 案例 | 原表达 |
|---|---|
| fixed_Poly_Default_0 | `Poly_Default(x)` |
| fixed_Poly_Default_1 | `Poly_Default(x) + x*.125` |
| fixed_Poly_Default_2 | `Poly_Default(x).rotate(1)` |

都在 `MPCB.py:77` 的 Earth 乘法类型推断失败：同一 level 11 的两个输入分别为 `ci<90*11>` 和 `ci<75*11>`，尚未进入 Runtime 或数值比较。这三项未计入 627 构造任务，也未用其他模型替换。

修改方向是审查固定 helper 的中间 scale 运算和 Earth 乘法类型规则；需要编译语义补丁或新参数配置时单独提出方案。本轮未修改它们。

## 覆盖范围

[叠加账本](poseidon-semantic-migration-overlay-r261.json)保留原 401 项分母：构造 209/209、编译协议 5/5、拒绝 16/16 已映射；helper 15/18 为历史来源绑定核验，3 项依赖真实 bootstrap。数学分区 67/153 有三个已执行拓扑上下文，含历史 helper 来源证据。

原 459 项数学上下文和 92 个补充模型完整保留。按原任务的模型 SHA，280 个不同模型中，58 个与构造记录匹配，另有 14 个匹配 helper 记录，尚有 208 个需继续映射或执行。未重新生成模型集；构造全部通过不替代这部分验收。

107 项上游 API 中，106 项 AST 一致。新 `HE_Linear` 增加 `nt=2**16` 参数，默认数学行为相同；额外参数未开放给 Agent、未扩大验收范围。分类核验不计为逐 API 执行通过。

r259 审计器首次把身份 v1/v2 格式当作相同记录而停止；差异仅为 format 与新增构建配置字段，共同源码及二进制哈希全部一致。日志保留，r260 完成严格兼容核验，旧身份未补写成 v2。

## 后续

- 三个小模型的真实 Agent 请求保持 [r256](poseidon-whole-models-live-proposal-r256.json) 绑定，待独立付费批准。
- r233 参数试验已执行为 r234，有符号输入首次刷新误差约 `1.0398e-4` 超限。r235 ratio128 独立试验仍待批准，Agent bootstrap 保持阻塞。
- 复用冻结数学上下文，仅补未映射模型。旧通用 Benchmark 封装中的 SEAL 配置在付费绑定安全边界后迁移；当前定向工具已使用 Poseidon 单例入口。
- x86 原生、空白安装和 CUDA 暂缓。

本轮无安装、下载、付费、commit 或 push。完整结果：[JSON](poseidon-dsl-migration-r261.json)。
