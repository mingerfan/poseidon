# Native guidance draft v1

状态：仅草案，未接入生产，未激活 explicit-v16。当前生产最高版本为 explicit-v15。
运行中的 r1540 源码、请求、候选和原始证据保持原样。

## 范围

复用 explicit-v15 的布局、公开常量、逻辑语义、能力与资源说明，只按已验证请求的
construction_profile 选择准确的构造说明：

| 请求契约 | 草案行为 |
|---|---|
| 无 construction_profile：hecate-unified-native-v1 | 替换现有 type_rules、typed_values 和 native helper/loop 段落 |
| hecate-unified-public-v1 | 保留原公开 AST 构造说明 |
| 未知或 null profile | 拒绝；不推测能力 |

原生构造说明区分 c/p Expr 和静态整数字面量/展开后的 range 下标，明确 native 不接收
if、Compare、while 等语法，即使条件是公开字面量。原生数组仅允许现有
np.array(data, dtype=object)；public profile 的 empty/full/asarray、转换和控制流
说明不再混入 native 提示。原生 UnaryOp 也按现有检查器说明：Expr/storage 的 UAdd
不开放，静态整数字面量的符号折叠另计。

不按模型名、family、案例 ID 或 rank 分派，不包含模型 DSL 答案或 reference。
不修改 AST/类型检查、工具链、参数、布局、常量、work 限制或任何旧版本。

## 文件

- unified_contract_scope_guidance.py：拟接入模块，采用现有 guidance 字段；无新能力注册表。
- test_unified_contract_scope_guidance.py：14 个纯控制单元测试草稿。
- integration-notes.md：安全边界后的接线点与必要验收。
- source-bindings.json：起草时生产源码/公开请求的哈希及草案文件哈希。

specification(request, inherited_v15) 返回独立副本，不修改输入；只有外层 version 变为
explicit-v16，现有 poseidon_expression_rules 的 v15 身份与内容原样保留。
规划深度、估计 work、常量数和 compiler/execution feasibility 均不被此模块改写。
repair_hint 复用已有固定诊断分类及 v15 输出/布局反馈，原生类型提示换为相同作用域说明。
临时 v15 view 仅供原反馈适配器调用，重新计算它自己的哈希，不写入记录或替代新请求身份。

## 本次验证与待验证

本次只对两个 Python 文件执行 ast.parse 和 compile(..., "exec")，不执行模块或测试。
单元测试尚未运行；真实前端、编译、密态执行和真实 Agent 验收均未运行。
本目录不是当前生产路径的一部分。不得在冻结批次运行时复制模块到 scripts/baseline。

后续由主线程在安全边界审查接线，先运行纯控制和旧哈希兼容，再验证已有 native/public
正反例；只对受影响范围安排后端/Agent 验收。无需因为指导文字调整重跑整个 Benchmark。
