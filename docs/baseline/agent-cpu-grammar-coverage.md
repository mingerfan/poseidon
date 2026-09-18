# Agent 正确性：CPU 范围与 DSL 语法覆盖（2026-09-11）

## 当前范围

按用户最新决定，继续前三项目标：可自定义权重/连接的受限模型输入、
DSL 语义—支持状态—测试对应关系、语义增量和真实密态差分验证。
**不再开发 Poseidon GPU 对接，也不把它作为 Agent 正确性的验收门禁。**
目标文本残留的旧 GPU 条目由更新的明确排除指令覆盖。历史 GPU 修改和结果
保留，不删除、不回滚，也不把它们称作 DSL 的 GPU 端到端验证。

实际执行路线仍为：用户 JSON 图 → 可信 PyTorch/FX 分析 → Agent 输出受限
Hecate Python → Dacapo Earth/CKKS/HEVM/CST → SEAL CPU 真密态执行 → 解密
→ 独立明文 reference。不是新增一个 CPU 后端，不执行候选 Python 源码。

当前代码提供 13 种图算子、16 类模型的 96 例在线 Agent 累计通过记录。
相对冻结基线 6 种图算子 / 8 类模型，数量要求已达到，但不等于所有算子组合、
shape、参数和 DSL 语法都已经验证。schema-2/3 的 id 不参与目录模型查找，
用户可提供静态连接和公开权重；每个加密输入仍限 4 个逻辑元素，最后输出
1..4 个元素，隐藏 Linear 宽度 1..8，不能称为任意 PyTorch 支持。

## 受限 Hecate 语法

规范实现为 `scripts/baseline/hecate_contract.py::validate_function`，
候选资源限制进一步由 `candidate_contract.py::validate_candidate` 检查。
以下 EBNF 描述的是本项目接受的函数片段，不是全部上游 Python/Hecate：

```text
program       ::= decorator NEWLINE signature NEWLINE INDENT assignment* return DEDENT
decorator     ::= '@hc.func("' cipher_types '")'
cipher_types  ::= 'c' | 'c,c' | 'c,c,c' | 'c,c,c,c' | 'c,c,c,c,c'
signature     ::= 'def golden(' ordered_parameters '):'
assignment    ::= fresh_name '=' cipher_expr NEWLINE
return        ::= 'return' cipher_expr NEWLINE
                | 'return [' cipher_expr (',' cipher_expr)* ']' NEWLINE
cipher_expr   ::= cipher_name
                | '-' cipher_expr
                | cipher_expr ('+' | '-' | '*') operand
                | cipher_expr '.rotate(' signed_step ')'
                | '(' cipher_expr ')'
operand       ::= cipher_expr | public_name
signed_step   ::= '-3' | '-2' | '-1' | '1' | '2' | '3'
```

补充约束不能由上面的 EBNF 单独表达：

- 按 Python 优先级/结合性解析；返回列表是多个密文，不是一个密文里的槽。
- v0 为单输入，只允许 `+`、`*`、正旋转 1/2；v1 增加原生减法/取负；
  v2 扩展六个带符号步长；v3 AST 对应 2..4 个独立密文参数。
  v4 **请求协议**只澄清 v3 AST 的头部写法，不是新增一种运算语义。
  后续 v5 请求对应 v4 AST：仅当不可变 layout 明确声明辅助加密零时，
  在 1..4 个逻辑输入后追加 zero_ct，物理参数数量为 2..5。五参数签名
  不表示支持五个用户模型输入，旧版本不因此放宽参数规则。
- 参数顺序由不可变 request 决定，逻辑输入采用 x/y/z/t；辅助零仅为最后的 zero_ct。
  不允许默认值、
  注解、额外装饰器、文档字符串、全局语句、导入、循环或数据相关分支。
- 名称必须已定义；赋值只创建新密文变量，禁止覆写参数、常量或之前的变量。
  允许 `a = x` 这样的只读别名，但不能把别名当作新的数值运算能力。
- 算术左侧必须是密文；右侧可为密文或公开常量名。数值字面量不能直接参与
  候选算术，公开量由可信 registry 提供：有限实数 scalar、长度 1 或长度 4。
  常量范围、输入/输出布局和广播规则另外由类型与模型门禁限制。
- 旋转只允许字面整数；正号代表左旋。负步长的语法负号不是密文取负。
  不支持 rotate(0)、bool、变量步长、任意方法、切片和下标。
- 生成片段不手写 bootstrap、scale、level、rescale、modswitch 或 relin；
  编译器/runtime 负责这些操作，产物门禁仍检查不支持指令和实际所需密钥。
- 有源码字节数、AST 节点数、表达式深度和候选运算次数等资源限制。
  AST 接受不等于编译成功、可执行或数值正确。

## 新增的机器可核验覆盖链

`dsl_grammar_coverage.py` 在原检查器通过后分析类型与 SSA 依赖。
`audit_dsl_coverage.py` 首先复核已完成批次 lineage 和真实保存的数值结果，再：

1. 校验被追踪的 `trace-payload.json` 的不可变哈希及 request 一致性。
2. 重跑候选静态检查，与当时的检查结果核对。
3. 以 payload 内的真实候选为源；若旁边的 candidate.py 被换掉则拒绝。
4. 记录语法出现次数，以及沿返回值 SSA 依赖可达的语法出现次数。
5. 每个观察项关联语义清单状态、测试位置和具体通过案例；HEVM opcode
   另外记录，不能假定高层 AST 与低层指令一一对应。

32 个观察项包括：输入数 1..4、输出密文数 1..4、赋值/别名、单一/列表返回、
嵌套表达式、密文取负、三种二元算术各自的 CC/scalar/length1/length4 形式、
六个旋转步长。这只是有限结构分区，不是完整语言的程序组合空间。
没有出现在返回依赖中的 dead assignment 不计入有效观察；可达也不保证
数值影响或在编译优化后保留，因此仍不能用覆盖数字证明语义等价。

审计结果：`/home/lhy/poseidon-work/results/dsl-grammar-audit-gugzv80z/report.json`。
复核 96 例、384 个输入元组、1088 个输出值；最大绝对误差
`7.546270709424263e-7`，固定门限不变。32 个结构观察项中观察到 20 个。
12 项未在该在线 Agent 累计样本中观察到：

| 类别 | 缺失的观察项 |
|---|---|
| 多输入 | inputs.3、inputs.4 |
| 赋值形式 | statement.alias |
| 原生取负 | negate.cipher |
| 加法公开量 | add.length1 |
| 减法公开量 | subtract.scalar、subtract.length1、subtract.length4 |
| 乘法公开量 | multiply.length1 |
| 负方向旋转 | rotate.-3、rotate.-2、rotate.-1 |

这不是“这 12 项都不支持”，也不是“DSL 正确率 62.5%”。例如多输入、
负旋转已有单独人工 golden 证据；某些常量形式可能由当前 registry 选择决定，
不能为了提高覆盖数字而逼迫 Agent 使用数学上不需要的语法。
需要按“静态能力 / 人工 golden / 在线 Agent / 真实密态结果”分别补证据。

只读复核命令（写入一个新的小型审计报告，不发起 API 或新密态计算）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 260s python3 scripts/baseline/audit_dsl_coverage.py \
  /home/lhy/poseidon-work/results/agent-batch-3s484qfx/report.json
```

## 本轮真实多输入重跑

结果：`/home/lhy/poseidon-work/results/schema3-golden-batch-ma16wh0c/report.json`。
当前 v4 请求、既有 v3 AST，5 项全部符合预期；无付费 API。

| 案例 | 来源 / 预期 | 最大绝对误差 |
|---|---|---:|
| custom-dual-subtract | 人工 golden，通过 | 8.885983414363974e-9 |
| custom-dual-linear | 人工 golden，通过 | 3.750706178973218e-8 |
| custom-triple-merge | 人工 golden，通过 | 3.1552062493338e-8 |
| custom-quad-merge | 人工 golden，通过 | 4.186734692268601e-8 |
| custom-dual-subtract / wrong_order | 错误候选，数值比较拒绝 | 3.9999999984614227 |

每项都有 4 组输入，包括零、有符号、固定随机和边界值。反例不是语法报错：
它真实编译、密态执行、解密后才被比较器检出。测试还核对 ordered inputs、
独立 reference、不可变权重/输入/request/产物、真实密钥和隔离运行证据。
所有这 5 个运行的 private-keys 已由原有清理策略删除；模型、DSL、IR、
解密结果和报告保留。随机密钥原字节不可恢复，但测试可生成新的密钥重跑。

```bash
timeout -k 3s 1600s python3 scripts/baseline/run_schema3_goldens.py
```

修复了该证据测试只接受旧 v3 request 标签的过时断言；现在同时接受 v3/v4，
但仍校验各自的不可变规则和参数顺序，没有放宽实际 DSL 语义检查。

## 测试与失败归因

- Windows 上新增的纯 Python 覆盖测试：11 通过，1 真实证据测试按条件跳过。
- WSL 普通 Python 全量回归：350 项，277 通过、73 跳过、0 失败。
- 固定 Nix/Torch/NumPy 下显式启用真实证据的联合测试：26/26 通过，无跳过；
  包含 96 例覆盖复核、这次多输入真实结果、Torch/reference/规则与人工程序对照。
- 新测试最初用 Windows 文本写入导致 CRLF 与原 payload 字节不同；修改的是
  合成测试夹具的写入方式，没有关闭真实源码完整性检查。
- 一次全量测试误运行于 Windows Python，因 fcntl/resource/POSIX 文件接口
  和 Linux 路径语义不匹配而失败；在目标 WSL 上重新执行通过，没有安装依赖
  或修改系统。测试中的 max-repairs=4 报错属于已有的预期反例。

## 后续补证：Agent 与人工分别统计

DeepSeek Flash 六例补跑已通过三/四输入、取负与负旋转。
新的多批次审计支持 --additional-report，重新核验 96+6 个运行的不可变证据，
首次两队列合并时 Agent 结构覆盖为 26/32。对应六项已由
[broadcast 与别名的五个 golden、五个反例](broadcast-alias-cpu.md) 验证真实
CPU 执行，但仍不算 Agent 已覆盖；不能用人工证据增加 Agent 成功分子。
后续版本化语义说明加上同一组五个模型的 Agent 测试，覆盖提升到 28/32。
当时未观察到变量别名及三种原生公开量减法；它们有人工执行证据，Agent 可用
等价加法表达减法，不为凑数而强制写法。

后续 [加密零 ABI 的七个用户图](encrypted-zero-cpu.md) 全部首次生成通过，
MLP 输出依赖实际出现 SSA 别名。96+6+5+7 的独立证据合并为 114 个成功案例、
456 组输入、1304 个比较值，结构特征覆盖为 **29/32**。剩余三项是原生
公开量减法的 scalar/length1/length4 写法；已有人工密态反例测试，但还未在
这些 Agent 输出中观察到。它们不是三个失败模型，也不等于三个缺失数学算子。

## 下一步，不再被 GPU 阻塞

已把 broadcast 的不同常量表示、SSA 别名和 compiler 导出常量的负号来源
通过哈希绑定的 semantic_guidance/schema=1 教给 Agent。继续按语义增量验证更一般的用户图组合，不为凑原生语法
覆盖数字要求 Agent 生成不必要的操作。各增量的正常、边界和反例结果继续
挂到同一语义清单。不能为了通过而放宽门限或静默改写模型。

近似激活保持三份独立输出：原函数 f(x)、明文多项式 p(x)、解密结果 y。
分别报告 p-f 的近似误差、y-p 的实现/CKKS 残差、y-f 的总误差。
现有显式二次多项式示例不代表原 ReLU 等价，也不授权自动 ReLU/SiLU 替换。
