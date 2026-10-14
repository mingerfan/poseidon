# 接线草案（未应用）

## 当前依据

- unified_graph_contract.py:397–411：已登记 explicit-v1…explicit-v15。
- unified_graph_contract.py:419、428–429：以 construction_profile 区分 native/public。
- native_public_loops.py:43–74、145–174：native AST、call/attribute 白名单与静态 range 展开。
- native_array_core.py:14–16、47–62：rank/cell 边界及唯一构造器。
- decorated_functions.py:115–118、133–179、190–218：c/p 类型、Plain 绑定、运算和调用规则。
- unified_public_contract.py:8–38：public-v1 独立构造能力。
- docs/baseline/agent-native-guidance-review-r1541.json：观察与原始证据绑定。

当前失败反馈正确；提示混淆是独立发现，不声称是全部失败的唯一原因。

## 最小接线

1. 将草稿模块移入 scripts/baseline，增加 NATIVE_SCOPE_GUIDANCE="explicit-v16"。
   不改旧常量、RULES、旧 guidance 函数的既有分支或默认版本。

2. explicit_generation_guidance 新增唯一入口分支（在旧版本检查前）：
   - 为该请求生成 explicit-v15 指导，沿用它对 exact output binding 的要求。
   - 调用本模块 specification(request, inherited)。
   - 再调用 fit_public_context，让新增固定说明受既有总预算约束；只裁剪可选示例。
   - 不给全部旧版本元组机械追加 v16，不让未过滤的公共 typing 再覆盖结果。

3. prepare 的版本接受、完整 envelope 后再准备指导的延迟分支，及 validate_request 的
   版本接受应分别加入 v16；校验仍需重新构造全文并精确比较、再校验 request_digest。
   run_candidate --unified-guidance 只增加显式选项，CLI 和组件默认保持。
   旧请求不升级、不重写，v16 产生自己的请求和生成身份。

4. static_repair_hint 为 v16 单独调用本模块 repair_hint，保留共同 entry/header/ABI 检查。
   不同时调用旧 unified_typed_guidance.repair_hint 输出未过滤文本。
   validation_adapter 和 application_diagnostics.attach 的 v16 分支也指向本模块。
   后者已有 checked(witness) 路径，将 reason_code 传递给同一作用域规则。
   原错误码、owner、repair policy、预算判断和受控反馈脱敏保持。

5. 核验新模块被生产依赖身份、请求重建及候选隔离进程的只读模块闭包正确绑定。
   不因为草案新增文件而改写当前 SDK 或历史包 manifest。
   public profile 保留其实际构造能力，不统一降级到 native。

## 必要测试

### 纯控制/兼容

草稿中的 14 项测试需要在安全边界实际执行；目前只核验 Python 语法。
测试命令供审查后使用（不会启动 Nix/API/backend）：

~~~sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline python3 -B -m unittest discover \
  -s scripts/baseline/benchmarks/tools/native_guidance_draft_v1 \
  -p test_unified_contract_scope_guidance.py -v
~~~

再补生产接线测试：
- 无显式 guidance、explicit-v1…v15 的选定 native/public 请求字节及 request_id 不变。
- v16 不改变模型、常量绑定、布局、容量版本、后端配置或数值验收等级。
- 大公开 envelope 仍由既有上下文上限拒绝；不得截断必需类型/控制说明。
- 三个反馈入口都使用契约分流提示，未知错误保留原诊断，不回显任意 source/路径。
- 规划 compiler_feasibility=unknown 原样保留，不派生 executable=true。

### 现有 validator 正反例

使用已有可信校验器，不在草案复制其逻辑：
- native 正例：字面量 rotate；range(1,3) 展开后的合法旋转；包含 c/p Expr 的
  np.array(..., dtype=object)；允许的数组方法、星号传参及 name-target 增强赋值。
- native 反例：公开字面量 if/compare、while、for-else、return-in-loop、UAdd Expr/
  object storage、numeric np.array/asarray、empty/full、float/int；p formal 用作 range、
  rotate 或 storage index；普通局部 Plain 名称用作静态下标。
- public 正例：保留现有 public-v1 的公开比较、循环和 numeric/object 构造；不会误套
  native 拒绝提示。ciphertext/Plain 作为 Python 条件仍由原检查器拒绝。
- 参数、模型 family 和案例 ID 变化不决定提示作用域；只依请求实际契约。
- 无定义在本地候选中的函数、字段访问、I/O、网络或动态执行新增权限。

### 执行层

指导层改动不改变 checker/backend 行为，但应对代表性 native/public 请求进行
现有完整验收，并重新冻结受影响真实 Agent 请求。保留 r1540 的每轮原始结果；
人工等价表达、离线候选和真实 Agent 分开记录，不直接计为 v16 已通过。
