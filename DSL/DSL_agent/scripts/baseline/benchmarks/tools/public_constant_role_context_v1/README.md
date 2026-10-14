# 公开数学常量绑定草案

本目录保留独立草案及其验收工具。r1306 付费批次已关闭后，Agent 生产入口接入了显式 v2 策略；旧草案、初次失败和原批次记录保留。

## 缺口

`descriptors-v1` 将数值换成 shape/hash。GELU 的系数向量没有“系数下标 → 预绑定 Plain”映射，因此候选只能猜测 `pN` 的含义。首轮错误算式与真实解密的差约 `6.93e-12`，与 reference 的差约 `2.19`。

## 新契约

显式策略：`descriptors-math-roles-v2`。

可信准备阶段提供：

- 多项式的基、系数顺序、逐项公开值、准确零标记、最高非零次数及对应 Plain 名称。
- 标量算术的值及对应 Plain 名称；其他张量保留已有分块/布局绑定，不要求逐项存在标量 Plain。
- 精确 binary64 匹配；符号零和原始内容哈希均保留。
- 大型 Linear/Conv 权重维持描述符；现有布局映射继续负责其坐标关系。

数学公开量最多 256 个，额外角色记录最多 48 KiB，总提示仍为 128 KiB。
缺少非零量的完整绑定、无实际加密零能力或越界时，准备阶段返回明确拒绝。
零系数仍保留在数学定义中，客户端加密零与透明密文区分。

按算子及参数位置判断角色，与模型名称、family 或模板无关。字段只说明数学模型和绑定，不包含标准 DSL。补齐这些字段也不表示布局、编译或密态验收已完成。

## 文件

- `public_math_context.py`：标准库实现。
- `overlay/` 与 `integration-overlay.patch`：待审查 Agent 修改；尚未应用。
- `test_controls.py`、`test_overlay.py`：正反例与旧契约兼容控制。
- `prepare.py`：根据原三份冻结请求生成独立草案；原提案与请求保持不变。

```sh
python3 -B -m unittest test_controls test_overlay test_entrypoints -v
python3 -B prepare.py
```

旧策略保持默认，旧请求哈希保持相同。新策略必须显式选择，产生新请求绑定和新的付费授权范围。

最新标准库控制：63 passed / 0 failed / 0 skipped。包含三份真实冻结请求的 prepare_task → reconstruct_request 与 projected_context 验证；以及真实非均匀向量分块 ABI、CLI 策略及 4 MiB 上限、应用 provenance 上限。尚未执行 frontend、编译、密态或真实 Agent。

## 当前生产接入

5 个原 Agent 文件更新，新增 `public_math_context.py`；Poseidon 候选沙箱只读清单补齐该校验模块。旧默认和请求字节保持。历史 overlay-freeze 与 63 项草案控制表示当时的独立草案状态；切换和生产验收另存。

- `production-compat-pinned-r1307/report.json`：144 项旧接口兼容通过。
- `production-controls-current-capture.json`：25 项实际生产入口通过，应用后端为禁止执行的测试替身。
- `sandbox-controls-r1307.json`：12 项沙箱、绑定和资源控制通过。
- `capture_current.py`：原三个来源在新身份下重新隔离捕获。
- `acceptance.sandbox.py`：一个 functional GELU 规则程序的真实 Poseidon CPU、数值及包复用验收；API 为 0。
- `inspect_current.py`：只读检查当前源码、报告和保护清单。

原 3 项真实 Agent 测试为 0 通过 / 3 失败；新策略真实 Agent 评测尚未运行。所有旧请求、报告和失败目录保持。
