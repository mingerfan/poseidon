# 公开常量运算接入审计

当前 ARM Ubuntu checkout：`/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。审计日期：2026-10-05。

这批检查定位了加、减、乘的两处接入缺口。尚未修改生产转换器；无需改动后端来开展后续接入修复。

| 检查 | 接受或通过 | 拒绝 | 验证范围 |
|---|---:|---:|---|
| JSON 图原始输入 | 6/12 | 6 | 公开量在右侧的表达接受；在左侧的表达被分解层拒绝 |
| PyTorch 原始输入 | 8/12 | 4 | 实际隔离捕获；接受项完成 128 组原模块与独立参考比较 |
| 单列的人工等价图 | 12/12 | 0 | 192 组明文比较，最大误差 0；不计为原输入接入通过 |
| 诊断与绑定测试 | 6/6 | 0 | 错误层区分、精确诊断匹配、篡改记录拒绝 |

这两个入口测试的是相同的 12 个运算场景，不计为 24 个独立模型。冻结的 1,200 模型集保持不变。

## 已确认的原因

- JSON 的 `c + x`、`c - x`、`c * x`，其中 `c` 为公开标量或张量，均触发 `First operand must be encrypted`。检查位于 Agent 的图分解层。
- PyTorch 的标量 `c - x` 触发同一检查，错误带有 `fx_sub (subtract)` 节点位置。
- PyTorch 的公开张量 `C + x`、`C - x`、`C * x` 分别捕获为 `Tensor.add/sub/mul` 调用，当前映射未登记这三种调用形式。
- 标量 `c + x`、`c * x` 已有操作数交换路径；数据张量在左侧的六项也完成实际捕获。接受项原模块、图 reference 和两套分解 reference 一致，最大误差 0。

源码依据：`scripts/baseline/model_decomposition.py` 的首操作数检查，以及 `pytorch_model_capture.py` 的算术映射和未登记调用拒绝。

## 修复方向

在 Agent 输入规范化层统一识别公开量与数据张量。加、乘可交换操作数；`c - x` 可分解为 `negate(x) + c`。PyTorch 同时需要登记对应的 Tensor 方法，不能只调整 JSON 图。常量、原图和节点来源必须保留。

人工图的数学比较支持上述方案，但尚未证明新输入形式能够真实编译和密态执行，也未验收真实 Agent 生成。生产接入补丁应建立新版本和绑定，随后补受影响的编译、数值、程序包和 Agent 回归。

## 现有批次与保护

r870 批准的 24 项 split 验收已在 [r878](agent-pytorch-split-acceptance-r878.md) 完成并关闭。

[r914 的 48 项后端提案](agent-json-matrix-offline-proposal-r914.md) 仍待具体批准：676 份运行源码、786 份案例及证据文件、4 份驱动和绑定 `3c3168bd8919638722c688f3fe1fc90b6ea6c0cfd585c7ac68f6b41ae5df69c5` 均保持。没有创建新批准收据或启动该批次。

648 份后端文件、6 份 reference/门限/检查器文件及既有保护报告核对保持；1,965 条原有修改或未跟踪路径保持。分支和 HEAD 未变，无删除或后台验收进程。

本批新增 tracing、编译、密态执行、真实 Agent 生成、付费调用、安装、下载及 Git 发布均为 0。数值门限仍为 `1e-5 + 1e-4*abs(reference)`。

## 复现与原始记录

开发工具位于 `scripts/baseline/benchmarks/tools`：

- `audit_public_operand_roles.py`：JSON 接入和单列人工数学图检查。
- `audit_public_operand_capture.py`：12 个 PyTorch 原始写法的隔离捕获。
- `review_public_operand_capture.py`：核验现有捕获记录和失败层，无重复捕获。
- `test_public_operand_review.py`：6 项诊断及绑定测试。

原始 results：`agent-public-operand-role-reference-r924`、`agent-public-operand-role-capture-r927`、`agent-public-operand-role-review-r931`。机器报告保留文件绑定和来源。

初次分类器遗漏了三项 Tensor 方法错误；第一次复核遗漏带节点前缀的减法诊断。原报告、工具快照和失败日志保留，最终复核只补分类与完整性检查，没有重跑 12 个捕获或把拒绝转为通过。初次启动包装器未记录子进程返回值，其零值不能作为审计通过证据；r931 和 r933 已显式核验实际返回值。
