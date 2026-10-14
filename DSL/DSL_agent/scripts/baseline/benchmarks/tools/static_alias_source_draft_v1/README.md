# 原地操作与视图别名输入草案

该模块把有序的张量操作转成不可变计算图，供 Agent 输入层后续接线使用。

## 范围

| 操作 | 处理方式 |
|---|---|
| 原地 ReLU、add、sub、mul | 创建新存储版本，后续别名读取更新后的值 |
| reshape、view、flatten | 按步幅兼容性选择共享存储或复制 |
| transpose、permute、静态 slice/split | 保留视图映射 |
| clone、contiguous | 区分独立副本与已有连续视图 |
| tuple/list/dict 输出 | 保存结构、命名张量及静态 None |

输入采用独立、连续的 CPU float64 逻辑张量。公开常量保持只读。重复写入位置、部分重叠的算术操作数、错误 view、动态索引及未登记调用返回结构化诊断。

ReLU 使用调用方显式提供的多项式和有效区间；原函数与近似计算分别验证。

## 接口

- `plan(source, approximations, resource_profile=...)`：有序数据声明 → 数学图、分解图和别名事件。
- `verify(...)`：以外部固定请求哈希验证来源、内容及实现。
- `from_fx(...)`：映射已知本地夹具中的固定 Torch FX 调用。

## 验收

在现有锁定 pure Nix Python 环境内运行 `audit.py --output <新结果目录>`，结果写入 work root 的 `results`。

r1068：34 项测试通过；24 个上下文、384 组独立明文差分、6304 个输出值。另有 96 项布局检查，其中 80 项接纳、16 项拒绝，与固定 Torch 一致；rank 4/32 逻辑边界各有一例。

当前为数学草案状态。生产接入、密文布局、编译、密态执行、Agent 生成及应用精度各自待验收。

[验收报告](../../../../../docs/baseline/agent-alias-source-draft-acceptance-r1069.md)。
