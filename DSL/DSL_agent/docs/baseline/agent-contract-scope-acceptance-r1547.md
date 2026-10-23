# 契约指导 v16 阶段验收

`explicit-v16` 已接入 CLI、组件和三个反馈入口，按 native/public 构造契约提供类型与控制流说明。
普通输出使用既有 v14 基础，共享输出使用 v15 基础。旧默认及旧请求保持。

## 改动

- 区分 Hecate Plain Expr 与静态 Python 整数。
- 按实际契约说明数组构造、公开循环、比较和 unary 操作。
- 明确 tensor_route 已包含完整数学贡献；mean 向量中的归一化不能重复应用。
- 组件与验证器使用同一套受控修复提示。
- 长批次日志写入 Ubuntu 文件，由实际进程身份与终态记录监控。

## 验收

| 项目 | 结果 |
|---|---|
| 相关控制 | 96 passed，0 failed，0 skipped |
| 原请求兼容 | 10 份按原字节重建 |
| 规则基线 | 2/2，rank_axis_8、broadcast_tail |
| 人工 public 构造 | 1/1，public_square |
| 实际执行 | 3 次编译、9 次 Poseidon 密态调用 |
| 程序包 | 3 份重载、新输入、重复执行及缓存通过 |
| 独立审计 | 24 组输入、2,456 个值，maxabs 2.29712948485e-06 |

门限为 `1e-5 + 1e-4*abs(reference)`。新真实 Agent 生成结果另计。

## 前一批真实 Agent

r1540：0 通过、2 失败、7 未运行；5 生成／6 HTTP 全部结算，未知调用 0。
rank_axis_8 最后程序的 flat 输出 864 值通过；mean 与其别名失败。
133 项 mean route 与冻结绑定完全相同，但最后额外乘1/6，重复归一化。
rank_linear_16 两次传输终态失败，分别为 timeout 和 stream；未编译。

控制器退出码120与输出流刷新失败相符；原外层 traceback 未保留，确切原因尚未证实。
本轮日志写入客体文件，运行不会依赖原 SSH 输出连接。原响应、失败、源码和绑定保留。

首次测试的两处 fixture 错误、legacy graph 源码接入的配置错误及首次审计的 ABI 分支遗漏也保留。
修正了测试与开发工具，原模型、候选及 reference 保持。

## 当前批次

9 项 v16 真实 Agent 测试正在运行；同模型／权重／布局／参数，使用新生成指导与 SDK 身份。
scope `35fca2561278d843ed904607607cebab22a2ca2d194caffc8dd477d2fd0f7be4`，SDK `cd63d63c875a9c1625c81e1b76763d92dd2e90f23feeb77ee474e8e93f22ba3d`。
最多36生成／144HTTP、36编译／42密态；API与密态串行，每例4小时／全批12小时。
产物新增最多4GiB、总48GiB、余量至少4GiB。外发仍为公开模型、权重或描述符、布局、规则与受控反馈。

产物 37.50/48GiB，磁盘余量 31.85GiB。
Poseidon、Runtime、Dacapo和参数身份保持；本轮无安装、下载、网络变更或Git提交推送。

未验证的容量组合、原模型应用精度、Agent bootstrap、CUDA 实机与 ResNet20继续独立推进。
