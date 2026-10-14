# 公开参数类型转换：Poseidon CPU 验收

36 项配置全部通过，包含 33 项 v24 类型转换和 3 项旧版本兼容回归。数学图归一化后为 12 个请求。本批使用冻结的规则候选。

| 项目 | 结果 |
|---|---:|
| passed / failed / skipped / not_run | 36 / 0 / 0 / 0 |
| 成功编译 | 36 |
| 真实 Poseidon CPU 密态调用 | 42 |
| 首次数值验收 | 144 组次、1,152 值 |
| 程序包新输入 | 9 组次、90 值 |
| 程序包重复输入 | 3 组次、30 值 |
| 最大绝对误差 | 3.074798282654001e-8 |
| 用时 | 896.78 秒，约 14 分 57 秒 |
| 新增产物 | 141.16 MiB |
| Provider / HTTP / 真实 Agent 生成 | 0 / 0 / 0 |

门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。组次包含重复输入和配置间重叠，36 项配置与 12 个请求分别计数。

## 验收范围

公开参数和 buffer 的源类型包括 float32、float64、int32、int64、bool，目标为 float32 或 float64；另覆盖 float64→float32→float64 往返。三个上下文分别是公开量乘法、转置后相加、乘法与残差结合并返回命名结构。旧版兼容覆盖 v22 布尔 mask、v23 公开参数计算及 named 输出。

每例均完成真实 Hecate tracing、Dacapo-modified 编译、RuntimePlan/OperatorSpec/plaintext bundle 检查、Poseidon CPU 密态执行、解密及四组独立 reference 逐项比较。公开 float32 的舍入先在公开状态中体现，然后进入既有 float64 图计算；本结果适用于这些登记写法与配置。

三份程序包 `f64_to_f32_c0`、`f64_roundtrip_c2`、`compat_v23_named` 完成重载，各执行 3 组新输入和 1 组重复输入。复用没有生成或编译，输出另经独立数学 reference 比较，包文件哈希保持；named 字典/列表中的 None 正常返回。

## 绑定与记录

- 开发 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e480`。
- SDK：`8ca41ede84d20c654fdc7885658e343d1d263cfc0885fd1a96ad259c54e38f8a`。
- 冻结方案：`2ed1e15ffdf508b856b8f6b83def6c31eaee4ec1392d5cb094e5da4cdaf7be0c`。
- 原批次报告：`bdb091bd8e3f6b6c60e59cf346bca91811ab31a810234c584d467bdf47a3bd96`。
- [机器可读报告](agent-public-dtype-offline-results-r1380.json)包含原报告、四阶段身份与逐文件 SHA，以及保存结果的独立计数审计。审计结果为 `complete_verified`：编译和密态预留分别 36/42，确认计数相同；首轮及复用数值共 156 组次、1,272 值。
- 原始证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-dtype-cast-cpu-v24-v1`。
- 全局结果目录 28.65 GiB，剩余磁盘 41.06 GiB。预算保持 20 分钟／新增 512 MiB／全局 32 GiB／余量 4 GiB，编译≤2、链接≤1、密态串行，native 4 GiB／CPU150秒。
- 后验报告封装首次出现重复字段的 SyntaxError，在读取或执行验收前停止；日志保留，修正后仅做保存结果审计，没有重跑编译或密态。

## 后续

v24 显式选择方式保持，默认 v18、latest=v21。真实 Agent 生成与自动修复能力仍须按新增功能单独冻结、批准和验收。

下一条只读命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_dtype_offline_v1/run.py --plan
~~~

本批已结算，原证据及历史记录保留。网络、依赖、后端库、参数、模型、reference、门限保持；本批安装、下载、提交、推送均为 0。
