# 公开参数类型转换：Poseidon CPU 离线验收

## 范围

36 份来源配置：公开 float32/float64/int32/int64/bool 转 float32/float64，各三个组合上下文；三个浮点往返上下文；三个 v22/v23 兼容项。归一化后为 **12 个请求**，每项按各自来源和程序包身份验收。

使用冻结的规则候选，检查真实 tracing、编译、RuntimePlan/常量/OperatorSpec、Poseidon CPU 执行及四组输入逐项比较。三个代表包另执行三组新输入与一组重复输入，检查返回结构、None 及包文件不变；共计划 **156 组、1,272 值**。

## 预算

| 项目 | 上限 |
|---|---:|
| 整批／首次任务 | 20 分钟／180 秒 |
| 编译／密态调用 | 36／42 |
| 编译／链接／密态并发 | 2／1／1 |
| native 内存／CPU | 4 GiB／150 秒 |
| 新增产物 | 512 MiB |
| results 总量／磁盘余量 | 32 GiB／至少 4 GiB |
| API／HTTP／下载／安装 | 0 |

门限保持 `1e-5 + 1e-4*abs(reference)`。身份、完整性、环境或预算故障停止相关批次并保留现场；独立案例失败分别记录。未运行、失败和规则候选不计作真实 Agent 成功。

## 绑定与启动

SDK `8ca41ede84d20c654fdc7885658e343d1d263cfc0885fd1a96ad259c54e38f8a`，方案绑定 `2ed1e15ffdf508b856b8f6b83def6c31eaee4ec1392d5cb094e5da4cdaf7be0c`。精确案例、源码、权重、任务与工具哈希见[JSON](agent-public-dtype-offline-proposal-r1376.json)。r1375 准备草案保留；本方案增加工具与任务绑定，尚未授权。

唯一开发/执行目录 `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`；后端、参数、reference、网络和依赖保持。

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_dtype_offline_v1/run.py --plan
~~~

批准记录建立后去掉 `--plan` 执行，复用一个已有 Nix 环境；各候选独立隔离。

[接入检查](agent-public-dtype-component-checkpoint-r1374.md)
