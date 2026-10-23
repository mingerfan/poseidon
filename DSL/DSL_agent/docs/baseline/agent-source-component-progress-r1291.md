# Agent 来源组件接入进度

日期：2026-10-08。开发目录：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

## 本轮完成

联合 V14 草案已通过真实 `PoseidonBackend` 身份下的任务接入检查。九份融合矩阵模型覆盖
addmm、baddbmm、addbmm，各含密文矩阵、公开右矩阵、公开偏置三个上下文。复用已有捕获记录，
没有重新执行来源工厂。

检查了规范化图提交、两文件来源与 NPY 权重快照、生成／编译／执行／验证身份、来源记录、
数值验收等级的保持、稳定复用键，以及取消。任务接纳仅表示排队；未授予编译或数值通过状态。

八项测试方法有合并通过记录：原运行中七项成功，拒绝测试修正后单项补验成功。
补验包含六例缺失权重、三例额外权重、九例源码哈希变化。原运行的三处夹具错误仍保留：
纯密文模型没有公开权重，旧夹具错误地尝试删除它们。未用 skip 消除错误，也未重跑其他七项。
首两次启动的显式平台配置／只读挂载问题及日志保留，均未进入测试正文。

213 个来源、请求与测试快照哈希已核验。固定 Python 3.10.14、NumPy 1.25.2、ARM Torch 2.0.1；
沙箱清空环境、隔离网络，只读覆盖十个 Agent 文件。身份来自现有后端对象，无身份替身。

## 当前状态

| 范围 | 状态 |
|---|---|
| V14 来源／请求／应用任务接纳 | 草案验证通过 |
| 联合 706 文件版本编译、密态及编译程序包复用 | 未验收 |
| 生产 704 文件、后端 648 文件、reference／门限 6 文件 | 保持原绑定 |
| 原 76 项真实 Agent 批次 | 56 通过／20 失败，计数保持 |
| 本轮 compiler／密态／API／安装／下载 | 全部 0 |

生产仍使用已有入口。前一批 common-call 的 alpha 和 eval dropout 两例已有真实 Poseidon
数值及程序包复用证据；布尔掩码 Attention 的原规则表达在累计 scale 735>720 处编译失败。
这些是 705 版本证据，不能改标为联合 706 版本通过。

单例新表达的离线密态续验 r1286、付费三例 r1232 仍待各自批准。
本轮没有新的批准收据，也没有运行批次。原算子、布局容量、bootstrap、整模型与 CUDA
里程碑保持独立，均未由此次任务接纳检查代替。

## 证据与下一入口

- `scripts/baseline/benchmarks/tools/source_component_acceptance_v1`：验收工具与只读审计。
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-component-admission-r1289`：原完整运行，保留 failed。
- 同目录 `agent-source-component-admission-r1290`：拒绝测试的定向补验。
- `audit.py`：再次核对原始证据、未修改方法、联合身份和保护文件。

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/source_component_acceptance_v1/audit.py
```

下一执行边界为 r1286 的单例离线验收。待联合编译／密态及包验收后，再按恢复方案切换生产；
本轮保持既有版本与全部本地修改，未提交或推送。
