# 默认规则生成：Poseidon CPU 验收

本批将通用表达调度接入默认 `--self-test` 路径。原九个归一化案例不再需要指定人工候选文件，原 RMSNorm 默认路径也已通过。模型、公开权重、request、reference、参数和误差门限保持原样。

| 验收 | 结果 |
| --- | --- |
| 静态、组件和旧接口测试 | 112 passed，0 failed，3 历史测试 skipped |
| LayerNorm / GroupNorm / InstanceNorm / 批量 matmul | 各 3/3 |
| 原 MLP / RMSNorm / Attention decode 默认路径 | 2/3；Attention 编译失败 |
| RMSNorm、LayerNorm 程序包重载 | 2/2 |
| 新真实 Agent 生成 / 付费调用 | 0 / 0 |

14 个通过案例共执行 56 组密态输入、比较 360 个输出值，最大绝对误差 **9.95399424441601e-7**。包重载另比较 40 个值，最大误差 **6.819776405403433e-8**；重载的生成、编译调用均为 0。最终串行批次耗时 131.47 秒，新增产物约 45.79 MiB。

所有数值结果均使用 `1e-5 + 1e-4*abs(reference)`。归一化验收针对已声明的 Newton 近似；九个补充模型相对原函数最大近似误差仍为 0.632650，尚未完成网络精度验收。历史 1,200 模型集保持原样。

## 实现

- `unified_graph_lowering.py`：形状匹配的整周期 sum/mean、广播标量和公开逐槽增益、平衡幂次多项式。
- `unified_graph_scheduling.py`：有界 DAG、公开因子前移、精确 one-hot 掩码恒等式、可信零密文所携公开因子复用、共享密文乘积提取。
- `test_unified_graph_scheduling.py`：独立参考、不同上下文、尾块/部分轴、旧配置及类型与预算拒绝。

调度只处理可信规则基线。Provider 与 replay 候选保留原源码，继续接受相同静态、编译和数值检查。算子判断依据类型、形状和图依赖，不检查模型名。深度统计用于诊断，实际可执行性由编译和后端验收决定。

完整解码路径为 Hecate → Dacapo-modified → RuntimePlan / bundle → CKKS Runtime → Poseidon CPU → 解密比较。Dacapo、Runtime、Poseidon 库及安全参数均未修改或重建。

## 原 Attention 默认失败

实际 frontend tracing 通过；第一次编译违规为 `earth.mul` 的两个输入 `ci<90 * 11>`，乘积需要 180 位 scale。Earth 消耗 level 11 对应 Poseidon physical level 1，即两个数据 Q 模数，模数乘积小于 `2^120`。未进入密态执行。

新调度将保守深度统计从 19 降至 15，仍未解决该容量问题。下一步是通用 packed 点积与分组多项式生成，保留独立 KV 输出分支。r242 的人工等价表达和重载结果单独保留；它们不是默认路径或真实 Agent 成功。

原 12 个高阶多项式模型的失败记录保持不变。真实 bootstrap 的精度失败、扩大物理布局和新 Poseidon Agent 评测分别处理。

## 源码与提案

应用核心 516 文件绑定；最终验收另绑定构建脚本、补丁和 frontend 依赖，共 573 文件。共享文件哈希一致，573 文件快照保存在 `results/poseidon-scheduling-final-r353/source.zip`。

旧 r286 付费提案保留原绑定，新 [r354 提案](poseidon-whole-models-live-proposal-r354.json) 绑定当前源码。三模型、公开 request、后端二进制及 12 次生成 / 48 次 HTTP 上限均未变，仍待批准。r235 bootstrap 配置也未运行。

r329 汇总遗漏两个“修复预算耗尽”终态的计数已另存审计记录；原报告和失败源码保留。

[机器报告](poseidon-default-scheduling-acceptance-r355.json) 记录逐例产物、错误位置、参数身份、源码和包绑定。唯一开发 checkout 为 ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。本批安装、下载、付费、commit、push 均为 0。

## 使用

在 Ubuntu checkout 中运行 RMSNorm 默认自检：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/poseidon-whole-models-v1/rmsnorm.json \
  --self-test --max-repairs 0 --compiler-configuration poseidon-cpu-eva-w45-v1
```

已有 RMSNorm 程序包直接重载：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay \
  --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-scheduled-packages-r352/rmsnorm/program \
  --execute
```
