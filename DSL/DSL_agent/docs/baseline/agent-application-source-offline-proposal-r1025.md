# 统一来源与既有输入的 Poseidon 离线验收 r1025

当前提案绑定：`f958c0074c4554f6254b29a68064edc285db51fefd1166abe3aa2704b789386d`。

本批只执行规则候选，验证 Agent 组件和现有 Poseidon CPU 后端的对接。生成和 HTTP 调用均为 0。

## 案例

保留 [r991 原 112 项](agent-operator-shape-offline-proposal-r991.md) 的模型、公开权重、reference、请求和候选源码，包含八个原编译失败。新增以下六个统一来源接口上下文：

| ID | 来源 |
|---|---|
| source_graph_adaptive_pool_0_functional_query | JSON 图 |
| source_pytorch_adaptive_pool_0_functional_query | 可信 PyTorch |
| source_graph_instance_norm_2_functional_query | JSON 图 |
| source_pytorch_instance_norm_2_functional_query | 可信 PyTorch |
| source_restricted_2 | 受限双文件 square，2 元素 |
| source_restricted_4 | 受限双文件 square，4 元素 |

总计 118 个验收任务，分片 48／48／22，串行执行。六个来源上下文包含相同计算的不同入口，按接口上下文计数。18 个成功程序包检查重载、三个新输入的两次执行，以及缓存复用；失败原项的复用检查记为未运行。

完整案例、源码、依赖和请求哈希在 [JSON 清单](agent-application-source-offline-proposal-r1025.json)。旧 r991、r1021 和全部原证据保留；旧提案不能启动当前源码。

## 预算与验收

| 项目 | 上限 |
|---|---|
| 整批时间 | 2 小时 |
| 单案例 | 180 秒，包含接入和验证 |
| 编译／链接 | 2／1 |
| 密态并发 | 1 |
| 新增产物 | 4 GiB |
| ARM results 总量 | 32 GiB |
| 磁盘余量 | 至少 4 GiB |
| API／安装／下载／SDK 重建 | 0 |

当前 results 表观占用约 23.64 GiB，可用磁盘约 45.00 GiB。执行前重新记录占用，运行中检查实际新增和全局空间；触及边界保留现场。

验证真实 Hecate tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、输入输出绑定、Poseidon CPU 密态执行、最终解密和逐项比较。门限保持 `1e-5 + 1e-4*abs(reference)`。现有 N=32768、Q13/P1、tc128、w45 参数与库保持。

独立审计复算数值比较，单列输入拒绝、编译失败、环境异常和未运行；规则程序的结果不计为真实 Agent 新生成。统一来源程序包额外核验 `model-source.json`，任务预算与模型来源复用身份分别检查。

授权后创建绑定本提案的范围收据，再运行：

```bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_model_source_backend_v2/launch.py
```

未获批准的启动已验证在进入 Nix 和产生 worker 前拒绝。当前未创建批准收据或后端目录。
