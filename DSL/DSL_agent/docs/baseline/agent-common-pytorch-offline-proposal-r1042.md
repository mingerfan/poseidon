# PyTorch v11 与兼容回归离线验收

本批共 **151 项**：原 118 项保留原模型、请求和 DSL；新增 33 个 v11 来源上下文。原 r1025 未启动，当前源码已使其绑定失效。本方案建立新绑定，不复用 r870 的已执行授权。

| 范围 | 数量 |
|---|---:|
| 原 JSON、PyTorch 与应用接口任务 | 118 |
| 六组常见写法：MLP、CNN、残差、LayerNorm、Attention、Dropout | 18 |
| 投影 Attention：无 mask、公开布尔、有限加性、因果，各三个上下文 | 12 |
| 静态 alpha 与公开量组合 | 3 |
| 程序包重载、新输入重复执行、缓存复用 | 25 份 |

规则候选已准备。当前接入检查：179 项相关测试通过；公开 CLI 33/33、528 组明文参考；应用入口 8/8。55 个原 PyTorch 来源已重新捕获，另六个原来源接口重新准备；原 118 项数学模型、请求和 DSL 逐字节保持。

验收使用现有 Hecate → Dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU，检查四组输入及程序包复用。非线性案例验证显式近似图；原函数应用精度另计。全部为规则程序，真实 Agent 新生成为 0。

整批最长 2 小时、每例 180 秒，四片 48/48/48/7 连续串行。密态并发 1，编译≤2、链接≤1。新增产物≤4 GiB，总量≤32 GiB，磁盘余量≥4 GiB。API、安装、下载、后端与参数变更均为 0。时间、完整性或环境边界触发时保留现场，失败、阻塞、未运行分别记录。

数值门限保持 `1e-5 + 1e-4*abs(reference)`。检查输入、布局、真实编译、产物、密态执行、解密及逐项误差；旧报告和原失败保留。尚未取得本方案批准，执行未启动。

- 提案绑定：`ed3ffa719754bf75978d127e793c6c357516396e5302c3721ad40a08032f631d`
- 执行与案例清单：[JSON](agent-common-pytorch-offline-proposal-r1042.json)
- 当前范围：691 个源码、5 个上下文文件、5248 个案例/证据文件、8 个新旧驱动。

获此绑定的批准并建立收据后，唯一启动命令：

```bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_model_source_backend_v3/launch.py
```

审批收据位于 ARM work root 的 `results/agent-common-pytorch-offline-prepared-r1040/approved-backend-scope-r1043.json`，由批准记录生成。未批准时启动器在进入 Nix 前拒绝。
