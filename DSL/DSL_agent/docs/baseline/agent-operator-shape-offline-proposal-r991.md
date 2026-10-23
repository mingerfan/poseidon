# PyTorch v10 后端离线验收方案

当前源码下共112项：保留原103项，包括八个原编译失败；新增九份已通过v10接入的来源。原103项的模型、公开权重、reference图、请求和规则源码逐项保持。46份受源码身份影响的PyTorch参考在新目录重新捕获，旧证据保留。

新来源为自适应平均池化三个上下文、InstanceNorm三个上下文和padding三个上下文。另对15份成功程序包检查重载、新输入两次执行及缓存复用。程序来自规则基线，本批不生成Agent程序。

| 预算 | 上限 |
|---|---:|
| 整批时间 | 2小时 |
| 每例时间 | 180秒 |
| 分片 | 48 / 48 / 16，顺序执行 |
| 密态并发 | 1 |
| 编译 / 链接 | 2 / 1 |
| 新增产物 | 4GiB |
| 全局产物 | 32GiB |
| 磁盘余量 | 4GiB |
| API、安装、下载、参数或后端修改 | 0 |

固定门限为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。检查实际Hecate tracing、编译、RuntimePlan、OperatorSpec、bundle、Poseidon CPU密态执行、解密和逐项数值比较。独立失败保留并继续其他案例；完整性、环境或预算故障停止相关范围并保留现场。达到整批预算时未运行项仍记为未运行。

精确案例、请求和哈希：`docs/baseline/agent-operator-shape-offline-proposal-r991.json`。
绑定：`32ade2c4bb66fc021f283e1264e79dad1fb12f68fb16e971bb99afb41ed3025d`。
已冻结684份源码、3171份案例/证据和四份驱动；未创建授权收据，未启动后端批次。

旧r976的103项提案因输入版本接入而过期，原文件和驱动保持；该旧提案的批准不能用于本方案。原r870的24项授权已执行关闭。本方案仍需本次明确批准。

批准并写入绑定收据后的入口：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_operator_shape_backend/launch.py
~~~
