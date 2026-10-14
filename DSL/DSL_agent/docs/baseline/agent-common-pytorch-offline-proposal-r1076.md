# PyTorch 兼容回归：r1076

范围沿用 r1042：151 项离线案例、25 份程序包复用检查，按 48/48/48/7 分片串行执行。模型、权重、参考、候选源码和数值门限保持。

新绑定包含四个新增 Agent 输出接口文件。固定 Poseidon 工具、库和参数保持。新 v4 启动器修正提案文件名，原 v3 驱动及 r1042 提案保留。

状态：已准备；没有当前批准收据，实际启动在进入 Nix 前拒绝，未执行。

| 项目 | 上限 |
|---|---:|
| 整批时间 | 2 小时 |
| 每例时间 | 180 秒 |
| 编译／链接／密态并发 | 2／1／1 |
| 新产物／全局产物 | 4 GiB／32 GiB |
| 磁盘余量 | 4 GiB |
| API／安装／下载／后端或参数改动 | 0 |

精确案例、哈希和范围见同名 JSON。程序包重载保持原有验收等级。此次范围是旧输入能力的后端回归，新增输出适配器的密态验收另列。

绑定：cda259c92d763f99d7f32dfad7efc1ec6f59d38bfd542cbc9f8b2c4bb9d7cbe0

批准后由可信宿主为此绑定建立 r1077 收据，再执行：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_model_source_backend_v4/launch.py
~~~
