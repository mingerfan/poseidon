# 静态广播来源规范化草稿

将 Tensor.expand_as 和 Tensor.broadcast_to 显式规范化到现有 expand／broadcast 语义。保留原 FX 图、规范化图、节点来源、dtype 和视图别名信息；依赖公开输入签名及已登记的静态形状规则，拒绝数据相关 shape、错误广播、非法 rank 和容量。

草稿位于开发工具目录，生产入口尚未注册。验证只执行可信的小型 CPU PyTorch 模型，与原图及独立 NumPy 数学参考比较。覆盖 35 个上下文，包含组合、命名输出、关键字及可变参数、float32／float64、非连续视图和 rank32；另有静态 shape 查询及拒绝案例。此记录不代表编译、密态或真实 Agent 验收。

首次启动被当前真实 Agent 批次持有的 Nix launcher.lock 串行锁挡住；最小只读探针超时，日志保留。等待当前批次的安全边界后，用已有锁定环境执行 verify.py。未修改启动器、环境、网络或后端。

下一条命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/static_broadcast_source_v1/verify.py
~~~

通过独立检查后，再完成来源契约、沙箱、alias 检查和应用接入；该草稿不能独立作为用户模型的接受或安全判断依据。
