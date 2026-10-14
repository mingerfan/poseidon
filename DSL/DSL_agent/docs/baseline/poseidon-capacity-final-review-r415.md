# 分块 Linear 最终复核

核心 ARM Poseidon CPU 链路保持已迁移。[主报告 r400](poseidon-capacity-linear-acceptance-r400.md)记录公开张量、跨块权重和 batched Linear 的验收；本报告补充最后两项组件兼容性修复。

## 修复与验收

- 同一个均匀公开张量用于广播及 Linear 时，保留其标量绑定，旧 v1 注册身份保持原值。
- 公开形状与权重绑定改为纯 Python。组件请求准备与程序包完整性检查可在系统 Python 中完成；实际 tracing、编译和执行使用锁定的 Nix 环境。
- 最终静态/组件测试 **114 passed / 0 failed / 0 errors / 0 skipped**。测试子进程明确禁止 NumPy/Torch 导入后，请求准备与重建通过。
- 新共享权重模型完成 CLI 密态验收；纯 Python 修复后，相同模型、DSL、request 和 reference 由组件重新编译并通过数值比较，4组/64值，最大绝对误差 1.0631442604047692e-07。
- 新绑定程序包经 CLI 重载，4组/64值通过，最大绝对误差 9.360720006762469e-08，生成与编译调用均为0。

真实执行为 Hecate → dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU → 解密逐项比较。当前 source 584、应用 527 文件绑定；二进制与后端身份和 r400 一致。

## 本阶段计数

跨明确源码绑定，共26次编译/密态通过，对应19个不同数学模型、104组输入、21,100个输出值；5个程序包重载通过，20组/5,248值。重复回归和重新绑定各记一次执行，模型数去重。新增的真实Agent生成数为0；候选来源为规则基线。

两个原矩形batched Linear仍在请求准备层阻塞，未进入编译，未计通过。bootstrap批准试验已执行，第二次有符号刷新15个值超过冻结门限，详见[r388](poseidon-bootstrap-ratio128-result-r388.md)。其余容量与深度缺口完整保留在[r400](poseidon-capacity-linear-acceptance-r400.md)。

## 历史与当前绑定

原共享权重包先因系统Python缺NumPy无法进入执行；修复后，该旧包正确拒绝验证器源码变化。原包、拒绝日志和第一次密态结果保留。新包由同一请求和原DSL重新编译建立当前绑定；没有改写旧包或旧通过记录。

r402/r403的测试断言错误、r407测试导入错误、一次导出路径错误也保留原日志；修复与重载结果单独记录。[机器可读报告](poseidon-capacity-final-review-r415.json)包含状态、路径、资源和全部绑定。

## 下一步

继续通用矩形batch映射压缩、跨块Conv/Pool、部分轴归约与非对齐连接，随后补v2构造/Agent生成及ResNet20物理布局。Bootstrap需要区分近似、归一化和密态噪声误差；新增参数试验另行批准。x86原生、空白安装与CUDA暂缓。

已验收的新包重载命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-shared-current-r412/program --execute
~~~

本阶段API、安装、下载、commit/push/PR均为0。Dacapo、Runtime核心、Poseidon库、Agent参数、reference及误差门限未改。
