# 有序 PyTorch 来源联合验收

日期：2026-10-07。Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl。

## 当前状态

V13 草案完成三例真实隔离 PyTorch 捕获及 reference 校验。
编译验收在共享锁获取处停止，尚未运行 Hecate tracing、编译、Poseidon 密态计算或程序包重载。
开发验收器已修复，29 项启动、绑定、预算与只读保护控制测试通过。
生产仍为 V12；原 704 个生产文件、648 个后端文件及 6 个 reference／门限文件保持。

| 范围 | 结果 |
|---|---|
| 常见调用与有序更新草案组件／明文／来源测试 | 前阶段 178 通过；本轮未重复 |
| 开发验收器控制测试 | 29 通过，失败／错误／跳过 0 |
| alpha、eval Dropout、布尔掩码 Attention 来源捕获 | 3 通过，独立 reference 与来源绑定核对通过 |
| 三例后端 qualification | 3 环境失败；编译与密态执行 0 |
| 新输入执行及缓存复用 | 6 阶段未运行 |
| 本轮 API、安装、下载、后端／参数修改 | 0 |

## 原因与修复

开发沙箱把 cache/agent-native-slots 挂成只读。资源池使用读写方式打开 slot-0.lock，导致三例在候选沙箱预检前停止。
现已仅向可信资源池开放现有锁目录，保留 SDK、其他缓存及历史结果的只读保护。
实际 bubblewrap 和资源池测试证明锁可获取、其他缓存仍拒绝写入；修复后的完整后端执行待续验。

启动器同时补齐显式 ARM 配置、bubblewrap 的参数文件描述符入口及分层只读挂载、原生工作盘绑定。
旧启动失败、原始请求、规则候选、日志和绑定全部保留。原30分钟窗口已经结束。

SDK 来源捕获使用 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1（CPU）。
控制测试使用 Ubuntu 系统 Python 3.10.12及标准库，其通过不计为同态数值通过。
三个候选来源均为规则基线，Agent生成次数为0。

## 续验

[同三例30分钟离线续验提案](agent-ordered-common-offline-retry-proposal-r1263.md)已冻结，待批准。
将复用同模型、权重、候选和reference，检查编译、Poseidon数值、新输入执行与缓存复用。
当前只读计划命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/runtime_acceptance.py --plan
~~~

r1232 三例新入口真实 Agent 提案另行待批准。原76项仍为56通过／20失败，145生成／145HTTP；旧结果保持原绑定。
V13尚未接入生产，近似函数相对原模型的应用精度、容量、bootstrap、CUDA及整模型继续单列推进。
