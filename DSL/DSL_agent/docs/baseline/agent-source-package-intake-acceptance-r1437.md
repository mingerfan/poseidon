# PyTorch 源码包接入

显式版本 poseidon-trusted-pytorch-v29 接入常规 Python 包：文件清单、相对导入、嵌套模块和工厂函数重导出。数学计算沿用 v28；默认 v18、latest=v28。

| 验收范围 | 结果 |
|---|---:|
| 控制与旧接口兼容 | 301 passed，failed/errors/skipped 为 0 |
| 实际 CLI 清单构造 | 3/3 |
| 实际源码归档与权重绑定 | 3/3 |
| bubblewrap 内 PyTorch/FX 捕获及独立明文参考 | 3/3，48 组输入、224 个值 |
| 新身份编译、Poseidon 密态及程序包执行 | 待运行 |
| 新真实 Agent 生成 | 待运行 |

三个来源为 helper 平方组合、嵌套包关键字广播与多输出、包级工厂重导出加 Linear。原 PyTorch 与独立数学参考最大差值 5.551115123125783e-17；归一化图比较差值为 0。这里记录 CPU float64 明文结果。真实密态门限保持 1e-5 + 1e-4*abs(reference)。

已确认：源码包接入 CLI、应用快照、隔离挂载、来源归档、权重绑定和请求准备；三份独立请求均可按同一请求规则重构。六份旧请求字节和三个旧请求身份保持。后端、参数、依赖、x86 配置及网络保持，本批 API、编译和密态调用均为 0。

SDK 从 65e3d2514d448221fb61a1d410a812d2bd1908f483db892c63595b2a3ad0a7b2 更新为 05fcafb594be96bfcbed5fafbf21b818b74d261b97c507352e390bd83b79029b。四阶段来源闭包共涉及 11 个 Agent 文件；旧密态报告继续绑定历史 SDK。来源包在应用能力清单中登记，编译、密态及 Agent 资格保持未验证。

首次控制测试保留 287 passed、6 failed、7 errors，根因是 JSON 规则文件被错误归入 Python 模块名；修正后 301 项通过。两次开发启动器在 preexec_fn 前失败，涉及父进程文件/CPU 硬上限冲突。改为复用现有 sandbox limits 后三个捕获全部通过，原工具、日志和失败绑定均保留。生产沙箱仍使用 4 GiB 内存、CPU 150/155 秒、单文件 16 MiB、128 个文件描述符限制。

推断：三个请求已通过输入、参考与准备层，现有 Poseidon 编译链可继续验收它们。编译通过、密态正确和 Agent 生成成功仍需各自的实际证据。

待验范围：其他源码包组合、v29 的 Poseidon 编译与密态执行、程序包重载及真实 Agent 生成；x86 实机也待验证。namespace package、动态行为、bootstrap、CUDA 和通用化其余目标继续按各自状态处理。

产物目录 32 GiB 额度仅余约 1.96 MiB。三例后续离线及真实测试已按用户总授权冻结为 [执行清单](agent-source-package-validation-plan-r1438.json)，等待 r1434 的 32→48 GiB 产物预算回复。每例最多 3 轮修复，全批最多 12 次生成/48 次 HTTP，API 与密态串行，编译 2/link 1；预算、案例及身份以清单为准。

[使用说明](operator-decomposition-and-python.md#pytorch-v29-源码包) · [机器可读验收](agent-source-package-intake-acceptance-r1437.json)

开发 checkout：/home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。本批没有安装下载、后端修改、参数修改、付费调用或 Git 提交推送。
