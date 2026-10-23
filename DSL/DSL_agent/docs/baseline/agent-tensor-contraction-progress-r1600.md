# tensordot：离线验收通过，真实 Agent 测试进行中

显式 PyTorch v38 接入 `torch.tensordot`，保留旧格式、默认 v18 和 latest v34。静态整数、轴列表及固定公开 int32／int64 Tensor 参数进入已有张量与矩阵运算链路。

## 已完成

156 项兼容／安全检查通过，failed/skipped 为 0。六个原始 PyTorch 模型完成 96 组独立参考与来源重载。

六例规则程序的 Poseidon CPU 验收全部通过：6 次编译、18 次密态调用，48 组输入、136 个输出值，最大绝对误差 **1.29438450409e-8**。全部程序包完成重载、新输入、重复执行和缓存复用；只读独立审计通过。

模型为向量点积、矩阵乘积、公开 Tensor 负轴收缩、外积、单元素轴广播收缩及 mean →标量外积。[模型与范围](agent-tensor-contraction-progress-r1598.md)。

## 当前真实 Agent 批次

同六例已冻结并启动 r1596，沿用计划内总测试授权。DeepSeek flash/high/stream、explicit-v19、原 w45；每例最多 3 轮修复／每次最多 3 次传输重试，最多 **24 次生成／96 次 HTTP**、24 次编译／36 次密态调用。单次 1200 秒／384000 tokens，每例 4 小时／全批 12 小时，API 与密态串行。原 native 4 GiB／CPU150秒、编译 2／链接 1 保持。

总产物 48 GiB／每批 4 GiB／磁盘余量 4 GiB。只外发公开图、权重或描述符、布局、规则与受控反馈；原模型、reference 和门限保持。实际 Agent 结果待主批次及独立审计结算，当前不计为通过。

当前 SDK `929e81310d83822bb46e381eae409642a836a1ea05a20c0d9ff9afd361e10027`。主 PID979576/start28380336，后置 PID979577/start28380337；共享源码冻结至两者终态并核对调用账本。

[机器可读记录](agent-tensor-contraction-progress-r1600.json)。源码、测试、配置和文档均保存在 ARM Ubuntu checkout；后端库、参数、网络与 x86 配置保持。安装、下载、提交与推送为 0。更广参数组合、容量、应用精度、bootstrap、CUDA 和整模型仍按整体计划推进。
