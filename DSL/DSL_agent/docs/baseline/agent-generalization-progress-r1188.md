# 通用输入接续（r1188）

更新时间：2026-10-07T16:43:46+08:00。ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。

## 输入层验收

| 项目 | 结果 | 验证范围 |
|---|---:|---|
| 新入口与旧格式兼容测试 | 106 通过，0 失败／错误／跳过 | 数据校验、来源绑定、参考与来源包 |
| v12 来源模型 | 10/10，160 组／1280 值 | 独立 CPU float64 参考 |
| 来源包重载 | 3/3 | Linear、归一化、公开 split 权重 |
| 正常隔离入口 | 3/3，48 组／272 值 | 真实 facade 捕获及来源包重载 |
| 错误架构 | 1 项拒绝 | ARM 实机选择 x86 配置 |
| 源码切换守卫 | 20 通过 | 原 76 项和调用账本 |

v12 接入代码保存在 scripts/baseline/benchmarks/tools/staged_ordered_intake_v1/targets，生产入口尚未切换。静态 Tensor view／split、别名及顺序写入在加载完整公开 NPY 权重后转为函数式图；输出保留 Tensor、静态 list／tuple／字符串键 dict 结构。

来源包保留原始图、修改前发射图、规范化图、分解图、节点映射和原始权重。重载时重新核对原始 NPY 及 transpose／split 派生常量。worker 检查实际平台和精确 Python／Torch／NumPy 版本，沙箱显式传入平台及 work root。正常入口复用了现有 portable Nix store，新增 Nix 启动为 0。

本批新代码的 Hecate tracing、编译、Poseidon 密态和真实 Agent 生成均为 0。v12 当前是已登记的有序张量子集，v11 全部 common-call 写法尚未组合接入；原始 float32 应用精度仍待认证。

## 原授权批次

原 76 项当前 **50 通过／15 失败／1 运行／10 未启动**。运行项：instance__unbatched1_2。当前目录为 /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-unstarted-continuation-r1174；绑定 0623167de71327bb1d2ec15a2d0563b74ee1c5bd1d35acf61f072027e2521c1b。已关闭任务累计 117 次生成／117 次 HTTP，运行预留另计。批准范围、修复次数、超时和总额度保持。

新增关闭范围独立审计 1 通过／2 失败。通过项 instance__unbatched1_0 完成真实编译、Poseidon CPU 密态数值比较、包重载和三组新输入，共 7 组／21 值，最大绝对误差 9.899662479e-10。累计 50 份原 Agent 包已审计，350 组／2387 值，最大误差沿用此前 1.0207768191e-6；审计复用原证据，没有新增生成或密态执行。

两项数值失败保留。instance__unbatched1_1 后两轮密态结果符合候选数学计算，候选与模型不同。instance__module2_input_2 使用多函数，现有直线数学诊断器不足以确定其数值失败根因。用尽额度项未重跑。

## 本地状态

生产 695 文件、执行工具 3 文件、历史证据 1375 文件、后端 648 文件及 reference／门限检查器 6 文件哈希一致。已有修改和未跟踪文件保留，新增仅 Agent 草案、开发测试和文档。安装、下载、后端与参数变更、提交和推送均为 0。原付费批次继续使用冻结源码。

r1180 的旧测试导入路径失败、r1184 的错误架构测试路径失败和较早的闭包断言失败均保留。修正入口后，r1187 与 r1186 在当前目标绑定下通过。切换守卫移除了旧 51 项历史数假设，改为核对历史＋当前恰好覆盖原 76 项；运行中返回 waiting_for_current_batch。

## 下一步

1. 继续原批准范围，审计新增关闭项；保留用尽额度的失败。
2. 原范围和账本关闭、Nix 锁释放后，联合验收反馈层与 v12 输入层。
3. 以新绑定验收代表性 Poseidon 编译、密态数值及程序包重载。
4. 新输入 Agent 质量评测单独冻结范围与预算；继续算子、容量、bootstrap 与整模型工作，CUDA 实机和 ResNet20 状态单列。

下一条安全检查命令：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/staged_binding_feedback_v1/guarded_apply.py
~~~

[机器记录](agent-generalization-progress-r1188.json)。草案绑定与复现入口见 scripts/baseline/benchmarks/tools/staged_ordered_intake_v1/README.md。
