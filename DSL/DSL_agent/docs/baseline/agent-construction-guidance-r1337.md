# Agent 构造成本指导与自动反馈 r1337

新增显式生成指导 `explicit-v13`，复用既有应用提交、生成、预算内修复和 Poseidon 验证路径。离线工程验收通过；该版本的真实模型生成效果尚未评测。

## 改动

- 根据实际布局列出旋转步数、保留绑定及构造工作量限制。
- native helper 的工作量从全部公开常量绑定数起算，每次调用累加被调用函数的展开成本。公开循环展开后计数；局部结果复用可以减少重复调用。
- 静态失败时重新运行已有 AST 分析，返回最大成本的三个 helper 位置、调用次数和展开成本。数字与可信拒绝诊断的原因、位置和上下文一致后才进入反馈。
- 原 `1024` golden 工作量、`4096` native 模块/AST 限制、`65536` 源码字节限制保持。构造工作量与密文操作数、乘法深度、内存分别统计。
- 修复普通统一图在 v12 请求准备时缺少 `elements` 字段的问题，元素数从已校验的 shape 计算。25 份既有有效请求内容保持，旧默认及 v1–12 继续兼容。
- 两份沙箱可信模块清单绑定新指导；AST、隔离、数值检查和原有执行资源限制保持。

历史 self Attention 的两份无效源码具有105个公开常量绑定，分别调用22、26次 native helper。已有检查器报告2766、3205工作量，其中直接重复的常量注册计数为2310、2730。新指导提前说明计数规则，并在失败反馈中给出调用位置。四份原失败源码仍按原规则拒绝；源码及历史记录保持。

## 使用

应用提交选项：

~~~python
task_id = app.submit(
    model,
    options={"generation_guidance": "explicit-v13", "chunk_period": 4},
    limits=limits,
)
~~~

CLI 对统一图输入使用 `--unified-guidance explicit-v13`。准备公开请求：

~~~sh
python3 scripts/agent.py --backend local candidate --   --case model.json --unified-guidance explicit-v13 --prepare
~~~

`model.json` 是用户的统一声明式图。`explicit-v13` 是生成指导版本，PyTorch 来源格式与模型声明版本分别保留。自由生成仍由同一个 Provider 完成；本改动只提供确定性规则和反馈。

## 验收结果

静态、组件及兼容测试：**103 passed / 0 failed / 0 errors / 0 skipped**。25份旧请求与两份历史付费请求保持；构造计数器字节保持，654项后端及参考文件匹配。

真实 Poseidon CPU：

| 案例 | 源码来源 | 初次数值最大绝对误差 | 包重载新输入最大绝对误差 |
|---|---|---:|---:|
| 分组均值中心化 | 规则基线 | 8.2703e-9 | 3.4894e-9 |
| Linear → transpose → sum | 规则基线 | 1.8325e-8 | 5.4483e-9 |
| masked/padding MHA | 未改历史真实 Agent 源码 | 7.4306e-9 | 4.6569e-9 |

3次真实 tracing/编译，6次密态执行；24组输入、128个逐项比较值，max abs **1.832472112939243e-8**。三份程序包均检查并重载，各执行四组新输入，复用时生成与编译调用均为0。两套独立明文参考在每例八组输入上比较一致。产物包括 Earth/CKKS IR、RuntimePlan、OperatorSpec、plaintext bundle 和输入输出绑定。

门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。该结果验证当前有限案例的工程链路。历史 Agent 源码重放与规则基线分别记录；当前新生成的真实 Agent 程序数量为0。

## 源码与证据

- 唯一开发 checkout：`/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。
- HEAD：`7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- 当前阶段身份：`6312151da9f93d55509afa80e21ff903f41dbb7191bc3557413367d0b8784f55`。生成629项；编译、执行、验证各623项。
- 控制报告：`scripts/baseline/benchmarks/tools/construction_feedback_v1/controls.json`，绑定 `998b948b88d5f664616ad4d7da8ab5e15d84398d93775d4a2740106145a0dd83`。
- 后端报告：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-construction-guidance-acceptance-v1/report.json`。
- 本轮汇总绑定：`47f05046949d7c78e3d28f8393350a1ec6d83bbf7f89f6df2774372f108ee127`。
- [机器报告](agent-construction-guidance-r1337.json)保存源码差异、请求、包及验收绑定。
- 新增结果约34.21 MiB，执行106.37秒；全局结果约25.75 GiB，磁盘余量约42.64 GiB。

原工具命名、测试夹具及验收启动器的问题已保留独立记录；最终控制报告和后端报告绑定当前实现。旧模型、权重、reference、安全参数和门限保持。Poseidon、CKKS Runtime、Dacapo及现有补丁保持，SDK未重建。新增API、安装、下载、提交和推送均为0。x86配置保留，本轮真实执行在ARM Ubuntu。

## 后续

两项剩余 MHA 的 v13 请求已完成离线准备，公开上下文分别126865、131062字节，均在131072字节内。数学模型与原请求相同，生成指导与请求身份更新。真实 Agent 重测需要新的具体授权；r1333批次已结束，旧中断调用的结果和费用仍未知，保留其记录。

继续完成通用模型、容量和应用精度工作；跨命令总体截止时间及运行中取消、真实bootstrap、密文分区、CUDA实机和整网目标仍待推进。v13指导改进不替代这些验收。

下一条只读命令：

~~~sh
python3 scripts/agent.py --backend local candidate -- --help
~~~
