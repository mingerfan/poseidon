# 统一模型提交接口：原型验收

当前生产应用仍使用 submit(JSON) 或 submit_import(已导入目录)。本批增加可接入的来源提交原型；待现有后端批次结束后再接入生产代码。

## 本批结果

| 检查 | 结果 |
|---|---|
| 组件测试 | 25 passed，0 failed / errors / skipped；注入后端，仅验证控制逻辑 |
| 实际来源导入 | 6/6：统一图 JSON、受限 Python 清单、显式可信 PyTorch 清单各 2 份 |
| 独立参考 | 96 组基础检查，导入模型与已有数学定义一致 |
| 来源快照 | 提交后修改调用方文件，捕获仍使用原快照 |
| 实际取消 | 观察到导入监督进程和 bubblewrap 后取消，未遗留子进程 |
| 实际资源限制 | 地址空间 4 GiB、CPU 150/155 秒、单文件 16 MiB、文件描述符 128、core 0 |
| 新编译 / 密态 / 解密 / Agent / API | 均为 0 |
| 生产接入 | 尚未进行 |

6 项是来源写法，不计作 6 个新增 Benchmark 模型。当前程序包和真实密态验收结论保持原样。

## 接口与实现

原型提供 submit_source(source, resource_profile=..., **requirements)。返回一个任务 ID，来源捕获、生成和结果查询共用它。提交只冻结来源；worker 才执行导入。工厂参数、权重声明、输入签名和输出结构沿用现有清单。

支持的来源是已经规范化的统一图或已登记的高层图、受限静态函数源码清单、明确授权的本地 PyTorch 清单。PyTorch 在现有 bubblewrap 中执行，Python 受限源码由静态解析器处理。

要求和控制规则：

- 后端、验收等级、生成选项和预算沿用现有组件；捕获 resource_profile 单独传递。
- 来源文件及权重在提交时快照并校验大小、SHA-256；状态只返回受控信息。
- 导入耗时从同一总预算扣减，生成次数及 HTTP 次数不增加。
- 取消覆盖捕获进程，丢失的 worker 须显式恢复，不能自动重做不确定任务。
- 原 JSON 的 request 保持；新来源控制身份单独绑定。缓存仍检查来源、依赖及程序包。
- 沿用组件来源记录上限；输入图容量不替代程序包或任务的存储规则。

源码位于 scripts/baseline/benchmarks/tools/model_source_component_prototype.py；不被正常应用路径导入。测试和实际导入审计位于同目录。它仍使用当前固定 worker/work root，生产集成时需登记身份和接口；本批不宣称已消除所有全局环境耦合。

首轮实际捕获的参考通过，但 4/6 在组件交接时拒绝。两处原型接口问题已修正：资源配置不能进入生成选项，32 MiB 图上限不能当作来源记录上限。首轮来源记录上限是首次失败点，生成选项也是潜在错误；原源码、日志和诊断修正都保留。后续仅重验受影响范围，最新结果见机器报告。

## 原八项编译失败复核

复用原 r859 请求、候选、响应和 compile.log，通过现有固定 Earth 诊断器重新核验，没有重新编译。

| 原模型 | 数量 | 已记录的失败检查 |
|---|---:|---|
| attention_core_0–2 | 3 | 累计 scale 780 > 固定预算 720 |
| channel_broadcast_mix_0–2 | 3 | 累计 scale 750 > 固定预算 720 |
| norm_depth_branches_1–2 | 2 | 累计 scale 750 > 固定预算 720 |

已确认：八项均完成旧 tracing，在 earth.mul 类型推导处拒绝；未得到编译产物、密态或数值结果。这里的 level 是 Earth 已消耗层数，不能直接当作 RuntimePlan/Poseidon 的数据模数索引。失败运算的占位返回类型也不是已经导出的非法 RuntimePlan level。

基于证据的推断：应继续检查候选的公开乘法链、布局掩码及深度需求；这些小张量的失败点不是单纯的输入元素上限。

尚未验证：等价表达能否通过、对应运行时容量以及 bootstrap 能否解决。旧规则候选的失败不能证明模型不可表达，也不是新的 Agent 失败。后续优先处理 Agent 层表达和布局，在批准范围内重新编译；后端库、参数和门限保持。

## 待验收范围

r991 的 112 项及 15 份包复用仍待单独批准。684 项运行源码、3171 项案例/证据、4 个冻结驱动一致；未创建批准收据或后端目录。r870 的 24 项授权已在 r878 执行关闭。

获批后下一条命令：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_operator_shape_backend/launch.py
~~~

本批保留 lhy-agent-dsl、HEAD 和 x86 配置。未安装、下载、构建 SDK，未修改后端、参数、模型、reference 或数值门限；未 commit / push。bootstrap、密文分区、物理容量、应用精度、ResNet20 和 CUDA 继续按各自状态推进。
