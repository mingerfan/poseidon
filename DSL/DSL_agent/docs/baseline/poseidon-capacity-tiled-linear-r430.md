# 矩形批量 Linear 分块验收

两个原容量阻塞已解决。核心 ARM Poseidon CPU 迁移状态保持已验收；本次扩展规则基线的物理布局，并完成组件与程序包回归。新 Agent 生成未运行。

## 模型范围

| 原案例或新增上下文 | 输入形状 | 输出形状 | 逻辑输入 / 输出元素 |
|---|---|---|---:|
| capacity_batched_linear_1，原案例 | [1,130,3] | [1,130,5] | 390 / 650 |
| capacity_batched_linear_2，原案例 | [1,2,64,4] | [1,2,64,6] | 512 / 768 |
| capacity_tiled_linear_2，新增上下文 | [1,1,65,5] | [1,1,65,7] | 325 / 455 |

三个模型都包含公开偏移及偏置。原两个案例的图、权重、reference 和测试输入不变。补充集新增一个数学模型，原 1200 模型集不变。

## 实现

公开布局分析使用二进制行位移，将每行补到统一宽度，再进行共享 Linear 映射，最后压回逻辑输出。跨块窗口和尾部使用绑定的公开掩码。生成基线复用真实 Hecate typed helper；模型 reference 与 DSL 构造分别实现。

新增 capacity-public-tensors-v3。v1/v2 请求按原版本精确重建；旧 v2 下的两个容量阻塞仍可复现。三个当前上下文的公开注册项分别为 57、43、58，golden 展开工作量分别为 780、587、485，保持 256 项及 1024 工作量上限。源程序、AST、模数深度和实际数值继续单独检查。

组件证据和导出包统一写入规范 JSON，保留 128 KiB 请求门禁。此前合法请求被缩进排版撑大，分别造成证据读取失败和导出包完整性失败；修正后包中的请求为 76,357、63,568、78,095 字节。请求语义与 request_id 保持不变，源码及产物建立新绑定。

## 验收结果

- 最终相关静态、双参考、布局、旧接口与组件测试：123 passed，0 failed/errors/skipped。三个新布局上下文各验证 16 组确定性明文探针。
- 首轮真实编译和数值比较：8 个不同模型，32 组输入，12,640 个值；最大绝对误差 8.36250e-7。范围含三个矩形上下文及既有 Linear、MLP、公开张量和 1024 输出回归。两项后续包导出失败单列。
- 当前最终源码下的三个矩形上下文：12 组密态输入，7,492 个值；最大绝对误差 6.99028e-7，全部满足冻结门限。
- 三个数值等级程序包均导出并重载。连同额外一次公开 CLI 复核，共 4 次重载、16 组输入、10,564 个值；最大绝对误差 6.28365e-7。重载生成调用和编译调用均为 0。
- 完整链路为真实 tracing、Earth/CKKS IR、RuntimePlan/OperatorSpec/常量 bundle、Poseidon CPU 密态执行、解密及逐项比较。参数仍为 N=32768、13 个数据 Q、1 个特殊 P、tc128，执行器不持有私钥，bootstrap=false。
- 数值门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。MAE、相对误差、cosine 及逐项输出保存在各案例证据中。

所有候选属于规则基线。上述数字不计为真实 Agent 生成成绩，也不证明任意矩形、所有容量算子或所有输入上的等价。

## 版本与保留记录

最终执行源码 587 文件、应用源码 530 文件。Dacapo、CKKS Runtime、Poseidon 库及参数身份保持不变。各批次独立绑定，旧结果保持原版本。

| 记录 | 状态及作用 |
|---|---|
| r417 / r418 / r419 | 编辑语法与版本错误、工作量拒绝、helper 绑定未完成；保留失败日志 |
| r420 / r421 / r423 / r425 | 分别 114 / 121 / 122 / 123 项静态通过，最终采用 r425 |
| r422 | 8/8 数值通过，2 项包导出因请求文件排版超过门禁失败 |
| r424 | 4/4 数值通过，2 项包导出仍因导出端排版失败 |
| r426 | 修正两端写入后，两项编译、数值、包重载通过 |
| r428 / r429 | 第一矩形上下文在最终源码下编译、数值和包 CLI 重载通过 |
| r427 | 既有新包通过公开 CLI 再执行 |

原始证据位于 ARM work root 的 results/poseidon-capacity-tiles-*；具体路径、请求、源码、模型、后端身份及误差见 [JSON 报告](poseidon-capacity-tiled-linear-r430.json)。修改前快照为 r416，最终源码与文档快照为 r430。

开发仍仅在 /home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。既有本地修改与未跟踪文件保留。本轮安装、下载、付费 API、commit、push、PR 均为 0，所有执行批次已结束。

## 下一步

继续跨块部分轴归约、非对齐连接及 Conv/Pool。新容量的 Agent 自由生成和定向构造仍需独立验收；旧三个整模型的已批额度未使用，启动前需核对当前绑定。12 项深多项式编译失败、bootstrap 第二次刷新精度失败及 ResNet20 物理容量与整网精度分别保留。x86 原生、空白机器安装、CUDA 暂缓。

当前无付费的包重载命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tiles-reload-r426/capacity_batched_linear_2/program --execute
~~~
