# PyTorch 静态 split 与 Poseidon CPU 验收 r878

## 结果

静态 torch.split／Tensor.split 接入已完成。24 项全部通过隔离捕获、独立参考、真实 Hecate tracing、Dacapo 编译、RuntimePlan／OperatorSpec／常量 bundle 检查、Poseidon CPU 密态执行和逐项数值比较。

| 验收 | passed | failed | skipped / 未运行 |
|---|---:|---:|---:|
| 相关单元和兼容测试 | 196 | 0 | 0 |
| 新 split 来源上下文 | 18 | 0 | 0 |
| 原失败的 split→stack Python 来源 | 3 | 0 | 0 |
| 矩阵、InstanceNorm、CNN 兼容回归 | 3 | 0 | 0 |
| 程序包重载、新输入重复执行和缓存复用 | 3 | 0 | 0 |
| 新真实 Agent 生成 | 0 | 0 | 24 未运行 |

所有候选均为规则程序，真实 Agent 新生成、API 和 HTTP 为0。原三项的模型、公开权重和来源未修改；此前人工等价 JSON 与原失败记录保留，新结果为当前源码下的真实 PyTorch 捕获和规则程序通过。

126 组输入、1,558 个输出值，maximum absolute error 为 5.9262181784092149e-07。门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。三份程序包重载使用原四组输入，三组新输入各执行两次，均通过逐项独立比较；缓存复用时生成与重新编译为0。

## 范围

| 来源写法 | 上下文数 |
|---|---:|
| torch.split 的显式 sections | 3 |
| Tensor.split 整数 size 与尾块 | 3 |
| 静态关键字与负轴，含 rank32 小张量 | 3 |
| 直接 tuple 输出 | 3 |
| 五至七个内部端口与公开循环 | 3 |
| 非重复公开 NPY 权重的分块 Linear | 3 |

拆分复用已有 split/slice 与布局，不改变后端语义。384 组原图／分解图独立参考通过；原三项另有48组与原人工等价图的差分。36 旧矩阵捕获逐字一致，41 个来源文件可复现。48 项来源身份拒绝、777 包文件和729编译产物哈希通过。

捕获与候选均保留独立沙箱。真实候选探针确认网络隔离、只读请求及既有资源限制；执行端仅获得评估密钥。N=32768、Q13/P1、tc128 和 w45 配置保持。

## 时间、修改与边界

获准的独立后端窗口执行 10.13 分钟，低于30分钟。新增产物约 170.4 MiB，全局约 24.77 GiB，余量约 45.25 GiB，均在批准范围内。

分支 lhy-agent-dsl、HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e 保持，原 1777 条修改／未跟踪状态保留。673 源码、648 后端、6 reference 和12历史文件核验。新增 helper、捕获适配、测试、来源生成器和说明保存在 Ubuntu checkout；运行产物保存在 ARM work root。安装、下载、SDK重建、后端／参数／reference／门限修改、Git提交和推送均0。

原一小时工作窗口的异常墙钟记录保持，原因未确认；本批使用新批准的独立窗口，未改写旧时间或历史结论。静态快照 r872 仍按当时未执行后端的状态保留。

## 剩余工作

八项原组合仍停在编译容量：attention_core_0–2、channel_broadcast_mix_0–2、norm_depth_branches_1/2。本批未重跑它们，也不将其判为算子不可表达。后续优先审查 Agent 层的布局与合法表达指导，后端库和参数保持。

r848 付费提案未执行，按原绑定保留；源码已经变化，当前真实 Agent 评测需重新冻结请求后单独授权。Bootstrap、CUDA、x86原生、空白安装、应用精度和完整 ResNet20 未新增通过。

[机器报告](agent-pytorch-split-acceptance-r878.json) · [张量拆分用法](operator-decomposition-and-python.md#pytorch-张量拆分) · [来源包](../../scripts/baseline/cases/trusted-pytorch-split-v1/)。
