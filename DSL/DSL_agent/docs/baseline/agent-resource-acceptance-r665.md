# 资源预检与万参数 Linear 验收

已完成可选资源预检接口及原万参数 Linear 的 Poseidon CPU 离线验收。所有修改在 ARM Ubuntu checkout；分支与 HEAD 保持。真实 Agent 的新增调用为 0。

## 实现

- `application_planning/resources.py`：复核请求、参数、源码和布局；记录逻辑图、公开数组、物理输入和存储下界。
- `scripts/analyze_resources.py`：模型 JSON 的布局与资源预检，支持复用已有布局计划。
- `assess_plan() → submit_planned()`：预检后委托原 Application，沿用其预算、取消、修复、验收和程序包流程。
- 独立开发工具验证 Poseidon 分配接口及大型随机权重案例；应用请求不读取实验编号、历史报告或人工探针。

当前参数为 N=32768、13 个数据模数、输入 level=12、两个密文分量。经现有 Poseidon 库核验，输入系数数据为 **6,815,744 字节／密文（6.5 MiB）**。检查器绑定公开 OperatorSpec 以及输入、分配代码哈希。可信辅助零另计。

预检预算用于输入密文数据下界筛选。原生运行器继续采用原有 4 GiB 地址空间上限。结果 `assessed` 表示已完成分析；执行可行性仍为 `unknown`，由后续编译与验收确认。密钥、中间密文、峰值内存和时间未变成静态保证。

## 验收

| 项目 | 结果 |
|---|---|
| 静态与兼容测试 | 51 passed，0 failed/errors/skipped：资源 19、布局 15、上下文 17 |
| Poseidon 分配接口 | 单个输入 6.5 MiB，分量 2、数据模数 13、系数 sizeof=8；两个独立分配。只验证分配，不算密态执行 |
| 资源 CLI | 正常预检、1 MiB 预算拒绝、已有布局重载均符合预期 |
| 100×100 Linear | 真实 tracing、编译、产物检查、Poseidon CPU 密态执行、解密与逐项比较通过 |
| 程序包 | 重载、3 个新输入各重复两次、缓存复用通过；重载／新输入无 API 或重新编译 |

万参数例沿用 r622 的模型、10,000 个不重复矩阵权重、100 个 bias 和请求，未修改模型或 reference。模型哈希 `2b8801feb08515dcedd97d71955ef2cd91fc228fa6c7e34b9052a30f54047207`，请求 `51ba092fa1dffd86cdb44a63993dff22f1f4d56f5bdbafcbe29013f3f65bb3d4`。

源码为**规则基线**，不是新 Agent 生成。候选 7,106 字节；输入周期 128，一个输入密文、一个辅助零、一个输出密文。输入数据下界 13 MiB。RuntimePlan 为 515 个 value、129 项初始化、384 项执行；实际输出声明 level=2、log₂(scale)=90，已经通过当前产物与数值检查。逻辑乘法路径不代表这份编译产物的 level 消耗。

- 首次：4 组、400 值，maximum absolute error `8.378680926846016e-8`。
- 重载：4 组、400 值，maximum absolute error `8.410158630622888e-8`。
- 新输入：3 组重复 2 次，600 值，maximum absolute error `8.690497836812139e-8`。
- 共 7 个不同输入、14 次组执行、1,400 次输出比较；不重复增加模型数量。
- 门限保持 `1e-5 + 1e-4 * abs(reference)`。两份独立明文实现交叉核验；执行器不获得私钥。
- 单模型完整验收 457.43 秒。开发观察器未记录到匹配进程，峰值内存仍未实测；该结果不作为内存测量成功。

## 来源与权限

完整机器报告：[JSON](agent-resource-acceptance-r665.json)。

原始证据位于当前平台 work root：

- `results/agent-resource-regressions-r659.log`
- `results/agent-resource-allocation-r660/`
- `results/agent-resource-large-weights-r661/`
- `results/agent-resource-cli-r663/`
- `results/agent-resource-boundary-r664/`

r656 的 18 项测试与最终测试重叠。r657 的分配探针 CKKS 命名空间错误、r658 的 unittest 多目录发现错误、r662 的 CLI 测试启动 PYTHONPATH 错误均保留；它们属于 Agent 开发工具，不是后端模型失败。修正没有涉及库或 SDK。旧请求、报告与绑定未重标。

核验 582 个后端文件及 3 个 reference／门限模块未改；分支 `lhy-agent-dsl`、HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e` 保持。新安装、下载、SDK 重建、参数调整、付费调用、commit、push、PR 均为 0。x86 配置保留；本次实际执行为 ARM。

r620 的 14 项真实 Agent 批次已经完成 14/14，本轮未重复。六项 v3 状态权重提案 r635 保持原冻结绑定 `d3377af18686f84d36e58a5b24b36407bf1513273c86cb8543d04d44627d7623`，仍待单独批准。

## 后续

继续补编译产物的内存／密钥估计、请求级路径注入、通用算子和未见组合。状态权重及新布局的真实 Agent 验收另按冻结范围执行。原 float32 应用精度、大图密文分区、bootstrap、CUDA、x86 原生、空白安装与完整 ResNet20 保留未完成状态。

使用入口：

~~~bash
python3 -B scripts/analyze_resources.py \
  --case scripts/baseline/application_planning/cases/square65.json \
  --output /absolute/path/to/new-resource-analysis.json
~~~
