# PyTorch 静态索引验收 r761

可信 PyTorch v6 已接入静态张量索引，保留 v1–v5、旧 JSON 和旧请求。15 个索引模型按两个源码绑定完成 Poseidon CPU 验收，另有两个重复输出回归。全部使用规则候选；新增真实 Agent 生成为 0。

## 输入与配置

| 分区 | 写法与范围 | 模型数 |
|---|---|---:|
| 静态切片 | 正步长、步幅切片、负边界、单个省略号 | 3 |
| 轴与单元素 | 整数／负整数、None、select、标量及嵌套输出 | 3 |
| 连续区间 | 函数／方法 narrow，静态整数 start/length | 3 |
| 公开向量 | 单轴整数列表或 int32/int64 状态 Tensor，支持重复位置 | 3 |
| index_select | 函数／方法、位置及关键字参数；非负公开索引 | 3 |

v6 沿用 v5 的工厂、NPY 状态权重、静态 shape、输出结构和逐节点近似配置。通用资源配置 rank 1–32，输入／输出各最多 4096 元素、中间张量最多 65536 元素；元素、节点、物理块及源码预算保持原值。密态代表模型为 rank 1–3，小张量。

加密数据相关索引、布尔掩码、空选择、负／零步长、多向量轴及同次选择中混用向量和整数返回明确拒绝。来源和公开状态参与哈希绑定；索引状态保留整数 dtype。

`app.submit_import` 对 v6 容量图默认选择 `capacity-public-tensors-v9`，显式旧选项保持。JSON 通用图的原 v8 默认保持；CLI 对导入图显式选择 `--unified-packing capacity-public-tensors-v9`。

## Agent 层适配

v9 复用已有 C-order 切片路由，准备并校验每个源／目标密文块、rotation 和 mask，继承 v8 的线性、BatchNorm、池化及归约配置。数学模型与布局、reference、公开绑定由可信代码固定。

重复返回同一张量时，Runtime 要求 final output value id 各不相同。可信规则 lowering 对可确认的重复表达增加受绑定的加密零输入加法。本次只处理规则源码；Provider 候选不经过该转换，IR 与 RuntimePlan 不作隐藏改写。

## 测试与真实执行

**184 passed / 0 failed / 0 errors / 0 skipped。** 覆盖静态选择、FX 和原模块、两套独立 reference、隔离导入、请求重建、路由篡改拒绝、旧 v8 请求及原归一化源码兼容。

15 个来源包各完成 16 组原 Module／数学／PyTorch／高层图参考检查。两个额外重复输出图分别完成 16 组独立参考。

- r754：15 例，14 passed、1 failed、0 not_run。失败为 `axes_2`，真实编译通过，Runtime 报 `duplicate final output`。
- r758：只补原失败例和两个不同上下文，3/3 通过。原 `axes_2` 模型与 request_id 不变，候选源码和执行绑定更新。
- 原通过 14 例的请求及候选源码在最新规则代码下逐字一致，保留 r754 原执行证据；未改写绑定或重新全量执行。

通过项均含 Hecate tracing、Earth／CKKS IR、RuntimePlan／OperatorSpec／bundle 核验、Poseidon CPU 密态执行、最终解密及四组输入逐项比较。参数保持 tc128、N=32768、Q13/P1；执行器没有私钥，bootstrap=false。

两批共 **128 组执行、866 个比较值**，包含重载和重复执行；最大绝对误差 **9.29913231973e-07**。门限保持：

`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`

两批各三份程序包完成重载、三组新输入的两次执行和缓存复用。当前源码三包为 `axes_2`、`duplicate_input`、`duplicate_square`；r754 三包的复用结果保留在原绑定下。重载、新输入和复用均为 0 生成／0 编译。新输入的数值比较由独立审计执行，客户端保留原验收边界。

## 绑定、资源与历史

| 批次 | 绑定 | 原记录 |
|---|---|---|
| r754 | `602c265ed7cbd17d7fc9d160146854b9983668541acbb241a3e5db256f20b243` | 14 passed / 1 failed |
| r758 | `258e55c656bc6ed30dd727d0c719bc50dacd233cb05370f488308de36aa8244d` | 3 passed / 0 failed |

648 个后端文件、6 个 reference／数值／检查器文件核验保持；当前冻结 654 个 Agent 源文件。两批核验 365 个程序包文件及 333 个执行产物哈希。

r751 的 v8 小切片准备阻塞、r754 Runtime 失败、r749/r752 测试失败和 r755/r759 审计工具失败全部保留。r749 为测试断言与真实输出树字段不符；r752 为规则文本插入位置错误，修复后回归通过。审计器按 canonical JSON 的键集合和序列表示核验；接口真实输出类型与顺序在执行 driver 中另有检查。

记录 worker 时间约 704.54 秒；新增产物约 73.48 MiB，全局约 22.34 GiB，余量约 46.17 GiB。维持编译≤2、链接≤1、密态串行，以及1小时累计worker／新增4 GiB／全局32 GiB／余量4 GiB边界。

付费/API/HTTP、安装、下载、SDK 重建、后端／参数／reference／门限修改和 commit/push/PR 均为 0。本批 worker 已结束。开发位置 ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

[机器报告](agent-pytorch-index-acceptance-r761.json) · [输入用法](operator-decomposition-and-python.md#静态张量索引)

## 后续

v6 新输入和重复输出的真实 Agent 评测另行冻结及批准。继续常见 PyTorch 写法、较大归一化组、权重和组件身份边界。Attention 深度、原函数应用精度、bootstrap、密文交接分区、CUDA、x86 原生、空白安装和 ResNet20 保留待验收状态。

当前源码的程序包重载命令：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay --bundle \
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-index-backend-r758/tasks/4c4b6b18bbe84da0b888fe01ae9bc438/program --execute
~~~
