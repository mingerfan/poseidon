# 归一化规则表达验收 r746

原 r721 两层 LayerNorm 模型已由默认规则 lowering 编译并在 Poseidon CPU 上数值通过。模型、公开权重、近似配置、reference、w45 参数和数值门限保持原样；变化仅在 Agent 自有 DSL 表达生成中。

本轮是规则候选的离线验收。真实 Agent 新生成、原函数应用精度和整个计划的完成状态另行记录。

## 改动与适用范围

- 新增形状证明的两元素中心化表示。对 `x - mean(x)` 保留一份幅值与正负关系，在需要时生成第二个槽；方差归约复用这一表示。
- 对平方反馈图采用有界乘法重排，复用既有常量绑定。含 eps 的方差项保持原计算关系，公开 gain 优先乘到较浅的表达式。
- 选择依据是图结构、静态形状和已有常量；不依赖模型名称或 family。只处理可信规则源码，不修改 Provider 候选或 IR。
- 仅在 Poseidon CPU 的 `capacity-public-tensors-v3/v8` 路径启用。较旧 v7 源码、五份旧容量 fixture 及旧请求格式通过兼容检查；直接调用 `lower()` 默认沿用原表示。
- 归一化中心化优化当前限定两个逻辑元素。明文测试包含 rank 1–4 的单元素维度组合，密态代表例为 rank 1–3。其他形状继续使用原 lowering，执行能力由后续真实编译与数值检查判断。

新增两个实现模块、11 项相关测试及六例补充清单。原冻结 Benchmark 没有重新生成。另修正一条过时的测试错误消息断言，不改变容量限制。

## 验收结果

105 项静态、代数、参考与兼容测试通过，0 failed / errors / skipped。六例各完成 16 组独立数学与 CPU float64 PyTorch reference 比较；四个高层归一化定义另与分解图差分。

| 案例 | 输入 shape | 输出数 | Poseidon CPU 最大绝对误差 |
|---|---|---:|---:|
| 原两层 LayerNorm | [1,2] | 1 | 6.729e-10 |
| 非均匀 gain/bias LayerNorm | [2,1] | 1 | 1.359e-7 |
| GroupNorm | [1,1,2] | 1 | 2.646e-7 |
| InstanceNorm | [1,1,2] | 1 | 7.109e-8 |
| RMSNorm | [2] | 1 | 2.067e-9 |
| split → 中心化／方差／原分块 | [4] | 3 | 4.080e-7 |

六例均完成真实 Hecate tracing、Earth/CKKS 编译、RuntimePlan/OperatorSpec/bundle 核验、Poseidon CPU 密态执行、解密和四组输入逐项比较。6 passed / 0 failed / 0 not_run；全部为规则候选，新增 Agent/API/HTTP 为 0。

六份程序包已导出并核验依赖。原 LayerNorm 包另外完成一次重载，以及三组新输入的两次执行；重载和新输入都没有生成或编译调用。缓存复用返回 0 生成／0 HTTP。客户端新输入执行本身不声明数值认证；开发审计工具独立计算双参考并比较本轮输入。

最终共 **34 组密态执行、80 个比较值，最大绝对误差 4.08034037446825e-7**。冻结门限仍为：

`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`

## 绑定与历史

- 唯一 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`；分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- 当前批次绑定：`11d7b02e307ebebffff05a61b9a495f59539e44304fc740b7b06bcfbfd924ce5`。工具源码、模型文件、647 个 Agent 源文件和实际依赖均绑定。
- 独立审计复核 148 个程序包文件与 142 个执行产物哈希；648 个后端文件及 6 个 reference／数值／检查器文件保持。
- 原 r721 编译失败与人工表达记录保留。原模型 SHA 和 request_id 与本轮一致，当前源码及执行产物建立新绑定。
- 中间批次 r737 的三项通过、三项编译失败保留。随后只改 Agent 表达生成并重验受影响六例；没有改写旧报告。
- 开发测试的兼容性失败、Tier3 选择诊断和 r744 审计路径错误保留。r745 修正审计工具路径后复核既有 r743 产物，没有重新执行后端。
- 下载、安装、SDK 重建、参数／后端／reference／门限修改、commit/push/PR 均为 0。所有 worker 已结束。

完整机器报告：[r746 JSON](agent-normalization-lowering-acceptance-r746.json)。

本轮工作目录：

`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normalization-lowering-preflight-r731`

最终执行目录：

`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normalization-lowering-backend-r743`

## 后续

继续补静态公开索引、常见 PyTorch 写法和更大归一化组的表达／布局。新增输入能力的真实 Agent 评测需冻结新范围并单独批准。

Attention 深度、原函数应用精度、bootstrap、密文交接分区、CUDA、x86 原生、空白安装及 ResNet20 尚未完成。

无付费的程序包重载命令：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay --bundle \
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normalization-lowering-backend-r743/tasks/da720f86347843368a39a31a156cd2be/program --execute
~~~
