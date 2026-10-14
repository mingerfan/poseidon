# Replicated scalar 表达与程序包验收（r1309）

三个原低阶近似 Attention 模型采用新规则表达后均真实编译通过。
布尔掩码样例完成数值验收、程序包重载、新输入和重复执行；
另外两个模型受原 4 GiB 执行资源边界阻塞。
本批是可信规则基线验收，真实 Agent 生成和生产 MHA 激活均未发生。

| 原案例 | tracing／编译／产物 | 密态数值 | 程序包 | 当前原因 |
|---|---|---|---|---|
| self + Linear + residual | 通过 | 未完成 | 未运行 | 第 671 条 Rescale 记录 std::bad_alloc |
| separate cross QKV + None | 通过 | 未完成 | 未运行 | 第 694 条 AddCP 记录 std::bad_alloc |
| boolean mask + padding + residual | 通过 | 通过 | 新绑定重载／新输入／重复执行通过 | 初次来源序列化失败已单独修复 |

两个资源失败的已知保留 payload 下界为 3.60／3.79 GiB，均不包含完整密钥、
分配器和工作区开销；实际峰值未知。退出码 124 后面还有分配失败日志，
不能仅记为超时或生成错误。编译器已消除可证明的局部重复操作，未为微小 CSE 重跑。

## 验收范围

最终 35 项控制通过：18 项规则与绑定控制、16 项完整来源／运输控制、
1 项“pv 含相同值仍不能授权标量字面量”的拒绝测试。失败、错误、skip 均为 0。
运输初次测试的 stage 导入配置错误及 16 项未运行状态另有原报告保留，
修正测试入口后使用新绑定，未覆盖旧记录。

布尔掩码样例先在四组输入上完成 40 值数值资格。
新程序包另完成三组新输入 30 值和一组重复输入 10 值。
合计 8 次输入组／80 值，其中一组为重复执行；最大绝对误差 **7.3737466133e-9**。
门限保持 `1e-5 + 1e-4*abs(reference)`。
reference 是原已声明近似图；原生 Softmax／应用精度仍未认证。

规则使用输入单元一次提取、复制到周期槽位，中间张量保留独立单元，
只在最终按原 period-4 chunks、offset、elements 打包。
缺少 pN 标量的权重来自原公开常量的精确 float64 值，绑定坐标及来源哈希；
原请求、常量 registry、权重、数学图、reference 和编译配置保持。
该规则侧能力尚未作为公开权重 getter 提供给真实 Agent。

## 程序包序列化修复

原完整来源校验在内存对象上通过，但导出 JSON 的排序把 weight_10 放到 weight_2 前面。
当前来源重建依赖字典声明顺序，造成 public／storage 编号不一致。
来源 canonical 值完全相同；问题在 Agent 序列化及来源重建边界。

显式 staging 适配器在新目录中保留原声明顺序，重新绑定来源文件字节和 manifest。
模型、候选、参数以及 48 个 Runtime 产物逐字节保持；旧失败包、报告和绑定不改。
原 check_provenance 两次及 check_bundle 完整校验保持，未跳过来源检查。
这证明修复包可用，生产 exporter 尚未接入该修复，MHA v16 仍为草案。

## 尚未完成

- 两个原多头／交叉 Attention 案例的运行资源问题。
- 生产导出的顺序稳定性和相应兼容回归。
- 此表达策略的真实 Agent 公开权重访问和生成指导。
- 池化、卷积、BatchNorm、Chebyshev 分支在本策略中的独立覆盖。
- 大权重来源账本的准备资源界；输入 primitive 下界不能证明总内存。
- 原计划中的其他算子、容量、bootstrap、整模型、ResNet20 与 CUDA。

下一批先审计混合 packed／replicated 表达的真实产物数量与内存下界，
再选择受影响范围验收。新增真实 Agent 测试仍需冻结案例与调用预算。

生产 685 文件、后端／reference／门限 654 文件核验保持。
本批 API／HTTP、安装、下载、后端和参数修改、commit／push 均为 0。
x86 配置保留，本机实际执行为 ARM CPU。

[机器报告](agent-replicated-scalar-results-r1309.json)

只读核验：
`python3 -B scripts/baseline/benchmarks/tools/staged_replicated_scalar_rule_v1/inspect_current.py`
