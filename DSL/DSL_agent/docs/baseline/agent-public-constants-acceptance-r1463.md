# PyTorch v32 公开常量构造验收

公开常量构造已接入现有模型到 DSL 组件。选择 `poseidon-trusted-pytorch-v32`，可在 forward 中使用
`torch.zeros_like`、`torch.ones_like`、`torch.full_like`、
`Tensor.new_zeros`、`Tensor.new_ones`、`Tensor.new_full`。
形状、dtype 和填充值决定常量；常量随公开参数包绑定。
调用可与加密算术、真实 Linear、残差、结构化输入和命名输出组合。

## 结果

| 验收 | 通过 | 失败 / 跳过 / 未运行 | 真实编译 | Poseidon 密态调用 | 输入组 / 比较值 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| 独立明文与隔离来源捕获 | 21/21 | 0 / 0 / 0 | 0 | 0 | 336 / 1856 | 按 1e-12 绝对与相对门限 |
| 规则候选 | 21/21 | 0 / 0 / 0 | 21 | 63 | 168 / 928 | 6.594208062926654e-8 |
| 当前真实 Agent | 18/18 | 0 / 0 / 0 | 18 | 54 | 144 / 768 | 7.580106431426259e-9 |

270 项组件与接入控制测试、12 项执行预算与账本测试通过，失败和跳过均为 0。
规则候选包含 18 个新来源上下文和 3 个原 v31 兼容来源。
真实 Agent 使用 18 个不同冻结请求，18 次生成、18 次 HTTP，均首次通过，修复 0 次。
这些是来源表达式上下文，冻结 1,200 个模型集的结构计数保持原值。

所有候选完成真实 frontend tracing、Earth/CKKS 编译、RuntimePlan / OperatorSpec /
plaintext bundle 检查、Poseidon CPU 加密执行、解密及逐项数值比较。
数值门限为 `1e-5 + 1e-4*abs(reference)`。
每包另完成重载、3 组新输入、1 组重复输入和零生成／零编译缓存复用。
执行端使用评估密钥；最终候选源码逐字对应原 Provider 响应，两轨证据审计均通过。

## 范围

- 每种调用各 3 个上下文：逐项计算、Linear 与残差、双输入与嵌套命名输出。
- 支持静态公开形状与有限公开填充值，float32、float64、int32、int64、bool。
- 在分配前检查 rank、轴长度、单个张量和剩余公开常量总量；转换和舍入结果单独绑定。
- 本批真实密态形状为 rank 1/2。rank 32、公开标量、容量和错误声明另有控制测试。
- 默认来源 v18、latest v28 保持；v32 显式选择。JSON 和旧请求接口保持。

## 初始失败与修复

初始 v32 SDK `2d0e73e30ce43634d5957c59c35144ec50feed08a3c033363f37e49fd300b48c`
在生产隔离 worker 的路由中遗漏了 v32，首个 `zeros_like_0` 因走入旧捕获路径被拒绝。
原记录为 1 失败、20 未运行，API／编译／密态调用均为 0。

修复 `pytorch_ordered_intake.capture_request` 和 `public_options` 的显式版本路由后，
新增 worker 路由回归测试，并建立新冻结身份。
原失败报告、原 SDK 快照和源码保留；其余结论使用当前身份：
`a9ac477c3cc3e5470cf371c97da0962f6efc888e0c250641b66944efd6a600f3`。

## 绑定与资源

- 阶段绑定：`e86ecf6bfd54fa2481d11dabdf9a064815e2acca72e27e4d0acb1083bf70718b`。
- 规则审计：`4de5b3ca877824a4f73316c7e8a5a8c53004162569ad73db0b3257125ea675de`。
- 真实 Agent 审计：`5a17fd7ff93f5a1e24c61614dd7587f2c373716e9b14a06bf5d7a6806596b6c5`。
- 唯一 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，
  HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- SDK、请求、来源和依赖冻结已核验；任务终态成功、退出码 0、HTTP 全部结算，运行进程退出。
- 总产物上限 48 GiB、每批／每例 4 GiB、磁盘余量至少 4 GiB；收尾约 33.752/48 GiB，可用磁盘约 35.755 GiB。
- API／密态串行，编译最多 2、链接 1；原沙箱内存、CPU 和文件限制保持。
- Dacapo、CKKS Runtime、Poseidon、历史 SEAL 报告、参数、reference、x86 配置和网络保持。
  本批安装、下载、commit、push、PR 均为 0。

Provider 返回 usage：prompt 203272、completion 163447、total 366719 tokens；
reasoning 161173 包含在 completion 内，单独列示。18 个响应都有 usage。
金额未由接口返回；上述为运行器留存记录，费用以 Provider 账单为准。

## 结论与后续

已确认：本批来源、编译、真实密态、数值、程序包及真实 Agent 结果通过。
基于证据的推断：静态形状生成的公开常量可复用现有图计算，现有后端接口足以完成本批组合。
尚未验证：任意 Python、动态形状或填充值、GPU 构造设备、零元素张量、全部 rank 32 /
最大容量的密态计算、原模型应用精度、真实 CPU bootstrap、CUDA 实机和完整 ResNet20。

下一批继续公开 mask 的 where / masked_fill、公开与加密张量组装及更大权重／物理布局。
新增范围先冻结，用计划内测试总授权；新增依赖、后端／参数变更和资源扩大另行确认。

机器可读证据与全部报告路径见
[agent-public-constants-acceptance-r1463.json](agent-public-constants-acceptance-r1463.json)。
