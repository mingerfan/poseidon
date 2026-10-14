# 空间算子 Poseidon CPU 验收

45/45 项通过，失败、阻塞、跳过和未运行均为 0。本批使用冻结的规则候选，API 和 HTTP 调用为 0。

| 范围 | 通过 |
|---|---:|
| 1D resize：nearest、nearest-exact、linear | 9/9 |
| 2D resize：nearest、nearest-exact、bilinear | 9/9 |
| ConvTranspose1d：形状、groups、stride、padding、dilation 等参数组合 | 9/9 |
| ConvTranspose2d：形状、groups、stride、padding、dilation 等参数组合 | 9/9 |
| 空间算子与矩阵、算术、分支组合 | 6/6 |
| 旧 JSON v1–v3 接口兼容 | 3/3 |

## 编译、执行与程序包

- 45 次真实 Hecate tracing 和 Dacapo-modified 编译；Earth / CKKS IR、RuntimePlan、OperatorSpec 及常量 bundle 均已核验。
- 51 次真实 Poseidon CPU 密态调用，包含每例四组固定输入，以及三份程序包的新输入和重复输入。
- 共 192 组次、3,496 个输出值；最大绝对误差 `4.861826131685731e-7`。逐项门限保持 `1e-5 + 1e-4 * abs(reference)`。
- 45 份程序包重载通过。1D linear resize、2D bilinear resize、分组 ConvTranspose2d 三包的新输入、重复执行和缓存复用通过；复用时生成及编译调用为 0。
- 独立核验直接读取 float64 解密数组和冻结 reference，核对模型、候选、参数与程序包文件哈希。原始数组保留在 Ubuntu 的工作目录。

## 批次与资源

r1400 为已批准的原方案。首例已通过编译、密态和数值验收，随后开发脚本误读来源字段，将包错误记为失败。程序包实际使用 `origin.model_source` 描述符。续验 r1402 仅修正开发工具，复用首例的原始资格记录，并将已用调用、时间与空间计入原预算。旧失败报告和冻结绑定保留。

累计用时 1489.81 秒（约 24.83 分钟）；新增产物 401.56 MiB。总产物 29.88 GiB，磁盘余量 39.75 GiB。45 编译／51 密态、40 分钟、512 MiB 新产物和 32 GiB 总量上限均满足；编译≤2、链接≤1、密态串行，原 native 4 GiB／CPU150 秒限制保持。

唯一开发位置 `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

- SDK：`388e891a44863d95e6a8ce7e83c4206799434e08e122c63ff58afa2b62452337`
- 原批准绑定：`86b1c249c3844f350de4e2dc2304b08994d24d5744712eb2cd37585a74928202`
- 续验绑定：`a78e40af6ef70ed143de82e4ecf54a6fbe77fb163c69e1696f6bebf3e7344bd3`
- 结果绑定：`0e6b9950005576d8b95fcc7706bca71c0c2e08c15dc8b46df9b590334ba6d7ba`
- 独立核验绑定：`8d606664b687c48519b0b9c6c291cd6aa18988663570b1e28aa24e528564d43f`

## 当前范围与下一步

JSON `poseidon-operator-graph-v4` 的这 42 个新增上下文和 3 个旧接口回归已完成 CPU 数值验收。其他形状、容量和参数仍按各算子及资源规则检查；本批候选来源为规则生成。

PyTorch 空间映射草稿已完成 27 个来源／组合上下文的明文差分，尚未接入生产来源版本。下一步将 Upsample／interpolate 和 ConvTranspose 模块、函数调用接入现有可信捕获入口，补拒绝测试、旧输入兼容和组合检查，再冻结新增真实 Agent 评测。

本轮仅修改验收工具和文档；Poseidon、Runtime、Dacapo、生产 SDK、模型、reference、参数、门限、x86 配置与网络保持。安装、下载、付费调用、commit、push 为 0。bootstrap、CUDA 和大型整模型继续按独立里程碑推进。

[机器记录](agent-spatial-offline-results-r1403.json) · [输入格式](operator-decomposition-and-python.md#json-空间计算图) · [原批准方案](agent-spatial-offline-proposal-r1400.md)
