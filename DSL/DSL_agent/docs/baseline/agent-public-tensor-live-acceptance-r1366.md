# PyTorch v22 真实 Agent 验收

r1364 批次的 5 项真实 Agent 测试全部通过。4 项首轮成功，GELU→Linear 修复一次后成功；共 6 次生成、6 次 HTTP，全部调用已结算到终态。

| 模型组合 | 结果 | 生成次数 | 全部执行的最大绝对误差 |
|---|---|---:|---:|
| 布尔 mask 乘法 | 首轮通过 | 1 | 8.6680433942e-09 |
| 索引＋mask＋残差 | 首轮通过 | 1 | 6.4154981327e-09 |
| 按行 mask＋残差 | 首轮通过 | 1 | 4.6167807771e-09 |
| float32 BatchNorm（float64 计算） | 首轮通过 | 1 | 4.5237948665e-09 |
| 公开除法＋近似 GELU＋Linear | 修复 1 次通过 | 2 | 2.8229765192e-09 |

每项都完成 Hecate tracing、Earth/CKKS 编译、RuntimePlan／OperatorSpec／plaintext bundle 检查、Poseidon CPU 密态执行、最终解密和独立 reference 比较。每个程序包又以 3 组新输入运行，并重复执行其中 1 组；重新提交相同请求命中缓存，没有再次生成、编译或执行。

合计 5 次编译验收、15 次密态执行调用、40 组输入执行和 304 个输出值比较（含重复执行）。最大绝对误差 8.6680433942e-09，MAE 8.6686907891e-10，非零 reference 最大相对误差 1.8736404872e-07；有定义时最小 cosine similarity 0.99999999999999978。全部逐项满足 `1e-5 + 1e-4 * abs(reference)`。

## 生成与修复

GELU→Linear 首次候选在第 5 行使用保留名称，静态检查返回 `native_reserved_binding`。同一生成模型根据受控反馈重新生成，随后完成全部验收。原响应、失败诊断和修复响应均保留；首次失败发生在编译之前。

5 份交付源码与对应最终 Provider 响应逐字一致。验证适配器对这些源码执行编译重放，重放本身不调用 Provider；真实生成来源由应用调用账本、原始响应和程序包来源绑定共同记录。没有人工替换候选。

## 范围与配置

- 入口：显式 `poseidon-trusted-pytorch-v22`，`general-tensors-v1`，explicit-v13，P4／Poseidon w45。
- Provider：DeepSeek deepseek-flash／high／stream；API 和密态执行串行，编译最多 2 jobs、链接 1。
- 冻结预算：20 次生成／80 次 HTTP，每例 4 小时／全批 12 小时，新产物 4 GiB／全局 32 GiB／余量 4 GiB；本批实际使用 6／6。
- 平台：ARM Ubuntu，`/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- Python 3.10.14、NumPy 1.25.2、Torch 2.0.1 CPU；LLVM/MLIR 18.1.2、GCC 13.2.0，实际库与参数身份保存在机器可读报告。

float32 权重按声明的 `convert-float64-v1` 转换后验收；GELU 对照冻结的近似计算。原生 float32 执行精度及原函数应用精度分别保持未认证。

本报告覆盖这 5 个组合。旧 schema、x86 配置、CLI 默认 v18 和 latest=v21 保持；这次没有新增 bootstrap 或 CUDA 执行验收。Poseidon、CKKS Runtime、Dacapo/Hecate、模型、reference、安全参数和数值门限均未修改。

## 证据与资源

[机器可读验收](agent-public-tensor-live-acceptance-r1366.json)保留原始报告与每例程序包的 SHA、参数／后端身份、三段数值结果及生成来源。通过、失败、skipped 和未运行分别为 5／0／0／0；首次候选失败另外记录为 1。

- 原始运行目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-live-v1`。
- 方案绑定：`8752cc0b9e0fb913dedea3f717be0320e5f0339802cd046603a04a772a1a21e2`。
- 运行绑定：`5866752fb63995a07d9eee61e35724aec7dda983b16e9579defb3574bd4091b2`。
- 验收绑定：`c1be8d92e6152d74f62e2ad063f6b0e79b3664525aa86d2650776e12472b4a16`。
- 新产物：79.54 MiB；全局 results：28.283 GiB；客体磁盘余量：41.51 GiB。

宿主和客体网络配置未修改，没有安装、下载、提交或推送。账本计数不等于服务商签名或货币账单；本次没有明确货币上限。

## 后续工作

公开常量 reshape／permute／静态 slice／bool cast 的真实 FX 来源映射已另做开发草稿，尚待新来源版本接线及真实后端和 Agent 验收。继续通用输入、容量、算子组合、bootstrap 和整模型计划；既有历史失败及未知调用保留。
