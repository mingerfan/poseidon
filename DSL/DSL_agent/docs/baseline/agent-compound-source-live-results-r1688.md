# 复合近似源码：真实 Agent 验收

6 项全部通过，5 项首轮通过，1 项一次修复后通过。失败、未运行和跳过均为 0。

## 结果

| 上下文 | 生成次数 | 首轮 | 最终数值与包复用 |
|---|---:|---|---|
| Softmax 模块 | 1 | 通过 | 通过 |
| Softmax 函数＋输入变换 | 1 | 通过 | 通过 |
| Softmax Tensor 方法，rank 3 | 1 | 通过 | 通过 |
| LogSoftmax 模块 | 1 | 通过 | 通过 |
| LogSoftmax 函数＋输入变换 | 2 | 修复后通过 | 通过 |
| LogSoftmax Tensor 方法，rank 3 | 1 | 通过 | 通过 |

真实链路包含 Hecate tracing、Earth/CKKS IR、RuntimePlan / OperatorSpec / bundle、Poseidon CPU 密态执行、解密与逐项比较。六份源码均与原 Provider 响应一致，六份程序包通过新输入与重复执行。

资格验收 24 组输入、80 个输出值，最大绝对误差 1.09596e-07；复用 24 组、80 值，最大误差 6.41858e-07。门限保持 1e-5 + 1e-4*abs(reference)。

LogSoftmax 函数案例首轮完成编译和密态执行，但数值失败；原响应和原绑定保留。下一候选通过，未人工替换。

## 调用与身份

实际 7 次生成 / 7 次 HTTP，7 次编译成功，19 次密态调用（含一次数值失败、六次资格验收和十二次包执行）。包复用的生成与编译调用为 0。执行器仅获得评估密钥。

Provider total_tokens 223,910；reasoning 已包含在 completion 中。货币费用尚未核验。

核验 163 个产物文件、157 个包文件。累计结果 39.589/48 GiB，磁盘余量 29.676 GiB。

SDK：6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429
审计：051b047de472ceb65f574aa2751497e36d564df7dc91493010f6f576650af304
批次：e53b5abd22389dafb1bcaefef6d628d42c250c72643823034095a26278e7b9ac

本批证明声明的近似模型在所测输入上的正确执行；原函数应用精度尚未认证。原始来源、校准、近似配置、捕获、数值和程序包记录已保留。

## 后续

继续接入 GroupNorm / InstanceNorm 的输入统计与原生参数绑定，并推进应用精度、容量、算子组合、bootstrap、CUDA 和整模型目标。Poseidon、Runtime、Dacapo、参数、reference、门限和网络保持；无安装、下载、commit、push 或 PR。

[机器可读报告](agent-compound-source-live-results-r1688.json) · [准备接口](agent-compound-source-calibration-r1686.md)
