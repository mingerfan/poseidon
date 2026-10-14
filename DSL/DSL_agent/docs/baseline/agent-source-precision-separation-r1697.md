# 原模型、近似计算与 CKKS 执行的分项验收

2026-10-14，ARM Ubuntu；lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。

12 份已有真实 Agent 程序包完成同输入验收：原生 PyTorch 与独立数学参考、近似图与分解图、密态输出与分解图均通过对应门限。原函数与近似计算的差异明显，当前近似配置尚不足以保证原模型精度。

## 结果

| 检查 | 通过 | 失败 | 最大绝对误差 |
|---|---:|---:|---:|
| 原生 PyTorch 与独立原函数参考 | 12 | 0 | 6.661e-16 |
| 声明近似与分解图 | 12 | 0 | 2.220e-16 |
| 密态输出与分解图 | 12 | 0 | 4.453e-07 |
| 原函数与声明近似，按数值门限诊断 | 0 | 12 | 1.58178 |
| 原函数与密态输出，按数值门限诊断 | 0 | 12 | 1.58178 |

每例 8 组输入，共 96 组、544 个输出值。样本包括零值、有符号值、边界值和 5 组固定种子随机值，原始数据保存在客体内的受限结果目录。

原函数与分解使用 1e-12 + 1e-12*abs(reference) 检查数学差分；CKKS 使用冻结门限 1e-5 + 1e-4*abs(reference)。后二行保留诊断失败，不能归入原模型精度通过。此次未指定应用精度指标，不计为整数据集准确率验收或形式化证明。

| 案例 | 原函数→近似误差 | 近似→分解误差 | 分解→密态误差 |
|---|---:|---:|---:|
| compound--soft_module_base | 0.0547118 | 1.110e-16 | 3.020e-09 |
| compound--soft_function_affine | 0.0249167 | 1.110e-16 | 6.712e-10 |
| compound--soft_method_rank3 | 0.0419743 | 1.110e-16 | 4.745e-08 |
| compound--log_module_base | 0.0202123 | 2.220e-16 | 3.437e-09 |
| compound--log_function_affine | 0.01125 | 2.220e-16 | 7.854e-08 |
| compound--log_method_rank3 | 0.00376655 | 2.220e-16 | 4.453e-07 |
| normalization--group_module | 1.53827 | 1.110e-16 | 1.834e-07 |
| normalization--group_function_affine | 1.58178 | 5.551e-17 | 2.833e-09 |
| normalization--group_module_batch | 1.5632 | 6.939e-18 | 1.245e-09 |
| normalization--instance_module1_unbatched | 1.51062 | 1.110e-16 | 2.568e-07 |
| normalization--instance_function_affine | 1.52179 | 5.551e-17 | 2.203e-09 |
| normalization--instance_module2_unbatched | 1.55967 | 6.939e-18 | 1.602e-09 |

## 原因与适用范围

已确认：原生 PyTorch 与独立原函数计算差分最大 6.661e-16；分解误差最大 2.220e-16；同输入 CKKS 误差最大 4.453e-07。所有逐项比较均从保留的原生结果、声明图和真实解密输出重新计算，结果与组件回执一致。

基于这些样本的推断：主要差距来自当前声明的近似配置。Softmax 使用 Taylor exp 和低迭代 reciprocal；LogSoftmax 使用低阶 log；GroupNorm / InstanceNorm 的 rsqrt 采用 Newton 2 次迭代、区间 [1e-4,4]。区间观测通过只能证明样本处于声明范围，不能保证近似误差足够小。

待验证：更合适的近似方法、区间与迭代数能否同时满足应用精度和现有深度资源。新方案须形成新请求与新绑定；本批原模型、权重、近似配置和 reference 均保持原样。测试集不参与拟合。

## 组件调用

原函数与近似图的明文评估使用 TaskApproximationAssessment；同输入的原函数、分解与密态评估使用 TaskNumericalAssessment，并显式选择 native_source。两个接口复用已有 Application 任务和程序包。

max_absolute_error 或 Top-1 指标由宿主明确给定。声明近似的 CKKS 验收与原模型应用指标分别返回；近似配置造成的精度失败应交由模型准备阶段处理，DSL 修复继续遵守已冻结数学契约。

## 执行与绑定

- 本批程序包复用 12 份，原 Agent 响应和源码保持。
- 新增 Agent 调用 0、编译 0；真实密态调用 12、隔离原生参考 worker 12。
- 前置明文评估另外调用 12 个隔离参考 worker；API、编译、密态调用均为 0。
- 330 个程序包文件哈希复核，保留 48 个本批原生参考证据文件。
- 密态执行为 Poseidon CPU；执行器私钥标记为 false。
- failed/not_run/skipped 在各分项中独立记录，原函数诊断失败保留为 12 项。

审计绑定：ae0af1987de46ecff3a7ca80f5f4630fc2519fccbf191c5c5089fc94b1fff484。
SDK：6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429。
原始结果：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-coupled-precision-assessment-r1696。
前置明文结果：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-approximation-comparison-r1695。
机器可读报告：agent-source-precision-separation-r1697.json。

累计原始证据 39.832/48 GiB，磁盘可用 29.431 GiB。本批同输入执行新增约 29.3 MiB。后端库、参数、网络与 Git 提交状态保持；安装、下载、付费调用、提交、推送均为 0。

下一步：补齐应用精度要求的控制层拒绝和终态回归；以新的显式配置研究近似质量、编译深度和资源之间的可行范围。容量、算子、bootstrap、CUDA 与整模型目标继续独立推进。
