# PyTorch 空间入口 Poseidon CPU 验收

r1405 的 18 项离线验收全部通过。每项完成真实 Hecate tracing、Dacapo-modified 编译、RuntimePlan／OperatorSpec／常量 bundle 检查、Poseidon CPU 密态执行、解密和独立 reference 比较。18 份程序包重载通过，其中 3 份完成新输入、重复执行、命名输入一致性和缓存复用。

本批使用冻结规则候选，真实 Agent 新生成和 API 调用均为 0。

## 结果

| 项目 | 计划 | 通过 |
|---|---:|---:|
| resize | 6 | 6 |
| ConvTranspose | 4 | 4 |
| 组合 | 2 | 2 |
| 结构化输入与公开状态 | 3 | 3 |
| 旧接口兼容 | 3 | 3 |

合计 18 次编译、27 次密态调用，93 组次、1,094 个输出值。最大绝对误差为 `9.14513568931602e-07`。failed、blocked、skipped、not_run 均为 0。

数值门限保持：

~~~text
abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)
~~~

15 项使用 `poseidon-trusted-pytorch-v27`，2 项使用 v26，1 项使用旧 JSON schema 1。新增空间输入涵盖 1D／2D nearest、nearest-exact、linear／bilinear resize、模块和函数式 ConvTranspose、分组及静态参数，以及矩阵／残差组合。字典、嵌套关键字输入、float32 公开权重和命名输出中的 None 保留原有语义。

## 逐项结果

所有下列案例的编译、数值和程序包重载均通过。表中的值数对应初始四组输入；复用列包含额外的新输入检查。

| 案例 | 输入版本 | 初始比较值数 | 最大绝对误差 | 包复用 |
|---|---|---:|---:|---|
| resize_1_nearest_0 | poseidon-trusted-pytorch-v27 | 20 | 8.19759e-08 | — |
| resize_1_nearest_exact_1 | poseidon-trusted-pytorch-v27 | 20 | 9.36003e-08 | — |
| resize_1_linear_0 | poseidon-trusted-pytorch-v27 | 20 | 2.43871e-07 | — |
| resize_2_nearest_1 | poseidon-trusted-pytorch-v27 | 120 | 2.11702e-08 | — |
| resize_2_nearest_exact_0 | poseidon-trusted-pytorch-v27 | 96 | 5.52805e-08 | — |
| resize_2_bilinear_2 | poseidon-trusted-pytorch-v27 | 96 | 6.40958e-09 | — |
| transpose_1_0 | poseidon-trusted-pytorch-v27 | 56 | 7.94602e-08 | — |
| transpose_1_1 | poseidon-trusted-pytorch-v27 | 56 | 2.83677e-08 | — |
| transpose_2_0 | poseidon-trusted-pytorch-v27 | 120 | 1.77501e-08 | — |
| transpose_2_1 | poseidon-trusted-pytorch-v27 | 120 | 4.40695e-08 | — |
| composition_0 | poseidon-trusted-pytorch-v27 | 8 | 1.33208e-08 | — |
| composition_2 | poseidon-trusted-pytorch-v27 | 40 | 1.37765e-07 | — |
| structured_dict | poseidon-trusted-pytorch-v27 | 40 | 1.0474e-07 | 通过 |
| nested_keywords | poseidon-trusted-pytorch-v27 | 20 | 3.92543e-07 | 通过 |
| float32_public_weight | poseidon-trusted-pytorch-v27 | 28 | 1.27107e-08 | 通过 |
| compat_v26_keywords0 | poseidon-trusted-pytorch-v26 | 24 | 7.35667e-09 | — |
| compat_v26_nested0 | poseidon-trusted-pytorch-v26 | 48 | 3.38186e-09 | — |
| compat_v1_linear | JSON schema 1 | 8 | 3.97211e-08 | — |

复用的 `structured_dict`、`nested_keywords`、`float32_public_weight` 各完成三组新输入、一组重复输入和三组命名输入对照。程序包直接复用已编译产物，新增生成和编译均为 0。独立审计重新计算数学 reference，核对 694 个证据文件的哈希、模型来源、固定候选、公开权重和产物绑定。

## 环境和预算

开发与执行：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。P4／w45，后端身份与 r1400 相同。

| 项目 | 实际 | 批准上限 |
|---|---:|---:|
| 编译 | 18 | 18 |
| 密态调用 | 27 | 27 |
| 执行时间 | 700.56 秒 | worker 2,400 秒 |
| 新增产物 | 216.15 MiB | 512 MiB |
| 全局产物 | 30.22 GiB | 32 GiB |
| 磁盘余量 | 39.37 GiB | 至少 4 GiB |

编译最多 2 jobs、链接 1、密态串行，原 native 4 GiB／CPU150 秒限制保持。Poseidon、CKKS Runtime、Dacapo、参数、模型、reference、误差门限、依赖、x86 配置和网络保持。安装、下载、commit、push 均为 0。

首次独立审计遇到开发审计脚本的局部变量覆盖错误；修正仅限新审计工具，保留原脚本和失败日志。第二次审计通过，未增加编译、密态或 API 调用。该错误不属于模型或后端验收失败。

## 绑定与证据

- SDK：`5bd7c37b67d6e94cd28c6ad46660c17600f90e29b4c213fa5e41cc09d57239d6`
- 方案：`c563ee93d44018613d6ee4affc315414efacbace38a77fb5e81a60e2e9f7e592`
- 授权：`ee72e7b14c54ed9cca2c056bfe52d9616b1af442bc70d998b9336d83614d6e32`
- 执行报告：`a54a1cedeb9cb00c162967b116937a5f036de8ea4a7380c55cfabde3e2ac1a5d`
- 独立审计：`5732c87524838ff102f2496e5d3a4dce509ab4f52bd47d3fce57aea9beeee5c9`
- 本报告：`063b1e38b3f0298bc321583db1b1eb4a8e8a8b0d76c574494cb867bef8d993dc`
- 原始证据目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-spatial-source-offline-r1405`

[机器记录](agent-spatial-source-offline-results-r1406.json) · [额度结算](agent-spatial-source-offline-closure-r1406.json) · [原方案](agent-spatial-source-offline-proposal-r1405.md) · [输入用法](operator-decomposition-and-python.md#pytorch-v27-空间计算)

## 已确认范围和后续工作

已确认上述小型空间输入能够通过应用来源入口，完成 Poseidon CPU 数值验收并交付可复用程序包。三个仓库示例均有对应的真实后端证据。默认 v18、latest=v21、显式版本和旧接口保持。

新功能的真实 Agent 生成尚待单独冻结和批准。更大形状、容量、bootstrap 和 CUDA 仍分别验收。本批与 r1403 的 JSON 案例存在语义重叠，不合并计为唯一模型数量。r1404 接入快照、r1403 结果、原始失败和历史绑定保留。
