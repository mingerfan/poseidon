# Agent 参数身份边界与 Poseidon 回归 r1328

日期：2026-10-09。开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

本批已接通独立参数读取模块，并完成代表性离线验收。旧参数函数入口保留，编译、执行及验证不再因参数读取而依赖 Provider 和生成控制层。

## 代码变化

- `application_planning/resources.py` 复用已准备的 `backend_parameters.py`，保留原有函数名称及接口。
- `application_planning/compiled_resources.py` 两处读取入口改为直接依赖 `backend_parameters.py`。
- `test_compiled_execution_precheck.py` 三处 mock 目标随模块位置调整，测试断言保持。
- 修改前源码、补丁及失败记录保存在 `benchmarks/tools/parameter_activation_v1/`。

参数读取核验 OperatorSpec、请求配置及来源哈希；继续使用 N=32768、13 个数据模数、输入 level=12、tc128。原生执行限制仍为 4 GiB／CPU 150 秒。Poseidon、CKKS Runtime、Dacapo、模型、reference 和误差门限保持。

## 验收结果

| 范围 | 结果 |
|---|---|
| 参数、资源、阶段身份、旧入口及错误拒绝测试 | 64 passed，0 failed／errors／skipped |
| Linear 4→2 | 真实编译、密态数值验收及程序包重复执行通过 |
| MLP 4→4→square→2 | 真实编译、密态数值验收及程序包重复执行通过 |
| 产物 | Hecate tracing、Earth／CKKS IR、RuntimePlan、OperatorSpec、plaintext bundle 均保留并核验 |
| 本批调用 | 2 次编译、6 次密态执行；付费 API 0 |

两个模型各用四类基础输入完成数值验收；程序包导出后，各用四组新输入执行两次。累计 24 组执行输入、48 个输出值，最大绝对误差 `2.297512715188077e-6`，逐项满足 `1e-5 + 1e-4*abs(reference)`。累计组数包含重复执行，独立的模型／输入组合为 16 组。新输入同时与原 golden 的 CPU float64 PyTorch 模型及独立数学解释器核对；执行器取得评估密钥，私钥未进入执行器。

候选来自确定性规则基线。上述结果属于当前编译执行链的离线回归，真实 Agent 新生成另计。

首次控制测试的 mock 路径错误，以及验收脚本为旧布局选择 explicit-v11 的入口拒绝，均保留原报告。脚本使用既有 explicit-v10 后继续；累计时间和空间仍受原批次预算约束。接入拒绝发生在编译前，没有重跑已经完成的后端阶段。

## 身份与历史记录

新阶段源码数量：生成 625 项，编译／执行／验证各 621 项；此前均为 691 项。三个后端阶段不再包含 Provider、凭据适配和生成控制模块。实际 frontend、compiler、executor、动态库和参数的身份与旧绑定一致，654 项后端及参考保护文件无变化。

当前完整身份：`e0a925044f029ad7133888b1f13c837c4ce85ca93774aa139514a1971457e35c`。源码改变产生了新阶段身份和新程序包；历史任务、报告与包继续保留原绑定。

原 r1317 两项 MHA 待批准方案已因源码变化失效，其 `--plan` 明确拒绝 `frozen_source_changed`。旧方案和请求保持原样；再次测试需在当前源码下重新冻结范围及身份，再申请有界付费授权。本批没有启动这两项。

## 证据与接续

机器报告：[agent-parameter-boundary-acceptance-r1328.json](agent-parameter-boundary-acceptance-r1328.json)。

原始验收目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-parameter-boundary-acceptance-r1328-v2`。本批新增产物约 29.24 MiB，未安装、下载、提交或推送。

下一条只读回归命令，在既有 Poseidon CPU Nix 环境中执行：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/parameter_activation_v1/controls.py
~~~

本批确认 ARM Poseidon CPU 的两个代表模型及重复执行。x86 保留原配置，本批没有 x86 实机运行；其他模型、真实 Agent 新绑定评测、bootstrap、CUDA 与整网工作按原计划继续。
