# 工作上下文与程序包验收

新增固定 worker 和单次请求上下文。宿主显式绑定平台、SDK、源码、动态库配置和工作路径，再使用已有 Application 的提交、状态、运行及程序包接口。开发和验收位置为 ARM Ubuntu /home/lhohy/Code/Poseidon，分支 lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。

## 接口

在已配置的 Nix/Python worker 中：

~~~python
from application_execution.context import WorkerContext
from application_execution.backend import application

worker = WorkerContext.capture(host_owned_task_store)
app = application(worker)
task_id = app.submit_import(imported_directory, options={
    "chunk_period": 4,
    "public_weight_policy": "descriptors-v1",
})
outcome = app.run(task_id, trusted_provider)
outputs = app.execute(task_id, samples)
~~~

WorkerContext 绑定实际平台、Python 3.10.14、NumPy 1.25.2、ARM Torch 2.0.1 CPU、既有 SDK、源码闭包及路径。RequestContext 绑定完整请求、布局、编译配置和验收等级。编译、程序包交付和新输入执行前后核验一致性；环境漂移停止当前请求。

原 Application、Provider、Controller、修复策略、取消、预算和程序包接口继续使用。上下文记录保存在宿主私有任务目录，权限为 0600。程序包沿用 v5/v6 与原编译、执行、验证身份，旧 CLI 和请求规则保持。

## 验收结果

| 检查 | 结果 |
|---|---|
| 上下文、请求篡改、配置漂移、路径、预算、取消、交付边界 | 17 passed，0 failed/errors/skipped |
| 旧 schema Linear CLI 准备 | 通过，生成/API/编译/密态执行 0 |
| 三个规则候选 | 3/3 真实 tracing、编译、Poseidon CPU 密态、解密及独立比较通过 |
| 四组基础输入 | 12 组、68 个值，max abs 1.5153387e-08 |
| 三个程序包重载 | 12 组、68 个值，max abs 1.0172139e-08 |
| 新输入重复执行 | 9 组新输入 × 2，18 执行组、102 个值，max abs 5.861965e-08 |
| 原程序包缓存复用 | 3/3，新增生成/HTTP/编译 0 |

案例为公开 NPY 权重 Linear、外部 state_dict 平方 MLP、多输入多输出矩阵融合。新输入分别通过独立数学解释器与 float64 PyTorch reference，再与密态输出逐项比较。门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。MAE、最大绝对误差、非零 reference 相对误差及有定义的 cosine 保存在[机器报告](agent-context-acceptance-r655.json)。

本批三个程序使用规则候选；新增付费调用和新 Agent 程序均为 0。此前 14 项真实 Agent 结果仍单列为 [r626](agent-generalization-live-acceptance-r626.md)。六项外部状态权重的新真实生成提案 r635 仍待单独批准，绑定未改变。

## 历史错误

- r648 的一个测试读取了错误的字段层级。修正测试后补验该项，原实现与另外 16 个通过测试保持。
- r650 在开始模型验收前因开发工具导入路径失败。修正工具导入，r651 完成三例；模型、候选、后端保持。
- r652 的旧 CLI 与验收批次争用 Nix 启动器互斥锁，30 秒后退出。批次结束后顺序执行 r653，1.50 秒准备成功。

原日志和修改前文件保存在 work root，未将这些错误记为模型通过或模型失败。

## 当前边界

已确认固定上下文拒绝配置漂移，原组件控制层与程序包流程可正常工作。582 个后端文件、3 个独立 reference/数值模块、分支、HEAD 和 r635 提案绑定已复核。安装、下载、SDK 重建、参数调整、commit/push 均为 0。

底层运行器仍使用绑定的全局路径，编译与执行证据位于声明的 WORK/results；完整路径注入尚未完成。源身份继续采用保守模块闭包。编译前的峰值内存、运行时间、深度及中间密文数量仍有未知项。

通用算子、较大状态权重、原生 float32 应用精度、密文分区、bootstrap、CUDA 实机、x86 原生、空白机器安装和 ResNet20 保持独立状态。三例通过只证明本次上下文与代表性流程。

## 记录与下一行动

- 原始验收：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-context-numerical-r651
- 规则批次绑定：e005ff9b9f61f0fb690816cf06c0d9e3d4249e8f1d202bd400cce1f377a0e53d
- worker 绑定：f6ee26379d749f2a6bbf221a2fbff1482f2f8ea28b7b60fe351daf2fb83d7be9
- 最终审计：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-context-final-audit-r654.json

下一步完善编译前资源分析及通用算子组合。已有 MLP 程序包可直接重载：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/dsl_bundle.py replay \
  --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-context-numerical-r651/tasks/8c3c4b5859a241e7a0fd5212d5aa302a/program --execute
~~~
