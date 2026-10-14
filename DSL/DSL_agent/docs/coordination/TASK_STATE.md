# SSH 开发接续状态

2026-10-14。本文件为接续入口，计划见 PARALLEL_DEVELOPMENT.md、GENERALIZATION_GOAL.md、RESNET20_GOAL.md。

## 当前事实

- 开发位于 ARM Ubuntu /home/lhohy/Code/Poseidon，分支 lhy-agent-dsl。
- 旧本地 HEAD：7588e48093631eaf8385506e3db058e7ca3cfe7e。
- 采用上游 poseidon：b24d95aac87322e441e94414cb9283207d197f6b。
- 上游 Runtime gitlink/当前源码：f7e9115eaec22e44c8e12ff9f53528e421691688。
- Runtime 嵌套 Dacapo：39c1fb255adddbfb9d7180f9d1b8535a31241bff。
- 合并冲突已解决。用户于2026-10-14明确批准本地合并及 Agent 目录迁移；本文件随该双父提交保存，父提交为上述旧 HEAD 和上游 poseidon。未 push、PR，也未创建 DSL 或两个功能分支。
- Agent 代码在 DSL/DSL_agent/scripts；文档在 DSL/DSL_agent/docs。根 scripts 的旧入口及 docs/baseline 保留兼容转发。
- 运行批次为 0；旧 Goal 在 Mac 任务中处于暂停状态，不能假定自动复制到本 SSH 任务。
- 当前目录/旧布局、两种 Linux 架构及控制层相关检查共110项通过，failed/error/skipped均0。当前新后端尚未重建或真实密态验收，不能把旧SDK证据归于这些新提交。

## 本地保护

完整源码恢复归档及 Git bundle：
/home/lhohy/poseidon-work/backups/poseidon-integration-20261014-r1698
25,914 文件、31,697 归档条目均核验；归档约411MiB，最初3份Git bundle，后补Runtime嵌套Dacapo bundle。
保留 stash a5251d328e07a81980aa175f9a0bd22aaab9630f，不删除、不pop。
主快照绑定 ac3acb9d76539bca88a9178294c33fece9203baffcaaf9304c89eea5c8fed183。
14,820 份受保护本地文件逐项保持字节一致；路径适配单列。Bootstrap CMake 的11条本地新增行均保留。
3份与上游同名的旧Runtime API源另存于 DSL/DSL_agent/legacy_backend/src/poseidon/runtime_api，属于保留记录，不是新后端实现。
.env保持原位；未读取、复制或打印密钥。

## 最新验收与缺口

- 归一化源入口v3、GroupNorm/InstanceNorm新Agent6/6通过：3首轮/3修复，9生成9HTTP、9编译尝试6成功、18密态；报告 baseline/agent-normalization-source-live-results-r1694.md/json。
- 12份已有Agent包同输入重载：96组544值，真实Poseidon CPU CKKS12/12，maxabs4.453464664821993e-7。
- 原生PyTorch与独立数学参考12/12，maxabs6.661338147750939e-16；近似→分解12/12，maxabs2.220446049250313e-16。
- 原函数→声明近似和密态各12项诊断未达门限，最大差约1.58178；没有声明应用精度指标，不能认定原模型精度通过。报告 baseline/agent-source-precision-separation-r1697.md/json。
- 以上属于迁移目录前的固定SDK证据，SDK为6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429。原响应、source和结果均保留。
- Legacy Agent SDK使用third_party/ckks-runtime-agent及原独立构建；官方最新源码已下载，但是否兼容七个语义补丁、依赖、API、参数与产物，需要A0/B0审计和新绑定。尚未将其记为新后端通过。
- third_party/ckks-runtime-agent 和旧 third_party/dacapo 是原位保留的本地恢复/SDK资料，本次不将其添加为新的后端源码或 gitlink；当前受版本控制的后端仍为官方 Runtime 及其嵌套 Dacapo。
- 旧CPU bootstrap ratio128试验有真实多次刷新，但signed第二次刷新仍有误差失败；当前Agent bootstrap gate未认证开放。新的上游修复必须实际复核。
- 本机M5/ARM VM没有NVIDIA设备，CUDA实机未验收。
- 容量、算子、近似质量、三个整模型与ResNet20仍有剩余工作；通用化与整网验收计划并行，覆盖分母和源版本分别记录。

## 权限与下一行动

原计划内离线和真实Agent测试已统一批准，证据上限48GiB，当前约39.832GiB，磁盘余量约28GiB（执行前实时复核）。
不改网络。本次本地合并提交已获批准；新的安装、下载、数据/权重、参数、后端库、资源扩展，以及后续commit/push/PR仍保留分别批准。
常用生成flash/high/stream，最多3轮修复、3次传输重试，1200秒/384000tokens；单批精确调用和时间预算仍先冻结。
本地合并提交以Git实际记录为准：父提交须为7588e48093631eaf8385506e3db058e7ca3cfe7e和b24d95aac87322e441e94414cb9283207d197f6b。MIGRATION_REPORT.md/json保留提交前的历史快照及原绑定，不改写其中的pending状态。后续在独立工作目录获准并就绪后，以该共同提交分别启动两个Goal。
当前第一命令：
git status --short --branch
然后读取本文件和迁移报告，核对实际提交及工作区状态。不能开始两个共享checkout的并行改动或自动继续旧未知调用。

## 旧任务与记忆

原任务ID：01a0be17-dde5-71a2-8d6b-4ab494bbca3e，host local，标题“检查 Poseidon 迁移环境”，保持可查询。
新SSH任务使用项目“Poseidon on Ubuntu”，host remote-ssh-discovered:poseidon-local-arm64。
源码与文档是当前事实来源；本文件保存计划、权限、结果和下一行动，不声称恢复整个聊天或旧Goal。
历史接续：../baseline/mac-migration-handoff.md。历史源码/证据引用原路径，旧路径转发与恢复归档供追溯，不能改写旧绑定。
