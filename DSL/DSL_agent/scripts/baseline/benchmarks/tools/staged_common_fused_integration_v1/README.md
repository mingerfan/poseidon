# 有序张量与融合矩阵联合接入工具

将已验收的 common-call、有序张量和融合矩阵来源草案整理为一份接入包。
最终覆盖顺序与来源及生成请求验收时一致。

## 内容

| 文件 | 用途 |
|---|---|
| targets/ | 10 个最终 Agent 文件 |
| recovery-originals/ | 8 个现有文件的恢复副本 |
| plan.json | 原始与目标身份、覆盖顺序、验收依据 |
| check.py | 只读切换守卫 |
| test_check.py | 26 项完整性、权限及边界检查 |
| acceptance-r1281.json | 当前检查结果、权限及源码快照绑定 |
| acceptance.json | 前一轮 22 项检查记录 |

两个新增模块为 public constant provenance 和 fused matrix normalization。
来源契约覆盖 V12、V13 与 V14，原 JSON、CLI 和应用入口继续沿用
现有实现。该目录属于开发工具，应用请求不依赖它。

## 检查

在 ARM Ubuntu checkout 根目录执行：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/cutover.py --evidence-mode composite
python3 -B scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/test_check.py
```

守卫核对生产文件、最终目标、前置覆盖层、恢复副本、来源及请求验收、
后端/reference 身份、原文件与副本权限、当前分支和真实进程。结果包含阻塞原因和具体阶段。

联合版本已完成代表性 Poseidon CPU 数值验收并接入应用，当前状态为
applied_with_staged_joint_backend_evidence。切换日志、原文件和证据在
`cutover-records/20261008T072609-37b52747/`。
`check.py` 与原 plan 保留切换前快照；切换后的状态通过上面的 `cutover.py` 检查。

## 联合后端验收入口

`runtime_scope.py` 冻结 12 个来源、完整请求、独立参考和 706 文件身份。
`runtime_acceptance.py` 通过共享的 `runtime_driver.py` 运行资格验收、
程序包新输入执行和缓存复用。共 36 个阶段，使用既有隔离与资源限制。

具体范围与预算见
[离线验收提案](../../../../../docs/baseline/agent-common-fused-offline-proposal-r1296.md)。
原 r1296 保留失败结论；r1301 的 11 项补验全部通过，复用原 Boolean 后为
12 例／36 阶段通过。[验收与切换结果](../../../../../docs/baseline/agent-common-fused-cutover-acceptance-r1303.md)
记录独立绑定与候选来源。

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/runtime_acceptance.py --plan
```

`--execute` 要求对应的独立批准记录。上述旧批次已关闭，额度不复用；
旧运行脚本与证据保持，新的执行范围需独立准备。
