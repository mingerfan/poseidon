# 11 个原失败项的离线补验

补验已完成静态修复的原案例。使用当前 Poseidon CPU 链路；布尔掩码 Attention 已通过的三阶段证据单独复用。

| 范围 | 案例 |
|---|---|
| 常见调用 | common_alpha、common_dropout |
| addmm | addmm_cipher、addmm_public_right、addmm_public_bias |
| baddbmm | baddbmm_cipher、baddbmm_public_right、baddbmm_public_bias |
| addbmm | addbmm_cipher、addbmm_public_right、addbmm_public_bias |

每例执行资格数值验收、程序包新输入执行、缓存复用，共 33 阶段。
最多 11 次编译、11 个资格密态阶段及 11 个新输入密态阶段。

候选使用现有通用布局分派，11 份源码及原模型、公开权重、请求和 reference 已冻结。
通过静态预检不代表编译或数值通过。本批是规则基线验收，不计为新生成 Agent 结果。

| 预算 | 上限 |
|---|---:|
| 整批时间／每阶段 | 3600 秒／180 秒 |
| 新产物／全局产物 | 4 GiB／32 GiB |
| 最低磁盘余量 | 4 GiB |
| 密态并发／编译／链接 | 1／2／1 |
| API、安装、下载、参数及后端修改 | 0 |

沿用 Poseidon w45、N=32768、Q13/P1、tc128；数值门限为
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
保持独立沙箱和完整性检查；超限保留现场。

原 r1296 已结束，候选策略已改变，本批使用独立提案与执行绑定。
原失败、旧候选和 Boolean 通过记录保留；新运行中不重复执行 Boolean。
全部证据合格后，再核验联合源码切换所需的合并证据，当前应用源码仍保持。

- 提案绑定：`92a24d9abbd0e284961d5a2daac70e65adbb0e9125cb321957810a7b6dc60a4b`
- 提案 SHA-256：`1d821ac8207f1eda06a7b868663355836098e1f3c137d9203043628aa7b053b4`
- 执行目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-fused-capacity-runtime-r1301`
- 入口：`scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/candidate_strategy_v2/runtime_acceptance.py`

当前状态：提案已冻结，待本批离线执行批准。
