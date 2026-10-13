# Waterline 实现与 188Server 验证

状态核对于 2026-10-13 22:45（北京时间）。已推送版本为 Poseidon `4fd76b9f`（数值修复 `acd6a27d`）、ckks-runtime `081e07c`、DaCapo `d5ed984`。**完整 Qwen3-0.6B 前向及独立数值验收尚未完成。** 旧 Q40 BSGS Auto 进程 PID30187 已按用户要求于服务器记录时间 21:16:36 发送 SIGTERM，退出码143；没有完整 decoded 或数值报告，见 [停止记录](qwen3-bsgs-q40-user-stopped.json)。本次更新时，188Server 新二进制的CPU API 9项、GPU API 20项、scalar、三配置压力及Q50 q_proj的CPU/GPU/Auto三后端检查均已通过，检查监督进程exit0。Q50全部四卡编译已启动（PID57547），尚无编译完成结果；Q50单卡容量检查和完整前向随后进行。

Waterline 已接通后期 Prepare 改写、完整消费者组准入、每设备水位、共享 Host/pinned/raw 预算及事件回收，复用同一执行器的前台计算和通信函数。自动静态分批、批末 Fence 及相关独立执行分支已替换；显式 Encode/Transfer/Fence、drain、通信依赖、二进制计划、按需读取、CPU worker、Release/reuse 保留。具体引用与处理见 [清理清单](cleanup.tsv)。HEFT 源码未修改；[源码 hash 记录](heft-source-sha256.json)只证明所列源文件相同，不代表新旧完整图的 placement/order 指纹相同。

## 本轮修复与已通过的本地检查

旧 Rescale 按实际 Q prime 除模后，仅将 scale 元数据改为目标 `2^scale`，没有校正密文系数。独立 q_proj 测得约1.090078倍幅值，旧完整模型运行不能作为数值验收。CPU/GPU 现共同按实际 prime 选择原生 Rescale 次数，用正 uint64 整数乘系数对齐 scale，再用 ModSwitch 消耗剩余计划层。原生 scale 至少保留 `2^40`，整数对齐相对舍入误差不超过 `1e-9`，目标须低于实际剩余模数空间；不满足时明确拒绝。这些检查不构成通用噪声、幅值或安全参数证明。GPU scalar 全链路保留 uint64，并修复原地 scalar 覆盖输入 storage 的错误。实现、失败负控及限制见 [数值修复说明](rescale-normalization-validation.md)、[scalar 验证](gpu-wide-scalar-validation.md)。

| 本地检查 | 实际结果与证据 |
| --- | --- |
| CPU API | 9/9；N16384/Q50 的 dense Rescale 全8192复数槽最大误差约1.3e-11；另保留50-bit CPU prime 的单层近1对齐和低精度拒绝：[日志](cpu-rescale-normalization-tests.log) |
| GPU API | 20/20；普通及多层 Rescale、MulCC/Relinearize/Rotate、完整槽 dense 参考：[日志](gpu-normalized-rescale-api.log) |
| 正式 GPU scalar CTest | 16组原地/独立输出、974848个RNS word 与独立128-bit模乘完全相同，2048复数槽、非默认stream检查通过：[日志](gpu-wide-scalar-ctest.log) |
| 统一执行器 | 8/8，包含 Auto lane 准入及资源边界：[日志](core-auto-idle-choice-tests.log) |
| 编译器 | 10/10，新增物理模数及剩余空间检查：[日志](compiler-integer-rescale-tests.log) |
| 隔离 Q50 q_proj | 合成权重的 cyclic 线性测试图，1025 Prepare、1025 Rescale、无 Boot；全部8192复数槽 maxabs=9.909881804e-10，虚部及尾部通过，forward=2.932s：[报告](q50-q-proj-corrected-report.json)、[日志](q50-q-proj-corrected.log) |

隔离 q_proj 是独立回归，不是完整预训练模型，也没有替换正式 BSGS DSL。普通 GPU API 和 scalar 的本地结果不能替代 V100、全部 GPU 或完整模型验收。

最新生产参数为 N16384、50个30-bit Q、8个30-bit P，上下文默认 scale40，计划输入、高 scale 和 Host refresh 输出 scale120。runner 按计划描述编码输入。逻辑 level0..9 对应物理 level4..49，每个逻辑 level 对应5个物理 level；120-bit scale 下降额外预留30-bit模数空间。每算子工作空间默认上界1024MiB，以覆盖Q50 Rotate的保守上界。物理 latency 行40..49沿用旧表最后行估算，未作为新硬件测量；逻辑 latency 表及 HEFT 算法不变。参数及编译入口见 [Qwen DSL 使用说明](../../scripts/qwen3_dsl/README.md)。

## 慢因与 pinned 缓存对照

已定位两项实际开销：旧 Auto 在 CPU worker 忙时仍给新组排入 CPU 准备，形成串行供应积压；GPU Prepare 每份原始槽数据都分配、释放 pinned Host 内存，增加 Host 提交及回收成本。最新 Auto 仅在 CPU lane 可用时给新组选择 CPU；CPU 和目标前台都忙时暂停新组准入，已准入组继续推进。GPU Waterline 每卡复用一份 N×4 字节 pinned 原始缓冲，计入整段执行的固定 Host/pinned 预算，真实 Ready 完成前禁止覆写，drain 后释放。普通未启用 Waterline 的 API 保持逐次分配语义。

[旧模型局部 trace](188-bsgs-auto-pre-supersede-trace-stats.json)在固定快照中记录 CPU Prepare 45688次、均值43.126ms，GPU Prepare 83090次、均值14.023ms，Host Boot 1301次、均值243.701ms。duration是API提交或同步Host工作，多worker总和不可相加为墙钟占比；trace字节数、提交数均不表示CUDA完成比例或剩余时间，GPU利用率快照也不足以证明卡死。

188 物理GPU1上的同二进制组件对照在旧模型停止后进行，两种模式均N16384/Q40/P8/scale120、10次GPU Prepare、全8192槽检查通过：

| 指标 | 逐次 pinned 分配 | Waterline 缓存 |
| --- | ---: | ---: |
| pinned 分配次数 | 10 | 1 |
| API提交均值 | 1.025306ms | 0.219797ms |
| 完成回收均值 | 0.417778ms | 0.012027ms |
| GPU Encode event均值 | 0.272096ms | 0.270313ms |
| drain后 pinned live | 0 | 0 |

这是组件开销改善，不是完整模型加速率。缓存首份分配计入均值，固定缓存65536字节；CPU Prepare仍有逐份分配成本。完整命令、scope及source/binary/library绑定见 [对照说明](hotpath-pinned-cache-comparison.md)、[状态](hotpath-cache-188-current-status.json)、[来源hash](hotpath-cache-188-current-provenance.txt)。[Nsight说明](hotpath-components-profile-validation.md)只解释诊断组件的CUDA API成本，不将其比例外推到完整模型墙钟。

## 188Server 小图覆盖与完整模型边界

当前数值修复版本在188实际通过CPU API 9/9、GPU API 20/20、scalar 16组，以及物理GPU1、全部物理0/1/2/3默认通信、全部卡强制HostStaged三配置压力。每配置均执行CPU/GPU/Auto，N65536/Q32×30-bit/scale40，比较全部32768复数槽。监督进程[退出码](188-rescale-exit-status)为0，实际二进制/源码绑定见 [hash](188-rescale-build.status)；CPU/GPU/scalar日志分别为 [CPU](188-rescale-cpu-api.log)、[GPU](188-rescale-gpu-api-single.log)、[scalar](188-rescale-gpu-scalar-single.log)。

| 配置 | 三模式最大误差 | 进程墙钟 / RSS峰值 | 日志 |
| --- | ---: | --- | --- |
| 物理GPU1 | 9.93667e-7 | 34.41s / 3606328KiB | [single](188-rescale-waterline-single.log) |
| 全部0/1/2/3 | 1.11503e-6 | 52.95s / 4001976KiB | [all](188-rescale-waterline-all.log) |
| 全部0/1/2/3，HostStaged | 9.51937e-7 | 54.06s / 3985140KiB | [HostStaged](188-rescale-waterline-host-staged.log) |

三份压力日志均exit0，结束动态plain/front及pinned live归零。全部卡测试的三张peer卡均在延迟GPU0之前实际完成；HostStaged每模式post20次，20次均观察到pinned。RMM活跃分配、保留池与固定资源分别报告，保留持久资源时不能要求RMM live归零。[检查结束设备快照](188-rescale-devices-after.csv)全部GPU占用0MiB、volatile uncorrected ECC为0；这是收尾快照，不是整个运行的连续峰值或无其他作业的历史保证。

188物理GPU1也以正式runner执行了上述隔离Q50/P8合成q_proj计划，全部8192复数槽、虚部和尾部均通过：

| 完整Prepare后端 | max absolute error | forward | 报告与日志 |
| --- | ---: | ---: | --- |
| CPU | 9.907332732e-10 | 20.328s | [报告](188-rescale-q-proj-cpu-report.json)、[日志](188-rescale-q-proj-cpu.log) |
| GPU | 9.909530974e-10 | 4.572s | [报告](188-rescale-q-proj-gpu-report.json)、[日志](188-rescale-q-proj-gpu.log) |
| Auto | 9.907328291e-10 | 8.053s | [报告](188-rescale-q-proj-auto-report.json)、[日志](188-rescale-q-proj-auto.log) |

三种后端均结束pinned live=0。这是同一小图的真实CKKS数值和后端覆盖，不推导完整模型端到端加速。候选小图上下文默认scale120，与生产完整图的显式输入scale120一致；它不是完整模型的Q50容量或所有算子链证明。

较早pinned缓存版三配置日志 `188-pinned-reuse-*` 及其 [source/binary hash](188-pinned-reuse-source-and-binary.status)保留，maxabs依次9.99418e-7 / 1.02392e-6 / 1.07219e-6；当时与旧模型有运行重叠，其墙钟不可作为独占性能对照。更早压力、资源反例、stream正负对照分别保留在 `188-final-waterline-*`、[CPU Host边界说明](cpu-prepare-host-budget-validation.md)、[stream Ready说明](stream-ready-validation.md)，不因旧实验产生失败便删除证据。

官方目标是 Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`。权重SHA256为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；311个BF16张量、751632384个存储参数已核验，含重复tied LM head，去重596049920。模型完整28层，hidden1024、FFN3072、16个Q heads、8个KV heads、head dim128、vocab151936，Q/K norm在RoPE前、无attention bias、共享embedding/LM head。历史24层合成模型不能替代此目标。客户端对tokens `[9707,11]`、positions `[0,1]`查embedding后加密；不是密文token查表。

完整验收要求2输入、261密文输出：最后位置的151936个logits分149份（最后384个有效值），加全部28层、两个位置的112份K/V（114688个有效值）。独立NumPy dense FP64 exact/approximate参考的logit maxabs=3.787683944267428e-9、RMS=7.210505850742908e-10、top1均358。runner与 [独立检查器](../../scripts/qwen3_dsl/validate_decoded.py)核对全部有效值、所有虚部和零尾部，并核验signature/fixture/参考/source绑定，固定 `atol=0.001, rtol=0.0001`。`checker-test.log`的20项是明确标注的模拟decoded，只验证检查器拒绝与映射。

| 完整模型阶段 | 截至本次更新的实际状态 |
| --- | --- |
| 旧Q40 BSGS四卡编译/导出 | exit0，29:55.87，RSS137225352KiB；binary1182542616B、bundle61298262016B，19364300条指令；[编译日志](qwen3-bsgs-final-all-compile.log) |
| 旧Q40 BSGS单卡容量 | 当前预算基线拒绝，非GPU执行；[摘要](188-final-single-capacity-summary.json)、[日志](qwen3-bsgs-final-single-capacity.log)；不能据此断言所有单卡配置不可行 |
| 旧Q40完整前向 | GPU-only PID22555已superseded；BSGS Auto PID30187已按用户要求停止，exit143；两次均未完成数值验收 |
| 新Q50全部四卡编译 | 正在进行，未记录成功、加载或完整执行 |
| 新Q50单卡容量 | 尚待当前参数及计划检查，不能沿用Q40结论 |
| 新Q50完整前向及数值验证 | 尚未启动；全部261输出、独立参考通过及模型端到端资源/耗时仍待交付 |

所有完整模型CPU-only尚未启动；GPU-only旧图被superseded；Auto旧Q40图被停止。小图三模式和模型编译成功不补足完整模型后端覆盖。旧二进制scanner漏Prepare tag5、旧spec schema失败均已修复并保留原日志；[旧GPU-only停止记录](qwen3-full-gpu-n16384-v3-superseded.json)中的time-v同时显示signal15和exit0，不能算模型通过。

## 复现与剩余交付

大型权重、trace、plan、bundle、decoded及profiler文件保存在 `/home/xuming/poseidon-waterline-20261013`，不提交Git。远程编译、测试及取回结果全程保持一条独立长SSH；tmux只负责任务持久化，不能替代长连接。188单卡使用物理GPU1；多卡必须全部0/1/2/3，并记录UUID和ECC。

公开 [build-188.sh](build-188.sh)使用188既有离线依赖、CUDA arch70，构建CPU API、GPU API、GPU scalar tests及Qwen runner四个target，保存精确构建日志及hash。新数值版本已用于上述188实际回归；其本轮构建记录在服务器 `logs/rescale-normalization-build/`。旧 `188-build-reproduce.log` 的成功仅是旧版本增量复现（0.61s），不能作为当前数值修复的新构建通过证据。

```sh
base=/home/xuming/poseidon-waterline-20261013
POSEIDON_188_WORKSPACE="$base" bash build-188.sh
# 此小图检查须先备好 "$base/q50-small-regression.tar"，并使用新检查日志目录。
POSEIDON_188_WORKSPACE="$base" bash check-rescale-188.sh
cd "$base/qwen3_dsl-bsgs"
# 输出目录必须是新目录；已有Q50产物时为复现换一个新的output-name。
/opt/conda/bin/python compile_plans.py --workspace "$base" \
  --source "$base/qwen3-traced-bsgs/trace_qwen3.mlirbc" \
  --output-name qwen3-plans-bsgs-q50 --devices 4
/opt/conda/envs/apollo/bin/python export_runner_fixture.py \
  --signature "$base/qwen3-traced-bsgs/signature.json" \
  --fixture "$base/qwen3-fixture" --spec "$base/profiles/operator-spec.json" \
  --p-moduli-bits 30 30 30 30 30 30 30 30 --output "$base/qwen3-runner-bsgs-q50.json"
cd "$base"
POSEIDON_188_WORKSPACE="$base" \
POSEIDON_ACCEPTANCE_LABEL=qwen3-bsgs-q50-auto-final bash run-188-acceptance.sh
```

先按 [DSL说明](../../scripts/qwen3_dsl/README.md)部署生产脚本/profile及匹配新编译器，复用相同完整BSGS trace和权重绑定。编译脚本只允许1/4卡、默认顺序编译，任一子进程失败最终非零。上面的独立4卡命令与后续单卡容量检查避免重复同时创建产物。[check-rescale-188.sh](check-rescale-188.sh)用于本轮服务器CPU/GPU/scalar、三配置压力及Q50 q_proj三后端检查；其小图tar必须先按脚本路径准备，脚本以 `latest-build.status` 校验当前实际二进制hash；需要等待异步构建时可指定 `POSEIDON_BUILD_EXIT_STATUS`。不能仅凭启动检查脚本就宣称通过。[run-188-acceptance.sh](run-188-acceptance.sh)检查Q50编译exit0，再执行三配置压力、全部卡Auto完整模型和独立数值检查；label目录禁止覆盖旧日志。

`monitor_process.py`在CUDA setup前按本进程PID与身份采样RSS/HWM及driver GPU内存；采样峰值并非连续峰值。runner分别记录冷启动、setup/init/forward、首个及全部输出、Host Boot、CPU/GPU Prepare、GPU events、H2D、等待及收尾。外部time-v覆盖进程冷启动。最终仍需当前Q50编译/容量结果、真实完整261输出及独立数值报告、实际后端覆盖、每卡/Host/pinned峰值和收尾清单；这些未完成项不能报告为完成。

本参数使用 `sec_level_type::none`，不声称标准安全级别。Host decrypt/re-encrypt refresh使用秘密密钥，不是原生FHE Boot；固定两token非线性校准不构成其他输入的通用精度保证。Native Boot的Waterline workspace查询尚未接通，普通显式Native Boot能力保留。当前尚无完整模型端到端加速证据。
