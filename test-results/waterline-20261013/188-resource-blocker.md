# 当前188资源阻塞：完整模型未验收

截至2026-10-14 188时间19:09，完整Qwen3-0.6B前向和独立数值验收仍未完成。没有完整261输出的成功报告，不把实现/回归通过或加载成功作为验收。

## 代码与已通过检查

生产源提交为Poseidon `c17effa2`（Rescale临时Q按已验证target+最大gap取上界），前序 `00a65df7`（实际plan降层数）、`ee7d5c68`（物理池预留及Ready实际回收）、`3d4cff66`（实际RNS整数预乘对齐后Rescale）；均已推送。ckks-runtime `369bdc55a8e585442784d5a6001e6e9dd04ce6f6`、DaCapo `6e75be29a52b69bc56404fc07e44465fe8971d29`引用未变，新增修正仅在父仓库。原lazy位置、160→40目标、level/HEFT和完整模型计划未改。

[最终188回归与绑定](188-level-aware-checks-summary.json)通过：CPU API9、GPU API24、scalar16、真实GPU Prepare分配失败/自动drain/同runtime显式再跑，单卡/全部四卡/HostStaged三后端压力及q_proj三后端。单卡为物理1，多卡始终全四卡。本地 [24项及真实三分量分配证明](level-aware-rescale-api-validation.json)通过；[清理清单](cleanup.tsv)、[覆盖审计](waterline-coverage-audit.md)、[完整HEFT前后比较](188-full-qwen-heft-audit-canonical-summary.json)保留。局部通过不等于完整模型通过。

## 最新完整尝试的实际失败

188的最新源API SHA256为`af90f47096fd338a95d17d48850f549eb271e7725e2c2e1e82e86475b259d2bd`，runner为`c261b9a5846b53fdab32f189e3b86a13923fc073fbced9630f22e42fb4422e73`。[启动](188-level-aware-full-launch.json)、[实际进程绑定](188-level-aware-full-live-binding.json)对应PID188271、全部物理卡顺序`1,2,3,0`，复用原1,185,203,847B计划及已验证SHA256 `9e906c0d32983dccbea866a7d506e677d039b57d537caa4d9bd0c568f61de813`。

这次在整体元数据预检退出1，[原报告](188-level-aware-preflight-model-report.json)、[日志](188-level-aware-preflight-model.log)、[诊断](188-level-aware-preflight-instruction-diagnostic.json)、[限定解释](188-level-aware-preflight-rejection-interpretation.json)。trace仅54B表头，没有提交图中的初始化传输/Prepare/Compute；之前的CUDA建池、key上传属于setup。

失败为`#16527122 MulCP`，logical GPU1通过UUID确认映射physical GPU2，`reuse=null`且该算子workspace=0。完整组`front=6,064,701,440B + plain=3,538,944B`，高于实际已reserved池扣固定资源后的`6,067,443,616B`额度，共796,768B。此次不是Rescale工作区上界回归，不能再从MulCP删除不存在的workspace。这里证明当前保证式准入在现有资源下拒绝；不是所有调度/复用实现的不可优化物理下界。

physical2当时的pool已实际reserved=max=8,399,093,760B，邻居作业挤占了可用空间。[随后设备与账号快照](188-resource-blocker-current.json)显示physical2由hanshubin/UID1020/PID185614占25,270MiB，physical0由songxingnuo/UID1016/PID148819占22,734MiB。xuming/UID1012无GPU残留；他人进程未终止，ECC当次无uncorrected volatile错误。早先物理0两次预检过计的修正及证据分别保留，不混用不同显存状态的结论。

编译器报告的front/object峰仍是保守额度诊断：含驻留明文和常量workspace，未按alias减物理峰；不能将其减去1GiB后称为真实CT常驻下界。当前边界以本次精确完整组及实际reserved cap为准。

## 资源恢复后的复现

需要能持续完成一次运行的全部四卡窗口；建议每卡至少空闲12GiB，作为实际协调目标，并非已验证的最小配置。不得终止他人进程、绕过预检或改成部分卡；也不通过放宽数值容差、缩层/词表或换合成权重来验收。已向用户请求协调；没有配置任意轮询重跑或隐式OOM重试。

保持独立SSH长连接，在188的tmux中执行；label必须尚未存在：

```bash
base=/home/xuming/poseidon-waterline-20261013
env -u POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB -u POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB \
  POSEIDON_QWEN_CUDA_VISIBLE_DEVICES=1,2,3,0 \
  POSEIDON_QWEN_BACKEND=gpu \
  POSEIDON_ACCEPTANCE_LABEL=qwen3-bsgs-q50-after-resource-release-20261014 \
  POSEIDON_VERIFIED_PRESSURE_DIR="$base/logs/level-aware-rescale-final-checks-20261014" \
  bash "$base/level-aware-rescale-final-update-20261014/run-188-acceptance.sh"
```

入口会核对构建状态完全一致、全四卡及原计划，成功前向后再调用独立检查器。目标是完整28层预训练Qwen3-0.6B、151936词表、固定tokens[9707,11]、261输出，全部有效real/imag及零尾部按既定atol=.001/rtol=.0001核对；尚未取得该结果。CPU/Auto只有小图覆盖，完整模型均未验收。完整模型实际使用13275次secret-assisted Host refresh，native Bootstrap=0、sec_level_type::none，不能称原生全FHE Boot或128-bit安全，固定两token校准也不是任意输入精度保证。

恢复后还须保存成功模型报告、decoded/checker hash、完整PID采样、Host/pinned/allocator与driver各自口径、收尾资源及实际限制，再更新验收状态。当前 [旧失败前向资源图](188-q50-lazy40-failed-resource-history.png)来自PID153204已关闭的756次采样及 [输入/峰值绑定](188-q50-lazy40-failed-resource-history.json)，明确是旧2:29:19失败历史，不是新代码成功模型测量或连续显存峰。
