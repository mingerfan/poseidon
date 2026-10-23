# Agent 续跑与绑定反馈修复

截至 2026-10-07T14:01:36.717695+08:00，原76项真实Agent评测为 42通过、6失败、1运行、27未启动。剩余原定范围继续执行，生产源码仍按旧绑定冻结。

## 已确认

- 已关闭15项组：10通过、5数值失败。矩阵组已审计部分：28通过、1服务失败。两组38份程序包均有真实生成来源、Poseidon CPU密态验收、重载及3组新输入。
- 对这38份包重新核验了266组输入、1694个输出值；最大绝对误差 1.020776819e-06。冻结门限为 1e-5 + 1e-4*abs(reference)。本轮审计复用已有密态结果。
- 5项数值失败为LayerNorm三项、GroupNorm一项、variance一项。原15轮密态结果与候选源码计算一致，候选计算与目标模型不同。
- matrix__public_vector_r_1前两轮遗漏或错置第二个批次的输出；候选数学误差最大0.03125，密态结果相对候选计算的误差最大1.144e-10。第三轮Provider报transport_invalid_stream，未收到可验收候选。原记录保留。
- 695份生产源码、3份原执行工具和698项保留证据哈希匹配。Poseidon、Runtime、Dacapo、reference及参数保持。

## 本轮修复草案

| 改动 | 状态 |
| --- | --- |
| 容量ABI下的归约指导，以及非标量广播指导 | explicit-v11草案 |
| 公开标量、向量、单槽掩码与逻辑常量名区分 | explicit-v11草案 |
| 未绑定变量、错误数组dtype的准确源码位置 | Agent检查器草案 |
| 忽略注释及空白的重复候选识别；允许修正响应request_id | 控制层草案 |
| Provider停止时返回Provider终态，保留此前数值诊断 | 控制层草案 |

113项组件／兼容测试、14项可移植回归均通过；续跑控制13项和安全接入守卫7项均通过，failed/skipped为0。测试使用ARM系统Python3.10.12和现有锁定NumPy1.25.2，组件后端与Provider采用测试替身。生产运行继续使用固定Nix Python3.10.14。这些测试没有新增编译或密态执行。

草案位于 scripts/baseline/benchmarks/tools/staged_binding_feedback_v1；12个拟接入文件尚未写入生产。原explicit-v1至v10请求配方保持，新v11仅显式选择。实际接入守卫已在运行批次上返回 waiting_for_current_batch，applied=false。

## 续跑范围与预算

原批次因流式响应错误关闭后，只选取31项未启动矩阵／InstanceNorm及1项Attention继续。已通过和失败项均保留；停止的Provider案例未自动重试。32项最大128次生成／512次HTTP，累计预留截至快照为 78生成／78 HTTP；原单例4／16限额与原组截止时间保持。

DeepSeek flash/high/stream，1200秒／384000 tokens；API与密态串行，编译2／链接1；单例4GiB、总32GiB、磁盘余量4GiB。当前总产物约24.57GiB，空闲约43.93GiB。实际计费金额未知；没有新增预算、安装或下载。

当前续跑目录：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-unstarted-continuation-r1113

当前续跑绑定：b7a48ad854529ae215ae3133c1c0e53cacffd89a840c63aa06c4b2e74fb5ae47

## 判断与待验证事项

基于已有输入上的差分，5项数值失败及矩阵失败候选应优先修复Agent的常量角色、归约广播和输出坐标表达。提示改动的真实Agent收益仍待新绑定下验证。

身份预览确认后端库哈希保持，但当前保守模块闭包仍会使生成、编译、执行、验证身份全部变化。接入后应建立新绑定并验证代表包；旧包按历史身份保留，缓存失效与后端错误分别处理。生成指导与执行身份尚未完全解耦。

v11实际沙箱、编译、密态及真实Agent尚未验收；Bootstrap、CUDA、规模扩展及原模型应用精度保持独立状态。本轮仍使用现有Poseidon CPU w45。

## 下一步

继续当前冻结32项，完成后的独立审计只检查新增结果。批次关闭且Nix环境释放后，运行安全接入守卫，再做组件、旧接口及代表后端回归；失败案例的新v11付费重测另冻结具体请求与预算。

    cd /home/lhohy/Code/Poseidon
    python3 -B scripts/baseline/benchmarks/tools/staged_binding_feedback_v1/guarded_apply.py

上述命令只检查接入条件。--apply在当前批次仍运行时返回等待；生产接入不会重写旧报告、切分支、commit或push。

[机器可读快照](agent-live-continuation-and-fix-r1120.json)
