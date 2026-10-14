# 应用源码入口评测

## 范围

复用 Application 和请求级 WorkerContext，验收 v12 来源捕获、explicit-v11 DSL 生成、自动修复及程序包复用。首批为 Linear 视图写入与快照、LayerNorm 近似原地更新、公开权重 split→Linear。

提案位于 `docs/baseline/agent-v12-live-proposal-r1232.json`。生成与 HTTP 上限分别为 12 和 48；每例 4 小时、4 GiB，整批 12 小时。API 和密态执行串行，编译最多 2 jobs、链接 1 job。

## 计划

~~~sh
python3 -B scripts/baseline/benchmarks/tools/source_live_acceptance_v1/run.py --plan
~~~

计划模式检查源码、模型、请求和历史参考绑定，返回案例与预算。

## 执行

用户批准后，由可信控制端保存绑定提案的授权记录，再启动新结果目录：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/source_live_acceptance_v1/run.py \
  --execute /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v12-live-r1236 \
  --approval-record docs/baseline/agent-v12-live-approval-r1232.json
~~~

每例通过原 Application 捕获模型、生成与验证。成功后，检查程序包重载、三组新输入密态执行及缓存复用。调用账本、原始响应、编译与数值证据保存在该结果目录。

每次执行使用新目录。未确认的调用结果进入停止状态；历史 76 项和旧预算独立保留。

## 离线检查

agent-v12-live-controls-r1235：21 passed、0 failed、0 errors、0 skipped。涵盖授权绑定、案例与调用上限、未知调用、计划模式及子进程清理。

该组检查使用 Ubuntu 系统 Python 3.10.12，API、编译和密态执行均为 0。实际后端执行使用既有 Nix Python 3.10.14 环境。
