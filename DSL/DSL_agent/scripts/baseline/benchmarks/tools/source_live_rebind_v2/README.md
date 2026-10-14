# 当前 706 下的三例真实 Agent 待批入口

这是独立开发验收工具，不接入应用正常路径。保留原 r1232 提案、旧请求及 r1229 规则证据。

三例仍为 linear / norm / split_weight：V12 有序源码输入、explicit-v11 生成与修复，
以及真实 Poseidon CPU 数值、包重载、新输入和缓存复用。它不验证新增 common/fused 的 Agent 生成。

## 已完成的免费准备

复用当前 Application.submit_source → prepare_source，在既有 pure Nix Python/SDK 中
受控捕获来源、独立 float64 参考和完整请求。没有 Provider、编译、密态执行、安装或下载。
原窗口 2026-10-08 07:43:41.480410 至 07:53:41.480410 UTC；
全部来源准备在 07:52:32.430385 UTC 完成，累计 530.949975 秒。

- linear：16 组 / 80 值，复用 attempt2 已完成的真实捕获。
- norm：16 组 / 128 值，attempt3 新捕获。
- split_weight：16 组 / 64 值，attempt3 新捕获。

合计 48 组 / 272 值。完整请求均与原三例相同；当前来源 706、保护 654、
验证闭包 611 文件另行冻结。固定近似的原函数应用精度未认证。

初次 launcher 锁冲突和第二次独立审核器的路径/schema 错误保留在各自目录；
这些不是模型数值失败。原 deadline 未延长，没有重复 linear 捕获。

生产模块、后端及参数保持。原始捕获产物、报告、请求、producer 四文件版本和 SHA
均保存在 work results 的三个独立 attempt 目录。当前 live 工具 hash 与 producer hash 分开，
INITIAL/report/两次 capture freeze 均有固定 SHA，不能靠一起重绑改写原用时。
private arrays 留在本地原产物，不进入提案或 Provider。

## 只读核验

在 Ubuntu checkout：
```sh
python3 -B scripts/baseline/benchmarks/tools/source_live_rebind_v2/run.py --plan
```

当前待批文件为 proposal-ready.json。
Binding: fadf85332f2ee938ff16c61adda89461edefd27403c6ad928af5588fd6de84b0
SHA-256: d81042c67d0cfe427c1f8f813297702bee94de5f46236329b3d5f534d711bcc3

60 项标准库控制日志：controls-final.log。控制包括拒绝扩大三例、预算、源码/请求/
参考/工具身份、缺批准和旧批准，以及实际跨进程锁、supervisor、后台后代清理。
没有运行付费入口；无批准记录，无 live 运行目录。

## 付费执行边界

仍沿用原 12 次生成 / 48 次 HTTP、每例 4 小时 / 4 GiB、整批 12 小时、
全局 32 GiB / 余量 4 GiB，API 和密态串行、编译 2 / 链接 1。
currency_limit 为 null，费用未知；超时重试可能重复计费，原披露保持。
批准须绑定新 proposal 的 SHA/binding 和全部 7 个工具 hash；旧 r1232/r1296/r1301 批准不适用。
未批准执行在 Nix、结果目录与凭据加载前拒绝。当前 fixed RUN 为
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v12-source-live-rebound-v2，
不允许继续旧批次、改 RUN 或从隐藏 --inside/--case 独立启动。
