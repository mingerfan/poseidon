# MHA 两项重测入口验收

日期：2026-10-09。状态：入口准备完成，等待原两项批准问题的答复。

## 已完成

- 复用现有 Application、Provider、Controller 和 Poseidon 后端，接好两个原模型的自由生成入口。
- 独立批次新建预算边界，最多 8 次生成／32 次 HTTP；API 与密态串行，预算和沙箱限制保持。
- 授权、逐次源码校验、调用账本、PID 身份、所属进程清理、程序包验收及缓存复用检查均接入。
- 54 项控制测试通过，failed/errors/skipped 均为 0。
- 真实只读 --plan 通过，返回两个固定案例、approved=false、provider_calls=0。
- 654 个受保护文件及当前四阶段各 691 个源码哈希核验通过。

## 修正的元数据问题

首次只读 --plan 正确拒绝 r1316：辅助表中三个哈希仍指向修复前文件。
原方案四阶段身份早已记录了这些文件的当前哈希。r1317 只校正该重复表；
原 r1316、失败日志及修正前工具快照保留。

三个文件是 candidate_check_diagnostic.py、decorated_functions.py、
unified_graph_contract.py。没有再次修改这些源码。

模型、请求、四阶段身份、Provider、预算、外发内容和误差门限逐字段一致。
工具会再次核验这种等价关系，因此原两项问题的授权范围没有扩大。
旧三项批准和旧剩余额度不能用于启动本批。

## 本轮证据范围

本轮新增 API、编译、密态执行和数值验收均为 0。54 项是批次控制测试，
不是模型通过数。原 r1315 的成功 Agent 源码、程序包和密态数值结果保持原绑定。

正式授权记录和执行目录尚未创建，当前没有运行批次。
没有安装、下载、后端或参数修改，没有 commit 或 push。

下一条只读命令：

```sh
python3 -B scripts/baseline/benchmarks/tools/mha_feedback_live_v3/run.py --plan
```

修正方案：[r1317](agent-mha-feedback-live-proposal-r1317.md)。
机器记录：agent-mha-retest-readiness-r1318.json。
准备验收绑定：`2875207dfad79786b708ad221a834682f91ac0da4f6feae32b408af7d2bfa3d9`。
