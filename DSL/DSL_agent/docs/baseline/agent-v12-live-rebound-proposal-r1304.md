# 有序 PyTorch 输入：三例真实 Agent 测试提案

待批准。原 r1232 保留，新提案绑定当前已接入的 706 文件版本。

## 案例

| ID | 模型 | 输入 | 命名输出 |
|---|---|---|---|
| linear | 有序张量写入与 Linear | [1,3] | current、previous、part |
| norm | 有序张量写入与 LayerNorm 近似 | [2,2] | current、previous |
| split_weight | 公开权重拆分与 Linear | [1,2] | current、previous |

仍使用原 v12 清单、公开权重、近似配置、布局与生成请求。
三例完整请求逐字段比较一致；源码、来源参考闭包、应用与后端身份重新绑定。
此测试验证 v12 新入口的自由生成和自动修复；v13/v14 的新增生成能力另列。

三例已完成当前来源捕获与独立明文准备：48 组／272 值，530.95 秒。
linear 复用已完成的第二次准备产物；另两例来自第三次准备。
首次启动锁冲突和第二次审核字段错误保留，没有修改捕获产物。
这些是来源准备结果，尚未调用模型 API 或进行本提案的编译／密态验收。

## 调用与执行预算

| 项目 | 上限 |
|---|---|
| Provider | DeepSeek deepseek-flash / high / stream |
| 后端与规则 | Poseidon CPU w45 / explicit-v11 / capacity-public-tensors-v9 |
| 每例 | 初次生成＋最多 3 轮修复 |
| 传输重试 | 每次生成最多 3 次 |
| 全批 | 最多 12 次生成、48 次 HTTP |
| 单次 API | 1200 秒、384000 tokens |
| 时间 | 全批 12 小时、每例 4 小时 |
| 并发 | API 1、密态 1、编译 2、链接 1 |
| 空间 | 每例 4 GiB、全局结果 32 GiB、磁盘余量 4 GiB |

外发公开规范化图、固定权重或描述符、布局、DSL 规则和受控反馈。
私有测试数组、reference、原规则 DSL 答案及凭据保留在可信侧。

没有明确货币上限；超时及传输重试可能重复计费。额度不自动扩大。

## 验收

每例分别记录首次成功、修复后成功或明确失败，并要求：

1. 真实 Agent 生成源码及来源收据。
2. Hecate／Dacapo-modified 编译与 RuntimePlan／bundle 检查。
3. Poseidon CPU 加密执行、解密和四组输入逐项数值比较。
4. 程序包重载、三组新输入执行及缓存复用。

冻结门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
人工或规则候选结果不计入这批真实 Agent 通过数。

60 项执行控制测试通过，覆盖计划和预算绑定、原截止时间、锁、缺失收据、
调用账本及正常／异常退出的后台清理。测试使用标准库合成子进程和替身；
不代表真实 Agent 或密态执行。最终只读 verify/plan 通过，未批准执行被入口拒绝，
实际执行目录尚未创建。

## 绑定与入口

提案 binding：
`fadf85332f2ee938ff16c61adda89461edefd27403c6ad928af5588fd6de84b0`

提案 SHA-256：
`d81042c67d0cfe427c1f8f813297702bee94de5f46236329b3d5f534d711bcc3`

[机器可读提案](agent-v12-live-rebound-proposal-r1304.json) ·
[联合入口验收](agent-common-fused-cutover-acceptance-r1303.md)

只读计划命令：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/source_live_rebind_v2/run.py --plan
~~~

批准后由独立批准收据启动该固定范围；本提案不授权安装、下载、后端或参数修改、
提交、推送或 PR。
