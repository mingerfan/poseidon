# gather 组合验收进度

| 模型 | 输入 | 输出 | 当前离线结果 |
|---|---|---|---|
| gather → Linear ＋残差 | [5] | [4] | 通过 |
| Linear → gather ＋双输入融合 | 两个 [2,3] | [2,3] | 通过 |
| transpose → gather →命名输出与 mean | [2,3,2] | [3,1,2]、[3,1] | 通过 |

三份原始 PyTorch v37 模型各通过 16 组独立参考检查，共 48 组。第一、二例使用固定的非重复随机线性权重。

规则程序完成 3 次编译、9 次 Poseidon CPU 密态调用，以及三份包的重载、新输入、重复执行和缓存复用。24 组输入、152 个输出值；最大绝对误差 **1.85285398135e-7**。只读独立审计通过。

## 当前真实 Agent 批次

r1585 已启动，按计划内总测试授权执行。DeepSeek flash/high/stream，explicit-v19，固定 w45；API 与密态串行。最多 12 次生成、48 次 HTTP，单次 1200 秒／384000 tokens、每例 4 小时／全批 12 小时。产物上限每批 4 GiB／总计 48 GiB、磁盘余量至少 4 GiB，编译 2／链接 1 及 native 限制保持。

该批次尚未结算，当前不能计为真实 Agent 通过。结束后由后置只读审计核对响应来源、调用账本、真实产物、逐项误差与包复用；两者终态前共享源码保持冻结。

SDK `605f3b5e330454bfe8ff6b146d54c878e570d1500264f1f9ebd36dd189227a1b`。主 PID970607/start27877807，后置 PID970608/start27877807；进程身份仍须实际核对。

[基础 gather 已完成报告](agent-tensor-gather-acceptance-r1582.md)；[当前快照与绑定](agent-gather-composition-progress-r1586.json)。模型、reference、门限、后端库和网络保持原样；安装、下载、提交和推送为 0。
