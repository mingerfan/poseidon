# 公开张量变换接入与验收

显式可信 PyTorch v31 支持公开参数的 repeat/tile、repeat_interleave、flip、roll，
保留既有密文张量变换、源码包、结构化输入和命名输出。

## 已确认

开发与执行：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e480`。
实际运行 owner 已退出、退出码 0，源码冻结释放。

327 项数学、接入和兼容控制＋12 项执行边界/账本控制通过；失败、跳过均为 0。
18 个隔离来源通过双明文参考：288 组、1,472 个值。
来源包含 15 个新上下文（15 个不同生成请求）及 3 个原 v30 兼容来源。
每种新运算三个上下文，覆盖不同轴、次数、dtype、Linear、平方、残差、密文重排、
多输入、命名多输出和本地 helper；来源变体与冻结的 1,200 个 Benchmark 模型分开计数。

| 轨道 | 通过 | 编译 | 密态调用 | 验证组 / 值 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| 规则候选 | 18 | 18 | 54 | 144 / 736 | 5.9567445629e-08 |
| 真实 Agent | 15 | 15 | 45 | 120 / 640 | 8.6491925760e-08 |

两轨失败、跳过、未运行均为 0。真实 Agent 全部首次通过，修复 0；
DeepSeek deepseek-flash/high/stream，实际 15 次生成 / 15 次 HTTP，未结算调用 0。
Provider total 258,965 tokens，prompt 156,287、
completion 102,678；reasoning 100,838 已包含在 completion 中。
未推算货币金额。

检查真实 tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、tc128、
参数与输入输出绑定、Poseidon CPU 密态计算、最终解密和逐项比较。
每包另以三组新输入和一组重复输入执行，检查重载、缓存复用及零生成/编译调用。
最终 DSL 与 Provider 原始响应逐字相同；规则程序单独计数。

冻结门限 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`；
独立明文参考 `atol=rtol=1e-12`。

## 实现与判断

公开变换在接入时求值，派生结果作为固定公开参数绑定。
原始状态、逻辑 dtype、显式舍入、变换参数、派生 NPY 和源码来源分别核验。
分配前检查 rank、轴长度、公开量预算，变换结果采用独立复制语义。

v31 先规范化公开表达式，再规范化密文变换；重载通过数组与记录重算证明、核验阶段桥接。
旧版本保持旧次序和门禁。证据支持本批小型上下文的接入、生成与执行兼容性，
较大形状、权重和组合分别验收。默认 v18、latest v28 保持。

## 绑定与证据

- SDK：`772d9b987f4604682b8b0f06d5de95d66178ef967055d36eca7e6ce31e91639f`。
- 本阶段：`bc3bea655c8352949e5967a1b2eeaca38738ba3f2f0b0eff3769ad169da5f39d`。
- 规则审计：`fd07ca835d0fa80ee854c3b9e36cb1360606188eb9e077380e8cbbc4106228bb`。
- Agent 审计：`ce22163e04d0610924db83a52483bbf68492a06e99e3557a35151f5072ed63a4`。
- 原始证据：ARM work root `results/agent-public-transforms-capture-r1456`、
  `agent-public-transforms-offline-r1457`、`agent-public-transforms-live-r1457`。
- [机器可读结果](agent-public-tensor-transforms-acceptance-r1458.json)。
- [规则范围](agent-public-transforms-offline-scope-r1457.json) ·
  [Agent 范围](agent-public-transforms-live-scope-r1457.json)。

历史 v30 报告与绑定保持；初始探针、夹具和开发检查日志保留。
短暂 SSH 握手失败未改变网络或重启付费批次。
全局产物约 33.099 / 48 GiB，磁盘余量约 36.423 GiB。
编译≤2、链接≤1、API/密态串行、每例/批4 GiB、余量4 GiB及原沙箱边界保持。
安装、下载、后端库、参数、模型、reference、门限、网络、commit/push 变更均为 0。

## 未验证与下一步

本批实际密态形状为小型 rank 1/2；更高 rank、大权重和物理布局分别验收。
逐元素重复采用统一正整数次数；向量/零次数、动态行为仍未开放。
JSON 公开变换尚未接入。x86 配置保留，本批未在 x86 实机重测。
原模型应用精度、真实 CPU bootstrap、CUDA 实机和完整 ResNet20分别保持原状态。

下一批继续常见 PyTorch 表达式组合、权重与物理布局容量，先审计缺口并冻结新增范围；
计划内离线与真实测试沿用用户总授权。后端、参数、安装或资源扩限另行申请。

~~~sh
python3 -B -m json.tool docs/baseline/agent-public-tensor-transforms-acceptance-r1458.json
~~~
