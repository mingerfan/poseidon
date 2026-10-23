# 三例真实 Agent 验收（r1304）

三例均首次生成通过；失败、跳过、未运行均为 0。候选来自本轮 DeepSeek，未人工替换。

| 模型 | 验收输入／值 | 新输入／值 | 最大绝对误差（合计） | 结果 |
|---|---:|---:|---:|---|
| 有序写入＋Linear | 4／20 | 3／15 | 6.55756935e-09 | 通过 |
| 有序写入＋近似 LayerNorm | 4／32 | 3／24 | 1.27824303e-07 | 通过 |
| 公开权重 split→Linear | 4／16 | 3／12 | 3.62782500e-08 | 通过 |

共 3 次真实编译、6 个密态执行与最终解密阶段、3 次缓存复用。21 组输入、119 个输出值逐项通过；MAE 为 1.2899e-8，最大绝对误差 1.2783e-7。冻结门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

执行链为 Hecate → Dacapo-modified → RuntimePlan／OperatorSpec／plaintext bundle → CKKS Runtime → Poseidon CPU。每份程序包已重载执行三组新输入，并通过缓存复用；复用没有再次生成或编译。编译验证子报告的 0 API 表示该子步骤仅回放固定候选，本轮真实生成来源由独立 generation-provenance 和调用账本记录。

DeepSeek deepseek-flash／high／stream：实际 3 次生成、3 次 HTTP，语义修复与传输重试均为 0。响应记录 prompt 34,677、completion 99,970、total 134,647 tokens；其中 reasoning 99,263 已包含在 completion 中。Provider 未提供美元金额，费用尚未核验。执行至终态报告约 12.88 分钟；全部预算保持，相关 worker 已退出。

批准、源码与运行绑定：

- 提案：`fadf85332f2ee938ff16c61adda89461edefd27403c6ad928af5588fd6de84b0`。
- 运行：`a8bbbbff40dd198a08c2907dbf295762a54b8bfa4916005344e1ee5c6e09ecec`。
- 验收报告：`6b4fc3b1abd3a58b38627e3ccd98b2bd77a6f41c2ce84293c715ebbaa0fd7448`。
- 当前源码 706 文件、验证闭包 611 文件；654 个后端及 reference／门限保护文件与七个执行工具核验一致。
- 预启动控制测试 60 项通过；本批没有重复运行该控制套件。

[机器可读报告](agent-v12-live-rebound-results-r1304.json) · [批准记录](agent-v12-live-rebound-approval-r1304.json) · [原提案](agent-v12-live-rebound-proposal-r1304.md)

程序包与原始证据保存在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v12-source-live-rebound-v2`。各例任务、原始响应、源码、验证产物和包哈希见机器可读报告；历史报告未重绑。

本批验证的是三个 V12 来源模型在当前联合版本上的自由生成与执行。近似 LayerNorm 与已声明的近似参考比较，原始函数的应用精度另计。原 76 项的 56 通过／20 失败保持，本批单列；不据此宣称所有 v13／v14 组合、所有 DSL、x86、bootstrap 或 CUDA 已验收。

本轮仅新增授权和结果文档并更新接续。后端、模型、reference、安全参数和门限保持；安装、下载、提交、推送、PR 均为 0。

后续优先把已有产物存储下界检查接入首次执行和包复用入口，再分组推进 GELU 常见写法与 MultiheadAttention。下界只能用于拒绝确定超预算的产物，合格仍不能证明峰值内存足够。v13／v14 新生成验收、物理容量、大图密文交接、bootstrap、整模型与 CUDA 仍按原计划推进；新付费范围另行冻结。
