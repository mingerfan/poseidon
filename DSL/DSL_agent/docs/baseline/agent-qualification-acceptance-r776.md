# 独立资格验收会话（r776）

日期：2026-10-05。ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。

## 结果

应用后端直接调用 `QualificationSession.qualify(request, candidate, validation_level=...)`，共用原 `CandidateValidationAdapter`。
旧 CLI 的准备、生成和修复方式保留，沙箱预检及候选验证使用同一个会话。

静态回归 235 项：232 passed、0 failed/errors、3 skipped。跳过项为依赖旧 SEAL 实验目录的条件证据检查，未计入通过。
三个原规则候选、显式编译等级及旧 Linear 故障场景完成真实后端验收。

当前数值通过范围：46 组密态输入、372 个输出值，最大绝对误差 `1.2803859728e-07`。
门限保持 `1e-5 + 1e-4*abs(reference)`。新增 Agent 生成、API、HTTP 为 0。

## 接口与职责

| 接口 | 职责 |
|---|---|
| `component_backend.qualify` | 调用会话，独立审计通过结果，返回既有组件结果 |
| `QualificationSession.qualify` | 验收一个固定请求和候选；准备独立 reference、沙箱、编译、产物及指定等级验证 |
| `QualificationSession.preflight/evaluate` | 旧 CLI 共用的沙箱与候选验证接口 |
| `public_feedback_contract.public_feedback` | 原公开反馈信封的字段、数值、长度检查 |
| `deepseek_provider.public_feedback` | 原错误类型及错误码兼容包装 |

生成、传输重试、修复、取消、任务预算和程序包交付继续由原 Application／Controller 管理。
内部单例验收沿用一次离线 replay 的证据格式，最多验证一个候选，付费调用为零。
应用资格验收无需 CLI 参数、临时响应文件或确定性 DSL 答案；`rule_baseline_status` 为 `not_requested_by_component`。

会话绑定实际工具、配置、Hecate、源码目录、动态加载声明、请求、权重、测试数组、逻辑绑定和冻结 reference 设置。
Python 入口与 JSON 数据分别处理；所有原 CLI JSON 声明仍核验哈希。
网络隔离、AST、资源、原生执行槽锁和数值门限沿用既有实现。

## 当前后端验收

| 场景 | 等级与结果 |
|---|---|
| `axes_2`，可信 PyTorch v6 静态索引 | numerical：真实编译、密态、逐项比较、结构化输出、重载及重复执行通过 |
| `duplicate_input` | numerical：真实编译、密态、逐项比较、重载及重复执行通过 |
| `duplicate_square` | numerical：真实编译、密态、逐项比较、重载及重复执行通过 |
| `duplicate_input`，显式 compiled | 编译及产物通过；重载保持 compiled，密态及数值声明为 false |
| 旧 catalog Linear `--self-test` | 保持 response_parse 失败 → numerical_comparison 失败 → complete 通过 |

三个规则候选的模型、请求、源码与 r758 完全一致，当前重新编译与执行建立新绑定：
`86d810059a7905f1f43baff869495be887f4b5211864ba8ba38b11d987f10731`。
新输入三组重复两次、原四组重载及零生成缓存复用均通过。
CLI 的故意错误另有 4 组密态执行、数值拒绝，单列为负例，不计入上面的数值通过范围。

独立审计核对 48 组双参考、36 个结构化输出值、77 个程序包文件、103 个执行产物哈希。
CLI Linear 使用原 `layers.0.weight/bias` 进行独立 NumPy 矩阵计算，与原冻结 reference 和密态结果逐项比较。

## 保留的记录

- r768 初次静态检查：228 passed、2 failed、1 error、3 skipped。两条旧 CLI mock 假设及一个新增夹具字段随后修正。
- r769：231 passed、0 failed/errors、3 skipped。
- r770：三个规则候选及 compiled 通过；旧 CLI 初始化误将 JSON 当作 Python 根入口。原 report、日志、程序包和会话源码快照保留。
- r771：232 passed、0 failed/errors、3 skipped，增加 JSON 哈希篡改拒绝。
- r772：当前源绑定下完整验收成功。
- r773 审计工具把无关公开缓冲误当作 Linear bias；r775 按明确参数名重新审计同一产物，没有重跑密态或改 reference。

原绑定 r770 的通过记录仅属于其原版本，不冒用为当前结果。模型和候选不计重复新增。
648 个后端文件及 6 个 reference／门限／产物检查器与保护快照一致。

## 尚待的组件边界

资格验收实际执行路径已脱离 CLI 和 Provider 构造，但源码身份仍采用整个模块的保守闭包。
当前生成 654 个文件，编译／执行／验证各 653 个，共享 653 个。

主要剩余关联：

- `candidate_bundle → application_generation → deepseek_provider`：生成来源核验与生成适配仍在同一模块。
- `candidate_bundle → benchmark_runner`：共享文件读取工具带入开发工具依赖。
- 沙箱／helper 清单、历史保留工具仍引用旧单例和规则 lowering。
- 同一个已绑定 worker 仍使用原 WORK/SDK 全局路径。

下一批优先拆分来源核验和数据读取接口，再审计实际阶段闭包。
不通过删去源码哈希项来实现独立性；未完成部分保留原有拒绝规则。

## 证据与资源

`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-qualification-preflight-r767` 保存测试、历史快照、预算、绑定和独立审计。
当前程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-qualification-backend-r772`。

- 最终静态：`tests-r771.json`。
- 实际闭包：`dependency-audit-r772.json`。
- 独立数值及产物审计：`audit-r775.json`。
- [机器报告](agent-qualification-acceptance-r776.json) · [应用接口](application-agent-component-v1.md)。

最终检查时新增 45.09 MiB，全局 22.40 GiB，余量 46.10 GiB。
已记录测试／后端／审计 worker 约 316.68 秒；失败尝试包含在内。
最终核验659个冻结Agent文件、23个模型／请求文件及历史绑定一致；原1374条修改路径保留，当前1380条。当前无活动项目worker，git diff --check通过。详见结果目录中的final-check-r776.json。
编译≤2、链接≤1、密态串行；安装、下载、SDK 重建、后端／参数／reference 修改和 Git 发布为 0。
本批证明组件和旧 CLI 兼容性，不宣称新增输入功能真实 Agent、bootstrap、CUDA 或 ResNet20 完成。
