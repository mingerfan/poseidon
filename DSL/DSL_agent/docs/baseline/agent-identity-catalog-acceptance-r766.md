# 组件身份目录与依赖审计（r766）

日期：2026-10-05。开发：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。

## 结果

92 项静态与组件测试通过，0 failed/errors/skipped。三个原规则候选完成真实 Hecate tracing、Earth/CKKS 编译、RuntimePlan/OperatorSpec/bundle 检查、Poseidon CPU 密态执行及独立数值比较。
程序包重载、三组新输入重复两次和零生成缓存复用通过。

共 42 组密态执行、364 个输出值，最大绝对误差 `5.2761597974e-07`。门限保持 `1e-5 + 1e-4*abs(reference)`。
三个模型、请求和候选源码与 r758 逐字一致，当前编译及执行采用新绑定：
`41d6b53e268d64fcef08b991481b8d048c90e8b7220e2ca73130246cfa89ed69`。

来源均为规则基线；新增 Agent 生成、API 和 HTTP 为 0。本批未验收 bootstrap、CUDA 或新增输入功能的真实 Agent 生成。

## 改动

`application-source-catalog-v1.json` 保存四阶段入口、可选执行入口和五项动态加载声明，保持原入口及授权集合。
加载器检查大小、字段、重复键、相对路径、符号链接和动态调用格式；目录文件及实际选中的源码都参与身份哈希。
每次身份构建重新读取目录和当前文件，不缓存有效性判断。

此前分析器把自身入口清单中的 `.py` 字符串当作执行依赖，造成阶段间错误关联。
本次将清单移为受绑定的数据，增加真实分析器和真实源码闭包检查。身份格式、旧请求哈希和程序包格式保持；源码变化正常产生新执行身份。
旧报告及绑定原样保留，旧绑定不会被改写为当前通过。

## 实际依赖边界

| 阶段 | 原闭包文件数 | 当前闭包文件数 |
|---|---:|---:|
| 生成 | 650 | 651 |
| 编译 | 650 | 650 |
| 执行 | 650 | 650 |
| 验证 | 650 | 650 |

当前四阶段仍共享 650 个文件。此改动消除了清单造成的错误依赖，尚未完成真实运行路径的阶段独立性。

仍存在以下真实调用路径：

- `component_backend → run_candidate → unified_graph_lowering`：资格验收仍适配旧单例 CLI。
- `application_backend → component_backend → deepseek_provider`：失败反馈仍由 Provider 模块处理。
- 共享的 `candidate_bundle` 和请求重建代码仍同时影响多个身份。

下一批先提取不构造 Provider 的资格验收会话及公开反馈接口，再按真实依赖调整阶段入口。
继续保留保守闭包校验，不能仅移除哈希项来宣称独立。

## 验收范围

| 样例 | 来源 | 编译／密态／数值 | 重载／新输入／复用 |
|---|---|---|---|
| `axes_2` | 可信 PyTorch v6 静态索引，多输出结构 | 通过 | 通过 |
| `duplicate_input` | JSON，重复输入输出 | 通过 | 通过 |
| `duplicate_square` | JSON，共享平方输出 | 通过 | 通过 |

独立审计检查 48 组双参考、36 个结构化输出值、60 个程序包文件和 56 个执行产物哈希。
648 个后端文件及 6 个 reference／门限／产物检查器与保护快照一致。
冻结参数、编译配置、库身份、评估密钥执行边界和逐项解码均核对通过。

## 证据与资源

- 静态测试：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-catalog-preflight-r762/tests-r763.json`。
- 实际依赖审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-catalog-preflight-r762/dependency-audit-r764.json`。
- 独立后端审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-catalog-preflight-r762/audit-r765.json`。
- 当前执行：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-identity-catalog-backend-r764`。
- [机器报告](agent-identity-catalog-acceptance-r766.json) · [旧执行报告](agent-pytorch-index-acceptance-r761.md)。

审计时新增产物 20.39 MiB，全局 22.36 GiB，余量 46.15 GiB。
测试和后端 worker 合计约 130.29 秒；审计终端另记在 `audit-terminal-r765.json`。
编译≤2、链接≤1、密态串行；安装、下载、SDK 重建、参数／后端／reference 修改及 Git 发布为 0。

## 依赖审计命令

~~~bash
python3 -B scripts/baseline/benchmarks/tools/audit_application_dependencies.py
~~~

该开发工具列出实际闭包、目录哈希及目标模块的最短依赖路径，只读取源码。
