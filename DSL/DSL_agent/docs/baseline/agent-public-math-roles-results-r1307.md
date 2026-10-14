# 公开数学常量绑定修正（r1307）

已接入显式策略 `descriptors-math-roles-v2`。公开标量与多项式系数现在包含数值、基与次数、零系数标记，以及它们与预绑定 Plain 名称的对应关系。大型权重继续使用形状和内容哈希。旧策略 `descriptors-v1` 仍为默认；三份历史请求的字节与哈希保持相同。

修改限于 Agent：公开上下文、统一请求、CLI、应用输入和 Poseidon 候选沙箱的只读模块清单。新模块不读取模型测试数据，也不提供规则 DSL 答案。原文件和权限恢复材料保存在工具目录的 cutover-records；后端、reference 和门限 654 个文件保持。

## 验收

| 范围 | 结果 |
|---|---|
| 旧接口兼容 | 144 passed，0 failed / skipped |
| 实际生产入口 | 25 passed，0 failed / skipped；应用后端使用禁止执行的测试替身 |
| 沙箱清单、源码绑定、资源限制 | 12 passed，0 failed / skipped |
| 原三个来源重新隔离捕获 | 3 passed；48 组独立参考检查，模型和公开权重保持相同 |
| functional GELU 规则候选 | 真实编译、Poseidon CPU 数值、程序包重载及新输入执行通过 |

以上 181 个控制方法不计为密态验收。另行执行的 functional 规则候选进行了 1 次真实编译、2 个密态阶段，7 组输入 / 28 值；最大绝对误差 2.4062e-11。缓存阶段未运行。

本批比较已声明的多项式，原 GELU 应用精度未认证。新策略的真实 Agent 生成仍待新绑定评测；原 r1306 的 3 项真实 Agent 数值失败保持原记录。

## 边界

数学角色最多 256 个值、48 KiB；完整生成上下文保持 128 KiB，可信请求保持 4 MiB。万参数随机权重仍以描述符接入：显式 explicit-v10 上下文 126,168 字节，可通过；同一模型 explicit-v11 为 132,710 字节，按原边界拒绝。

原捕获报告绑定旧源码版本，新版本重新捕获同一来源后取得新证据。初次旧绑定拒绝、开发入口路径错误和沙箱缺失模块的失败报告全部保留。没有修改旧绑定、reference、模型或误差门限。

安装、下载、后端修改、参数修改、付费调用、提交、推送和 PR 均为 0。上一个已批准 r1306 批次实际使用 12 次生成 / 12 次 HTTP，见[独立报告](agent-gelu-live-results-r1306.md)。

## 使用

~~~bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case imported-model/model.json --prepare \
  --public-weights descriptors-math-roles-v2 \
  --unified-guidance explicit-v11 \
  --unified-packing capacity-public-tensors-v9
~~~

组件选项为 `public_weight_policy: "descriptors-math-roles-v2"`。名称和系数映射由可信准备代码生成；缺少必要绑定时在准备阶段拒绝。

下一步：冻结修正后的三个来源请求及调用预算，再申请真实 Agent 重测；同时继续统一图中的 MHA 接入。容量、其他算子、bootstrap、整模型和 CUDA 的原计划保持。

[机器报告](agent-public-math-roles-results-r1307.json)
