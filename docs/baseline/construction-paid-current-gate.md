# 最近构造语法：历史付费启动门禁与复核

> 2026-09-16 更新：下文是之前审批拒绝时的历史快照。用户随后对11项原生函数
> 构造测试给予具体授权；该批已真实调用DeepSeek并全部完成、独立审计通过。
> 当前结果见 [原生函数付费结果](native-function-paid-results.md)。历史v19–v22结果没有重标或重复计费。

## 本轮结果

用户要求继续付费API测试，做到最近新增构造语法的逐项真实Agent覆盖。
当前工作区为 `D:\Code Space\Poseidon`，WSL 为 `/mnt/d/Code Space/Poseidon`。
分支 `feat/agent-dsl-correctness`，HEAD `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`。

**新付费批次未启动：执行环境审批在创建进程前拒绝了命令。**
没有本轮新API调用、token usage、新付费批次目录或新密态执行结果。
没有换进程入口、间接脚本或降低检查来绕过拒绝。
原因是审批要求对具体外发内容及高预算付费操作取得精确授权；这不是TLS、
Agent生成、Dacapo编译或FHE数值失败。

## 现有真实证据已重新独立审计

| 冻结版本 | 范围 | 最终通过案例 | 观察项 |
|---|---|---:|---:|
| v19 | 公开构造、闭包、参数绑定、迭代、字符串、多项式、对象数组、字典 | 30/30 | 117/117 |
| v20 | 对象数组算术、broadcast、overlap、augassign | 10/10 | 10/10 |
| v21 | 公开标量提取、float/int、item、size-one、截断 | 15/15 | 19/19 |
| v22 | 对象一元运算、0-D、view/fresh及混合类型 | 8/8 | 12/12 |

本轮只审计旧文件，不重复调用模型或执行FHE。63个有效候选、1008个输出值，
最大绝对误差 `3.3573417601573396e-8`，加权MAE `5.272248406714349e-9`。
审计：`/home/lhy/poseidon-work/results/recent-construction-audit-mdzve5gu/report.json`。
SHA256：`55ec49b375c80d7159ff92ade9f536bb39b3d4696864a8694eb931aaebb7f549`。

另行审计最新已完成重复批 `agent-batch-go_bnk71`：8/8最终、7/8首次通过，
12/12项；128输出值最大绝对误差 `2.7667490964944363e-8`，
加权MAE `4.804414790917505e-9`。其135234个已报告tokens属于历史批次，
不计作本轮用量；一次历史超时未返回usage，实际账单仍未查询。
本轮审计：`/home/lhy/poseidon-work/results/object-unary-agent-audit-167_u9nd/report.json`。
SHA256：`68dc5d1fa549cf2185f70683ab7664036c1c749851cc617f021959ac4c33a4c4`。

### v22逐项落点

| 观察项 | 通过的真实Agent案例 |
|---|---|
| operator.USub | ou-neg、ou-zero、ou-view、ou-plain、ou-mixed |
| operator.UAdd | ou-positive |
| call.USub | ou-negative-call |
| call.UAdd | ou-positive-call |
| cipher_cells | ou-neg、ou-zero、ou-view、ou-negative-call、ou-mixed |
| plain_cells | ou-plain |
| public_cells | ou-positive、ou-positive-call、ou-mixed |
| boolean_cells | ou-mixed |
| zero_dim | ou-zero |
| input_view | ou-view |
| fresh.USub | ou-view |
| fresh.UAdd | ou-positive |

观察必须真正执行并通过有限输出影响检查；不是只搜索AST拼写。
mixed类型及view/fresh还检查证据属于同一次运算。
v19的117项中34项有输出扰动敏感证据，83项仅有执行结构证据和源码复核，
不把它们都说成输出敏感覆盖。八个v22案例是同一仿射模型的不同构造形式，
不是八个模型家族。这些有限测试不是全输入形式化证明，也不是全部DSL覆盖率。
真实历史执行后端为 upstream SEAL HEVM CPU，不是Poseidon GPU。

## 当前代码和连接检查

- 当前原生装饰函数调用probe：6/6对照通过；scalar、pair、nested、forward、
  two_inputs、public_argument各自的直接表达与调用表达均经过真实trace/compile，
  对应golden HEVM/CST字节一致。报告：
  `/home/lhy/poseidon-work/results/native-function-calls-qyu6uotp/report.json`，
  SHA256 `c231a4c8fa707d7e23176a11c12bab24b4877c118c06a5fa31ce6c901e4abe2f`。
  这不是Agent生成测试或新密态执行，也没有开放生成程序的多装饰函数调用权限。
- 无凭据TLS：默认路由3/3取得预期401；6478 CONNECT三次连接失败。
  TLS证书及hostname验证始终开启，不改全局代理、不硬编码IP。
  报告 `/home/lhy/poseidon-work/results/deepseek-tls-probe-drnnv73u/report.json`，
  SHA256 `cee7b47753a24166efa8dd186bd3ffdc014ea43abdbc8a0a78993373a2131107`。
  短GET成功不保证1200秒生成连接可靠，也不能证明历史超时根因为TLS。
- 离线回归61项：58通过、3条件跳过、0失败。显式启用上述两个真实证据审计。
  三项skip是未设置 `POSEIDON_CANDIDATE_RESULTS` 的既有候选回放测试。
  provider重试输出来自mock测试，不是额外付费请求。
- 无付费批量worker；WSL剩余约189.13GiB；本轮未产生需删除的新FHE密钥。

## 等待确认的精确外发/费用范围

拟重跑且仅重跑固定v22八例，使用现有provider字段白名单：
合成模型描述/FX图、公开固定常量及来源、shape/layout、DSL规则与构造要求、
response schema、编译profile哈希及请求ID；需要修复时附模型自己的旧响应、
受限编译/运行诊断和聚合误差。不上传整个仓库、`.env`内容、测试输入数组、
reference逐项结果或FHE私钥。API key仅作为向DeepSeek服务认证的请求凭据。

目的服务：DeepSeek官方API `api.deepseek.com`，模型 `deepseek-flash`；
high，单响应上限384000 tokens，API并发上限10（本批8例），native并发2，
每请求硬超时1200秒，最多3轮语义修复、每生成轮最多3次传输重试。
所以最多32个逻辑生成轮、128次HTTP尝试；极端重试可能重复计费，
超时并不保证服务商未处理或未计费。没有核实或承诺货币费用上限。
这些是上限，不是预计用量，也不是无限token预算。

待用户明确允许上述内容外发和配置后再启动。旧批次和失败证据全部保留。
当前工作区既有未提交改动不动；本轮新增本报告，没有安装依赖、sudo、
修改凭据、安全参数或数值门限，没有commit/push/PR。

## 复核命令（不收费）

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 660s python3 scripts/baseline/audit_recent_construction.py
timeout -k 3s 480s python3 scripts/baseline/audit_object_unary_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-go_bnk71
```

后续新语法不能继承这些历史证据。原生装饰函数调用、bootstrap容器修复和
上游HE_Concat等仍按各自语义、Agent开放及后端门禁单独验收。
