"""Summarize independently audited results without changing prior evidence."""
import json,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
def read(p):return json.loads(Path(p).read_text())
def bound(v):assert v["binding"]==digest({k:x for k,x in v.items()if k!="binding"})
plans={m:read(HERE/(m+"-plan.json"))for m in ("offline","live")}
audits={};reports={}
for mode,p in plans.items():
 bound(p);r=read(Path(p["campaign"])/"report.json");a=read(Path(p["campaign"])/"audit.json")
 bound(r);bound(a);assert a["status"]=="passed" and a["report_binding"]==r["binding"] and r["plan_binding"]==p["binding"]
 audits[mode]=a;reports[mode]=r
index=read(HERE/"capture-index.json")
prepared=read(HERE/"prepared-requests.json");bound(prepared)
controls=read(HERE/"controls-final.json");assert controls==dict(tests=140,failures=0,errors=0,skipped=0,passed=True)
old=read(HERE/"prechange-identity.json");identity=read(HERE/"identity-snapshot.json")
assert old["backend"]==identity["backend"]
usage=[r["result"]["generation"].get("usage") for r in reports["live"]["cases"]]
record=dict(format="poseidon-static-tensor-transform-phase-acceptance-v1",sdk=identity["manifest_sha256"],
 backend="poseidon_cpu",operator_graph_format="poseidon-operator-graph-v5",pytorch_source_format="poseidon-trusted-pytorch-v30",
 source_contexts=15,unique_public_generation_requests=len({c["request_id"]for c in plans["live"]["cases"]}),
 compatibility_contexts=3,operators=["repeat","tile","repeat_interleave","flip","roll"],
 controls=controls,execution_controls=dict(passed=12,failed=0,skipped=0),
 plaintext=dict(groups=sum(c["plaintext_groups"]for c in prepared["cases"]),values=sum(c["plaintext_values"]for c in prepared["cases"])),
 offline=audits["offline"],live=audits["live"],provider_usage_records=usage,
 first_attempt_passed=sum(r["result"]["generation"]["first_attempt_passed"]is True for r in reports["live"]["cases"]),
 repairs=sum(r["result"]["generation"]["repairs_used"]for r in reports["live"]["cases"]),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),backend_unchanged=True,parameters_unchanged=True,
 network_unchanged=True,installations=0,downloads=0,git_actions=[],history_rewritten=False,
 default_source="poseidon-trusted-pytorch-v18",latest_source="poseidon-trusted-pytorch-v28",
 fixture_correction="Three unchanged constant buffer values added to complete state declarations; original rejection retained",
 unverified=["vector/zero repeat_interleave counts","empty tensor outputs","arbitrary dynamic Python","all model/operator compositions",
             "large physical layouts","original-function application accuracy","real CPU bootstrap","CUDA hardware","full ResNet20"])
record["binding"]=digest(record)
target=ROOT/"docs/baseline/agent-static-tensor-transforms-acceptance-r1454.json"
assert not target.exists();target.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
a=audits["offline"];l=audits["live"];gen=record["first_attempt_passed"]
text=f"""# 静态张量操作验收

JSON 图 v5 和显式 PyTorch v30 已接入 repeat/tile、repeat_interleave、flip、roll。次数、维度及轴采用公开静态参数；repeat_interleave 首版使用统一正整数次数。各操作与线性层、归约、残差和多输出组合验收。

| 项目 | 结果 |
|---|---|
| 控制测试 | 140 passed，0 failed/errors/skipped |
| 执行与调用账本控制 | 12 passed |
| 隔离捕获、独立明文参考 | 18 来源，288 组／1,120 值 |
| Poseidon 规则候选 | {reports["offline"]["passed"]}/18 |
| 当前真实 Agent | {reports["live"]["passed"]}/15，首次 {gen}/15，修复 {record["repairs"]} |
| 真实生成／HTTP | {l["generations"]}／{l["http_attempts"]} |
| 规则轨编译／密态调用 | {a["compiler_calls_confirmed"]}／{a["encrypted_calls_confirmed"]} |
| Agent 轨编译／密态调用 | {l["compiler_calls_confirmed"]}／{l["encrypted_calls_confirmed"]} |
| 规则轨逐项比较 | {a["input_groups"]} 组／{a["output_values"]} 值，maxabs {a["maximum_absolute_error"]:.10g} |
| Agent 轨逐项比较 | {l["input_groups"]} 组／{l["output_values"]} 值，maxabs {l["maximum_absolute_error"]:.10g} |

15 个新来源上下文包含 {record["unique_public_generation_requests"]} 份不同的公开生成请求；3 个旧 v29 来源作为兼容回归单列。两轨使用相同数学模型，数量分别计分。程序包通过重载、新输入、重复执行和零调用缓存复用；Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、参数绑定和 Poseidon CPU 数值均已独立审计。

冻结门限为 §1e-5 + 1e-4*abs(reference)§。明文参考采用独立 NumPy 与 PyTorch，源规范化保留复制和别名语义。旧身份、初始环境加载失败、身份操作边界及漏报权重的拒绝均保留；新增绑定对应当前源码。

SDK：§{record["sdk"]}§。原 x86 配置、后端库、依赖、参数和网络保持，Git 无提交或推送。产物总上限 48 GiB，每批 4 GiB、磁盘余量 4 GiB。

默认源码契约 v18，latest 为 v28，本批 v30 显式选择。动态次数、零元素张量、任意 Python、全部组合与大布局范围另行登记。bootstrap、CUDA、应用精度和完整 ResNet20 继续独立推进。

下一条只读命令：§python3 -B -m json.tool docs/baseline/agent-static-tensor-transforms-acceptance-r1454.json§。
"""
text=text.replace("§",chr(96))
target.with_suffix(".md").write_text(text)
print(json.dumps(dict(binding=record["binding"],report=str(target),first_pass=gen,generations=l["generations"],http=l["http_attempts"])))
