"""Publish this bounded phase only after all final-source evidence checks."""
import json,hashlib,subprocess
from pathlib import Path
from benchmark_graph import digest,require
from benchmark_runner import DEFAULT,load,dump
from semantic_benchmark_execution import runtime_sources
from workspace_paths import ROOT,WORK

base=WORK/"results";prefix="upstream-virtual-r26-verified-";paths=[]
def read(name):
    p=base/(prefix+name);r=json.loads(p.read_text());paths.append(p);return r
source=digest(runtime_sources());complete=read("verification-complete.json")
require(complete["status"]=="passed" and complete["source_sha256"]==source,"Incomplete current verification")
unit=read("unit-final.json");plain=read("plaintext/report.json");reject=read("rejections.json")
compat=read("compatibility.json");original=read("original-corpus-audit.json")
require(unit["success"] and unit["passed"]==245 and not unit["failures"] and not unit["errors"],"Unit verification")
require(plain["passed"]==1200 and plain["failed"]==0 and reject["passed"]==48 and reject["failed"]==0,"Offline results")
batches={}
for name,count in (("native",44),("helper-directed",47),("regression",2),("chunk-regression",18)):
    r=read(name+"/report.json");require(r["passed"]==count and r["failed"]==0,"Batch failed "+name)
    if "source_sha256" in r:require(r["source_sha256"]==source,"Source drift")
    batches[name]={k:r[k] for k in ("planned","passed","failed","blocked","skipped","max_absolute_error",
        "weighted_mae","compared_values","seconds") if k in r}
audits={}
for label,count in (("native",44),("helper_regression",47),("legacy_regression",2),("chunk_regression",18)):
    r=read(label+"-compiler-evidence.json");require(r["audited"]==count,"Compiler audit count");audits[label]=count
rows,index=load(DEFAULT);coverage=json.loads((DEFAULT/"coverage.json").read_text())
require(len(coverage["helper_directed_tasks"])==47 and compat["old_helpers_identical"]==38,"Compatibility")
git=lambda *a:subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
remaining=["HE_MaxPad/bootstrap","HE_Max/bootstrap","HE_ReLU/bootstrap","chunk/helper joint layout",
    "parameter/resource boundaries","w40 precision failure","3 rule-budget and 2 historical compile failures",
    "final-source full encrypted corpus and 627 constructions","separately approved paid Agent evaluation"]
summary=dict(schema=1,phase="upstream_virtual_prefix_r26",status="phase_passed_overall_plan_incomplete",
    checkout=str(ROOT),branch=git("branch","--show-current"),head=git("rev-parse","HEAD"),
    runtime_source_sha256=source,profile="upstream-poly-virtual-prefix-v8",suite=str(DEFAULT.relative_to(ROOT)),
    model_count=1200,topology_groups=779,construction_tasks=627,helper_tasks=47,rejection_tasks=48,
    manual_batches=batches,fresh_executed_runs=111,unique_model_count_not_claimed=True,compiler_audits=audits,
    original_virtual_helper_corpus={k:original[k] for k in ("planned","passed","failed","blocked","skipped","families","splits",
        "models_unchanged","total_actual_inner_virtual_calls","total_public_zero_eliminations")},
    unit=unit,plaintext={k:plain[k] for k in ("planned","passed","failed","skipped","tests","max_reference_error")},
    rejection={k:reject[k] for k in ("planned","passed","failed","skipped","not_run","scope")},
    old_requests_unchanged=compat["old_requests"],old_helper_tasks_unchanged=38,
    compiler_configuration="seal-cpu-eva-w45-v1",default_configuration_changed=False,
    tolerance="abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",security_checks_unchanged=True,
    holdout_scope="Original split preserved; manual backend acceptance is not unseen Agent scoring",
    actual_compile=True,actual_seal_cpu=True,actual_decrypt=True,paid_api_calls=0,
    installed=False,downloaded=False,commit=False,push=False,pr=False,x86_retested=False,remaining=remaining,
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
target=ROOT/"docs/baseline/upstream-virtual-prefix-acceptance.json";require(not target.exists(),"Preserve summary");dump(target,summary)
n=batches["native"]
evidence=f"""已确认：原冻结24模型（3类各8）及20个边界/组合fixture全部真实编译、SEAL执行、解密通过。
最终源码批次为44主案例、47helper定向、18分块、2旧native/public，共111次含重叠执行；
不是111个不同模型，全部0 failed/blocked/skipped。111份编译/密态产物另分四片审计。
主批 {n['compared_values']} 个解密值，max absolute error {n['max_absolute_error']:.16g}，
MAE {n['weighted_mae']:.16g}；全部满足冻结门限。
保留真实tracing、Earth/CKKS IR、HEVM/CST、SEAL执行、四组逐项误差和helper调用/返回贡献证据。
编译器relinearization证据仍限融合指令/源码/输出结构，不冒充逐操作运行时打点。

245单测passed，0失败/错误；3个单项证据依赖skip及包含5项的历史r18源码测试类skip，
合计8项不计pass。1200模型/19200组双参考及48静态拒绝全部通过。
{compat['old_requests']}旧请求字节一致，旧38helper任务、原模型分片和构造/拒绝任务保持。
原24模型split为{original['splits']}；任何人工检查过的留出图都不称为未接触图Agent成绩。
原24模型tracing共{original['total_actual_inner_virtual_calls']}次真实内部helper调用，
不是每组密态输入重新运行Python helper。列出的范围通过不代表所有参数均通过。
默认freeze semantic-v1-virtual-r26；1200模型/779拓扑/627构造不变，helper任务47、拒绝48。
源码SHA：{source}。
原始证据：{base}/upstream-virtual-r26-verified-*。
报告及完整性hash见[机器验收摘要](upstream-virtual-prefix-acceptance.json)。
"""
p=ROOT/"docs/baseline/upstream-virtual-prefix.md";s=p.read_text();require("R26_ACCEPTANCE_PENDING" in s,"Documentation placeholder")
p.write_text(s.replace("R26_ACCEPTANCE_PENDING",evidence))
intro="""当前 r26：[固定65536槽helper](upstream-virtual-prefix.md)增加可选v8：真实MPBN、Linear、ReshapeLinear。
原24模型＋20补充案例密态通过，47helper、18chunk、2旧入口回归通过；共111执行含重叠。
245单测通过，8项证据依赖跳过；全1200/19200双参考通过。不是全量密态或付费Agent成绩。
默认freeze semantic-v1-virtual-r26；1200模型/779拓扑/627构造不变，helper47、拒绝48。
旧profile及38helper任务保持；bootstrap、chunk/helper联合和最终全量评测仍未完成。
"""
for name in ("agent-current-contract.md","supported-inputs-models-configurations.md","semantic-benchmark-v1.md"):
    p=ROOT/"docs/baseline"/name;heading,rest=p.read_text().split("\n",1)
    p.write_text(heading+"\n\n"+intro+"\n"+rest.lstrip().replace("当前 r25：","历史 r25：",1))
p=ROOT/"scripts/README.md";s=p.read_text().replace("当前helper目录38任务。","v7阶段helper目录38任务。")
s+="\n统一图可选择 --unified-helpers upstream-poly-virtual-prefix-v8，启用固定参数的真实MPBN、Linear、ReshapeLinear。\n新增调用签名均为 (cipher_expr, zero_ct)，零密文由已有可信运行器提供；旧profile保留，当前helper目录47任务。\n范围与准备入口见[固定槽位helper说明](../docs/baseline/upstream-virtual-prefix.md)。\n";p.write_text(s)
handoff=f"""## 最新接续：MPBN / Linear / ReshapeLinear r26（2026-09-22）

已确认：仅ARM Ubuntu /home/lhohy/Code/Poseidon、lhy-agent-dsl，HEAD {summary['head']}，修改未提交。
profile upstream-poly-virtual-prefix-v8，默认freeze semantic-v1-virtual-r26。
完整65536逻辑槽的稀疏虚拟块、全常量hash、真实原helper、最终前缀投影；物理16384槽/安全参数不变。
原24模型+20补充真实编译/SEAL/解密通过，主批{n['compared_values']}值max {n['max_absolute_error']:.9g}。
47helper、18chunk、2旧入口通过，共111次含重复执行，111产物审计通过。
245单测passed/0失败；3单项+含5项历史类skip。1200/19200双参考与48静态拒绝通过。
{compat['old_requests']}旧请求、38旧helper任务不变；1200模型/779拓扑/627构造不变，helper47。
原24划分 {original['splits']}，人工验收不称未接触图Agent评测。

源码SHA {source}；说明upstream-virtual-prefix.md，摘要upstream-virtual-prefix-acceptance.json。
有效证据results/upstream-virtual-r26-verified-*。初始审计顺序错误、合并测试frontend类型错误日志均保留。
全部批次结束，无安装/下载/付费/commit/push/PR；Mac副本未动，x86/GPU未重测。
基于证据：所列helper范围有人工真实密态及有限贡献证据；不外推全部参数/全部DSL/Agent生成能力。
未完成：3类bootstrap阻塞、chunk/helper联合、参数/预算边界、旧w40精度、3规则预算及2旧编译失败、
最终源码全量密态/627构造、单独获批后的Agent。总体goal仍未完成。
下一实施：优先调查chunk/helper联合布局或3规则预算/2历史编译失败；保持版本化契约和旧失败证据。
下一只读命令：POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/benchmark.py helper-directed。

"""
p=ROOT/"docs/baseline/mac-migration-handoff.md";heading,rest=p.read_text().split("\n",1);p.write_text(heading+"\n\n"+handoff+rest.lstrip())
print(json.dumps(dict(status=summary["status"],source_sha256=source,passed_runs=111,unit_passed=245,reports=len(paths))))
