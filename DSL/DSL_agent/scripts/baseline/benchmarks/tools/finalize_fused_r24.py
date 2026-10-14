
"""Finalize this local phase only after current-source evidence is complete."""
import json,hashlib,subprocess
from pathlib import Path
from benchmark_runner import dump,DEFAULT,load
from benchmark_graph import digest,require
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import ROOT,WORK
base=WORK/"results";prefix="upstream-fused-r24-verified-";paths=[]
def read(name):
    p=base/(prefix+name);r=json.loads(p.read_text());paths.append(p);return r
source=digest(runtime_sources());complete=read("verification-complete.json")
require(complete["status"]=="passed" and complete["source_sha256"]==source,"Current verification incomplete")
unit=read("unit-final.json");plain=read("plaintext/report.json");reject=read("rejections.json")
compat=read("compatibility.json");original=read("original-corpus-audit.json");mapping=read("static-mappings.json")
require(unit["success"] and unit["passed"]==223 and not unit["failures"] and not unit["errors"],"Unit results")
require(plain["passed"]==1200 and plain["failed"]==0 and reject["passed"]==48 and reject["failed"]==0,"Offline results")
batches={}
for name,count in (("native",27),("helper-directed",35),("regression",2),("chunk-regression",18)):
    r=read(name+"/report.json");require(r["passed"]==count and r["failed"]==0,"Batch failed "+name)
    if "source_sha256" in r:require(r["source_sha256"]==source,"Source drift")
    batches[name]={k:r[k] for k in ("planned","passed","failed","blocked","skipped","max_absolute_error",
        "weighted_mae","compared_values","seconds") if k in r}
for label in ("native","helper_regression","chunk_regression"):read(label+"-compiler-evidence.json")
rows,index=load(DEFAULT);coverage=json.loads((DEFAULT/"coverage.json").read_text())
require(len(coverage["helper_directed_tasks"])==35 and compat["old_requests"]==193 and compat["old_helpers_identical"]==29,"Compatibility")
paths += [base/"upstream-fused-r24-native/report.json",base/"upstream-fused-r24-final-unit-final.json"]
git=lambda *a:subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
remaining=["HE_MPBN","HE_DS","HE_Linear","HE_ReshapeLinear","HE_MaxPad/bootstrap","HE_Max/bootstrap","HE_ReLU/bootstrap",
    "chunk/helper joint layout","parameter/resource boundaries","w40 precision failure",
    "3 rule-budget and 2 historical compile failures","final-source full encrypted corpus and 627 constructions",
    "separately approved paid Agent evaluation"]
summary=dict(schema=1,phase="fused_upstream_spatial_r24",status="phase_passed_overall_plan_incomplete",
    checkout=str(ROOT),branch=git("branch","--show-current"),head=git("rev-parse","HEAD"),
    runtime_source_sha256=source,profile="upstream-poly-fused-spatial-v6",suite=str(DEFAULT.relative_to(ROOT)),
    model_count=1200,topology_groups=779,construction_tasks=627,helper_tasks=35,rejection_tasks=48,
    manual_batches=batches,fresh_executed_runs=82,unique_model_count_not_claimed=True,
    original_fused_corpus={k:original[k] for k in ("planned","passed","failed","blocked","skipped","families","splits",
        "models_unchanged","total_actual_inner_spatial_calls","total_public_zero_eliminations")},
    unit=unit,plaintext={k:plain[k] for k in ("planned","passed","failed","skipped","tests","max_reference_error")},
    rejection={k:reject[k] for k in ("planned","passed","failed","skipped","not_run","scope")},
    old_requests_unchanged=193,old_helper_tasks_unchanged=29,current_corpus_binding_counts=mapping["accepted_nodes"],
    compiler_configuration="seal-cpu-eva-w45-v1",default_configuration_changed=False,
    tolerance="abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",security_checks_unchanged=True,
    trace_payload="compact JSON content-preserving; unchanged 131072-byte total gate",
    initial_failures_preserved=dict(public_vector_errors=5,native_passed=24,native_failed=3,
        failure_layer="dsl_trace payload whitespace exceeded byte limit",
        intermediate_unit_failure="stale 29-task count; corrected to 35 with new topology checks"),
    holdout_scope="development14 validation1 holdout1; inspected manually, not unseen Agent evaluation",
    actual_compile=True,actual_seal_cpu=True,actual_decrypt=True,paid_api_calls=0,
    installed=False,downloaded=False,commit=False,push=False,pr=False,x86_retested=False,remaining=remaining,
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
target=ROOT/"docs/baseline/upstream-fused-spatial-acceptance.json";require(not target.exists(),"Preserve summary");dump(target,summary)
n=batches["native"]
evidence=f"""原冻结16个ConvBN/DwConv模型（各8个）与11个非对称/分叉/多输入补充案例全部真实密态通过。
最终源码批次：主批27、helper定向35、分块18、旧native/public2，全部0 failed/blocked/skipped。
82是执行次数，包含重叠任务，不是82个不同模型，不叠加旧源码下的通过次数。
主批1080个解密值，最大绝对误差 {n['max_absolute_error']:.16g}，MAE {n['weighted_mae']:.16g}。
全部满足冻结门限，保留真实tracing、Earth/CKKS、HEVM/CST、SEAL和四组逐项解密证据。
82份执行产物另分三片审计；relinearization仍是融合指令/源码/输出结构证据，不冒充逐操作打点。

223单测通过、0失败/错误；3个需指定旧replay的单项测试，以及含5项的旧r18源码测试类跳过，共8项不计pass。
全1200模型/19200组双参考及48静态拒绝通过。193旧请求字节一致，旧29helper任务和模型分片保持。
本批原16例划分development14、validation1、holdout1；这个holdout图已用于人工后端开发，
不能把本批称为未接触图的Agent留出评测。原模型/权重/拓扑/划分未改。
原16例真实tracing内部调用 {original['total_actual_inner_spatial_calls']} 次，
精确公共零消去 {original['total_public_zero_eliminations']} 次；不是每组密态输入重新执行Python helper。

全表静态可绑定16个ConvBN、13个DwConv节点；只表示绑定能力，不是全量FHE成功。
源码SHA {source}。
机器摘要与原始报告hash见 [upstream-fused-spatial-acceptance.json](upstream-fused-spatial-acceptance.json)。
"""
p=ROOT/"docs/baseline/upstream-fused-spatial.md";s=p.read_text()
s=s.replace("r24 最终源码批次正在执行；通过数字待机器报告生成后补入。",evidence)
s=s.replace("semantic-v1-fused-r24-final","semantic-v1-fused-r24-verified")
s=s.replace("未完成：其余 helper、","未完成：其余7类helper（4类无bootstrap、3类bootstrap阻塞）、")
s+="\n另保留中间源码单测的目录计数失败；修正29→35并增加两类各3个拓扑检查后重新冻结验收。\n本轮无安装/下载/收费/commit/push/PR，只改Ubuntu源码；x86/GPU未重测。\n";p.write_text(s)
intro="""当前 r24：[融合 ConvBN 与深度卷积](upstream-fused-spatial.md)新增可选v6，
原16模型与11补充案例真实密态通过；35helper、18chunk、2旧入口回归通过。
默认冻结 semantic-v1-fused-r24-verified；1200模型/779拓扑/627构造不变，helper35、拒绝48。
223单测通过，8项历史证据依赖跳过；1200/19200双参考通过。不是全量密态或付费Agent成绩。
193旧请求和旧29helper任务保持；本批1个holdout图用于人工开发，不作未见图评测。
"""
for name in ("agent-current-contract.md","supported-inputs-models-configurations.md","semantic-benchmark-v1.md"):
    p=ROOT/"docs/baseline"/name;heading,rest=p.read_text().split("\n",1)
    p.write_text(heading+"\n\n"+intro+"\n"+rest.lstrip().replace("当前 r23：","历史 r23：",1))
p=ROOT/"scripts/README.md";s=p.read_text().replace("当前helper定向目录29任务。","v5阶段helper定向目录29任务。")
s+="\n统一图可选择 --unified-helpers upstream-poly-fused-spatial-v6，增加真实HE_ConvBN/HE_DwConv固定Conv→BN子图绑定。\n输入为卷积前值，输出为BN后值；旧profile保留，当前helper目录35任务。\n范围与准备入口见[融合helper说明](../docs/baseline/upstream-fused-spatial.md)。\n";p.write_text(s)
handoff=f"""## 最新接续：融合 ConvBN / DwConv r24（2026-09-22）

已确认：仅 ARM Ubuntu /home/lhohy/Code/Poseidon、lhy-agent-dsl，HEAD {summary['head']}；修改未提交。
profile upstream-poly-fused-spatial-v6；默认freeze semantic-v1-fused-r24-verified。
真实HE_ConvBN/HE_DwConv依据生产者边绑定，mean-bias等价参数+固定public maps；原模型/reference不变。
原16模型+11补充全部真实编译/SEAL/解密通过，主批1080值max {n['max_absolute_error']:.9g}。
35helper、18chunk、2旧入口通过，共82次含重复执行；82产物审计通过。
223单测通过/0失败，3单项+含5项历史类skip；1200/19200双参考、48静态拒绝通过。
193旧请求字节一致，旧29helper任务保持；1200模型/779拓扑/627构造不变，helper增至35。

源码SHA {source}；说明upstream-fused-spatial.md，机器摘要upstream-fused-spatial-acceptance.json。
有效证据results/upstream-fused-r24-verified-*；旧native保留24pass/3trace失败，
旧final保留82执行pass和目录计数单测失败。payload只去排版空白，128KiB总门禁不变。
初始5项公开测试错误来自共享派生对象，已修复；所有旧日志保留。
本批development14/validation1/holdout1，1个holdout用于人工开发，不称未接触图Agent评测。
全部批次已结束，无安装/下载/收费/commit/push/PR；Mac副本未动，x86/GPU未重测。

基于证据：所列融合helper分区具备人工密态与有限返回贡献证据，不外推全部参数或Agent能力。
未完成：HE_MPBN、HE_DS、HE_Linear、HE_ReshapeLinear、3类bootstrap阻塞；
chunk/helper联合、参数/预算边界、旧w40失败、3规则预算/2旧编译失败、最终源码全量密态/627构造、获批后的Agent。
下一实施优先真实HE_DS closure/输出映射，或MPBN/Linear固定65536槽假设的适配；先独立核查源码与参考。
下一只读命令：POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/benchmark.py helper-directed。
总体goal未完成，无需新安装或付费即可继续本地工作。

"""
p=ROOT/"docs/baseline/mac-migration-handoff.md";heading,rest=p.read_text().split("\n",1);p.write_text(heading+"\n\n"+handoff+rest.lstrip())
print(json.dumps(dict(status=summary["status"],source_sha256=source,passed_runs=82,unit_passed=223,reports=len(paths))))
