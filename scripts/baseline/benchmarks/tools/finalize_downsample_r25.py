
"""Local acceptance publication after current-source DS evidence is complete."""
import json,hashlib,subprocess
from pathlib import Path
from benchmark_graph import digest,require
from benchmark_runner import DEFAULT,load,dump
from semantic_benchmark_execution import runtime_sources
from workspace_paths import ROOT,WORK
base=WORK/"results";prefix="upstream-ds-r25-";paths=[]
def read(name):
    p=base/(prefix+name);r=json.loads(p.read_text());paths.append(p);return r
source=digest(runtime_sources());complete=read("verification-complete.json")
require(complete["status"]=="passed" and complete["source_sha256"]==source,"Incomplete current verification")
unit=read("unit-final.json");plain=read("plaintext/report.json");reject=read("rejections.json")
compat=read("compatibility.json");original=read("original-corpus-audit.json");mapping=read("static-mappings.json")
require(unit["success"] and unit["passed"]==231 and not unit["failures"] and not unit["errors"],"Unit verification")
require(plain["passed"]==1200 and plain["failed"]==0 and reject["passed"]==48 and reject["failed"]==0,"Offline results")
batches={}
for name,count in (("native",24),("helper-directed",38),("regression",2),("chunk-regression",18)):
    r=read(name+"/report.json");require(r["passed"]==count and r["failed"]==0,"Batch failed "+name)
    if "source_sha256" in r:require(r["source_sha256"]==source,"Source drift")
    batches[name]={k:r[k] for k in ("planned","passed","failed","blocked","skipped","max_absolute_error",
        "weighted_mae","compared_values","seconds") if k in r}
audits={}
for label,count in (("native",24),("helper_regression",40),("chunk_regression",18)):
    r=read(label+"-compiler-evidence.json");require(r["audited"]==count,"Compiler audit count");audits[label]=count
rows,index=load(DEFAULT);coverage=json.loads((DEFAULT/"coverage.json").read_text())
require(len(coverage["helper_directed_tasks"])==38 and compat["old_requests"]==275 and compat["old_helpers_identical"]==35,"Compatibility")
git=lambda *a:subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
remaining=["HE_MPBN","HE_Linear","HE_ReshapeLinear","HE_MaxPad/bootstrap","HE_Max/bootstrap","HE_ReLU/bootstrap",
    "chunk/helper joint layout","parameter/resource boundaries","w40 precision failure",
    "3 rule-budget and 2 historical compile failures","final-source full encrypted corpus and 627 constructions",
    "separately approved paid Agent evaluation"]
summary=dict(schema=1,phase="upstream_downsample_r25",status="phase_passed_overall_plan_incomplete",
    checkout=str(ROOT),branch=git("branch","--show-current"),head=git("rev-parse","HEAD"),
    runtime_source_sha256=source,profile="upstream-poly-downsample-v7",suite=str(DEFAULT.relative_to(ROOT)),
    model_count=1200,topology_groups=779,construction_tasks=627,helper_tasks=38,rejection_tasks=48,
    manual_batches=batches,fresh_executed_runs=82,unique_model_count_not_claimed=True,compiler_audits=audits,
    original_downsample_corpus={k:original[k] for k in ("planned","passed","failed","blocked","skipped","families","splits",
        "models_unchanged","total_actual_inner_spatial_calls","total_public_zero_eliminations")},
    unit=unit,plaintext={k:plain[k] for k in ("planned","passed","failed","skipped","tests","max_reference_error")},
    rejection={k:reject[k] for k in ("planned","passed","failed","skipped","not_run","scope")},
    old_requests_unchanged=275,old_helper_tasks_unchanged=35,current_corpus_binding_counts=mapping["accepted_nodes"],
    compiler_configuration="seal-cpu-eva-w45-v1",default_configuration_changed=False,
    tolerance="abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",security_checks_unchanged=True,
    holdout_scope="Original split preserved; any inspected holdout rows are manual backend acceptance, not unseen Agent scoring",
    actual_compile=True,actual_seal_cpu=True,actual_decrypt=True,paid_api_calls=0,
    installed=False,downloaded=False,commit=False,push=False,pr=False,x86_retested=False,remaining=remaining,
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
target=ROOT/"docs/baseline/upstream-downsample-acceptance.json";require(not target.exists(),"Preserve summary");dump(target,summary)
n=batches["native"]
evidence=f"""已确认：原冻结8个HE_DS模型和16个补充案例全部真实编译、SEAL执行、解密通过。
最终源码批次：主批24、helper定向38、分块18、旧native/public2，全部0 failed/blocked/skipped。
82是执行次数，包含重叠任务，不是82个不同模型。82份产物另分三片审计通过。
主批 {n['compared_values']} 个解密值，最大绝对误差 {n['max_absolute_error']:.16g}，
MAE {n['weighted_mae']:.16g}；全部满足冻结门限。
保留真实tracing、Earth/CKKS、HEVM/CST、SEAL、四组逐项误差与实际helper调用/返回贡献证据。
编译器relinearization证据仍限融合指令/源码/输出结构，不冒充逐操作运行时打点。

231单测通过、0失败/错误；3个需单独指定replay证据的测试与含5项的旧r18源码测试类跳过，
共8项不计pass。全1200模型/19200组双参考与48静态拒绝通过。
275旧请求字节一致，旧35helper任务、模型分片及原构造/拒绝任务不变。
原8案例划分为 {original['splits']}；不改变划分，也不把人工开发验收称作未接触图Agent成绩。
原8案例的tracing共 {original['total_actual_inner_spatial_calls']} 次实际内部HE_DS调用；
这不是每组密态输入重新运行Python helper。全表静态可绑定8个DS节点，不代表其他参数均已密态验证。
源码SHA：{source}。
机器摘要与原始报告hash见 [upstream-downsample-acceptance.json](upstream-downsample-acceptance.json)。
"""
p=ROOT/"docs/baseline/upstream-downsample.md";s=p.read_text().replace("R25_ACCEPTANCE_PENDING",evidence)
s+="\n本轮无新增安装/下载/付费/commit/push/PR；仍仅在Ubuntu源码开发，保留Mac副本与旧x86配置。\n"
p.write_text(s)
intro="""当前 r25：[真实 HE_DS 下采样](upstream-downsample.md)新增可选v7，绑定两个空间轴step-two切片。
原8模型＋16个边界/组合fixture真实密态通过；38helper、18chunk、2旧入口回归通过。
默认冻结 semantic-v1-downsample-r25；1200模型/779拓扑/627构造不变，helper38、拒绝48。
231单测通过、8项证据依赖跳过；1200/19200双参考通过。不是全量密态或付费Agent成绩。
275旧请求和旧35helper任务保持；仍有限制、未验范围及bootstrap阻塞。
"""
for name in ("agent-current-contract.md","supported-inputs-models-configurations.md","semantic-benchmark-v1.md"):
    p=ROOT/"docs/baseline"/name;heading,rest=p.read_text().split("\n",1)
    p.write_text(heading+"\n\n"+intro+"\n"+rest.lstrip().replace("当前 r24：","历史 r24：",1))
p=ROOT/"scripts/README.md";s=p.read_text().replace("当前helper目录35任务。","v6阶段helper目录35任务。")
s+="\n统一图可选择 --unified-helpers upstream-poly-downsample-v7，为两轴step=2静态slice链提供真实HE_DS调用。\n输入为两个slice之前的值，输出为第二个slice的值；旧profile保留，当前helper目录38任务。\n范围及准备入口见[下采样helper说明](../docs/baseline/upstream-downsample.md)。\n";p.write_text(s)
handoff=f"""## 最新接续：真实 HE_DS r25（2026-09-22）

已确认：仅ARM Ubuntu /home/lhohy/Code/Poseidon、lhy-agent-dsl，HEAD {summary['head']}；修改未提交。
profile upstream-poly-downsample-v7，默认freeze semantic-v1-downsample-r25。
真实HE_DS依据两个空间step2 slice的生产者链绑定，保留start/stop与轴顺序；
固定完整输入tile→原DownSelecting/Downsamp居中/复制→输出映射。原模型/reference、安全参数/密钥不变。
原8模型+16补充（含256元素、偏移、单维、多通道、多输入输出）全部真实编译/SEAL/解密通过。
主批 {n['compared_values']} 值max {n['max_absolute_error']:.9g}；38helper、18chunk、2旧入口通过，
共82次含重复执行，82产物审计通过。231单测通过/0失败；3单项+含5项历史类skip。
1200/19200双参考及48静态拒绝通过；275旧请求、旧35helper任务不变。
模型1200/拓扑779/构造627不变，helper任务增至38。原8模型划分 {original['splits']}，不声称Agent留出评测。

源码SHA {source}。说明upstream-downsample.md；摘要upstream-downsample-acceptance.json。
有效证据results/upstream-ds-r25-*，全部批次已结束；无安装/下载/付费/commit/push/PR，Mac副本未动，x86/GPU未重测。
基于证据：列出范围的HE_DS调用与返回贡献得到人工密态支持，不外推全部参数/DSL/Agent能力。
剩余3个无bootstrap helper：HE_MPBN、HE_Linear、HE_ReshapeLinear；它们硬编码nt=65536，需要真实表示适配。
另有3类bootstrap阻塞、chunk/helper联合、参数/预算边界、旧w40失败、3规则预算/2旧编译失败、
最终源码全量密态/627构造以及获批后的Agent。总体goal未完成。
下一实施：先核对MPCB.BN/Linear/Reshape的完整虚拟槽位读写，设计有界、可审计的表示，
不得直接丢弃非周期常量尾部或把65536当16384，不修改原helper数学算法。
下一只读命令：POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/benchmark.py helper-directed。

"""
p=ROOT/"docs/baseline/mac-migration-handoff.md";heading,rest=p.read_text().split("\n",1);p.write_text(heading+"\n\n"+handoff+rest.lstrip())
print(json.dumps(dict(status=summary["status"],source_sha256=source,passed_runs=82,unit_passed=231,reports=len(paths))))
