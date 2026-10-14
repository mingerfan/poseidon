"""Read-only failure triage and one explicitly authorized targeted retry plan.

Historical outcomes are immutable. A retry is one new evaluation, never proof
that the prior failure did not happen. No credentials are accessed here.
"""
import argparse, collections, json, struct, sys
from pathlib import Path
from stage2_agent_campaign import ROOT, BASE, PAID, LIMITS, identity
from stage2_agent_pilot_plan import check_binding, sha
from benchmark_graph import digest
from benchmark_runner import strict_file
from campaign_live_state import sealed, exclusive_json
from semantic_benchmark_execution import runtime_sources

USER_AUTHORIZATION = "那么对第二阶段失败的进行针对性的检查，如果可以重试解决的直接重新agent的测试，对于不能的记录原因和修改方向"
RUNTIME = "f6493518f4bf63c257fb7eab9f9a782a89cf5cf4a16c9006f607976f2c58b69b"
RETRY_COUNT = 257
REVIEW = ROOT/"docs/baseline/stage2-targeted-failure-review-r153.json"
AUTH_NAME = "stage2-targeted-authorization-r153.json"
QUEUE_NAME = "stage2-targeted-retry-r153"
FORBIDDEN_RETRY_LAYERS = {"compiler","artifact_gate","environment","harness_launch","key_setup","integrity","seal_runtime"}
NETWORK_CODES = {"transport_tls_failed","transport_dns_failed","transport_invalid_stream"}

def classify(evaluations):
    if any(e["status"] == "passed" for e in evaluations):
        return "already_recovered", "已有单列成功证据；保留旧失败，不重复付费", "不重跑，不合并版本成功率"
    reports=[e["report"] for e in evaluations]
    attempts=[a for r in reports for a in r.get("attempts",[])]
    layers={a.get("failure_layer") for a in attempts}
    if layers & FORBIDDEN_RETRY_LAYERS:
        if layers & {"environment","harness_launch","key_setup","integrity","seal_runtime"}:
            return "hold_pipeline", "执行/环境/完整性故障，不能由随机重生成掩盖", "先复现并修复对应执行层；未知运行错误保留"
        return "hold_pipeline", "修复轨迹含编译或产物故障，不能仅按最后静态/服务错误决定重跑", "核对乘法容量、scale/level调度及产物门禁；仅对已证明合法的等价表达再评测"
    report=reports[-1]
    if report["status"] == "provider_failed":
        call=report["provider_metrics"]["calls"][-1]
        code=call.get("error")
        if code in NETWORK_CODES:
            return "retry_provider", "已结束的传输故障，可作一轮有界恢复；历史计费不确定性保留", "若再现，按TLS/DNS/流解析分别诊断；不关闭TLS、不无限重试"
        if code == "empty_or_oversized_content" and call.get("content_characters")==0 and call.get("finish_reason")=="stop":
            return "retry_provider", "元数据确认最终content为空，并非候选过大；未取得可评测答案", "有界重新生成；再现时诊断最终答案/流协议，不将reasoning内容当候选、不放宽响应检查"
        return "hold_provider", "未知或非暂态provider故障，尚不足以判断可重试", "先检查安全元数据、响应边界和权限；不打印凭据"
    layer=attempts[-1].get("failure_layer") if attempts else None
    if layer=="static_check":
        return "retry_static_check", "未发现编译/环境阻塞；候选违反有限构造契约或缺少有效贡献", "采用已有explicit-v4构造说明再评测；仍失败时最小正反例核对说明与检查器，不放宽契约"
    if layer=="numerical_comparison":
        return "retry_numerical_comparison", "已真实执行但数学/布局错误；不属于已证明的CKKS精度极限", "重新生成并保留原误差门限；若失败，补rotation、补零、归约、拼接和多项式回归"
    return "hold_unknown", "未知失败层，禁止猜测后启动付费", "先补齐失败证据及分类"

def artifact_metadata(folder, attempts, parents):
    found=[]
    for a in attempts:
        if a.get("failure_layer")!="artifact_gate": continue
        for p in (folder/("attempt-%02d"%a["index"])/"output").glob("*.hevm"):
            parents[str(p)]=sha(p)
            data=p.read_bytes()
            if len(data)<64: continue
            magic,header,ni,no=struct.unpack_from("<IIQQ",data)
            if magic!=0x4845564D or header!=24 or ni>5 or no>16: continue
            offset=64; fields={}
            for name,n in (("arg_scale",ni),("arg_level",ni),("res_scale",no),("res_level",no),("res_dst",no)):
                fields[name]=list(struct.unpack_from("<%dQ"%n,data,offset));offset+=8*n
            bad=[]
            for kind in ("arg","res"):
                for i,(level,scale) in enumerate(zip(fields[kind+"_level"],fields[kind+"_scale"])):
                    if not (1<=level<=13 and 1<=scale<min(180,60*level)):
                        bad.append(dict(kind=kind,index=i,level=level,scale=scale,exclusive_scale_upper=min(180,60*level)))
            found.append(dict(file=str(p),sha256=sha(p),fields=fields,invalid_header_pairs=bad))
    return found

def review():
    from workspace_paths import RESULTS
    parents={}
    def read(path, expected=None, binding=True):
        path=Path(path)
        if path.is_symlink() or path.name==".env": raise ValueError("Unsafe evidence path")
        h=sha(path)
        if expected and h!=expected: raise ValueError("Evidence changed: "+str(path))
        value=strict_file(path,32*1024**2)
        if binding: check_binding(value)
        parents[str(path)]=h
        return value
    journal=read(RESULTS/"stage2-evaluation-journal-r152.json")
    if journal["binding"]!="402092cd1ff0a5ba57cd81106f55dfad775c42cba2e4b31afc979addc3b7c614": raise ValueError("Journal identity")
    for p,h in journal["parents"].items(): read(p,h)
    rows=[]
    for t in journal["rows"]:
        if not any(e["status"]=="failed" for e in t["evaluations"]): continue
        evaluations=[]; metadata=[]
        for e in t["evaluations"]:
            folder=Path(e["evidence"])
            rep=read(folder/"report.json",e["report_sha256"],False)
            if rep["status"]=="running": raise ValueError("Uncertain prior execution")
            req=read(folder/"request.json",binding=False)
            model=read(folder/"model.json",binding=False)
            if req["request_id"]!=e["request_id"] or digest(model)!=t["model_sha256"]: raise ValueError("Prior model/request identity")
            evaluations.append(dict(e,report=rep))
            metadata+=artifact_metadata(folder,rep.get("attempts",[]),parents)
        action,reason,direction=classify(evaluations)
        last=evaluations[-1]["report"]
        last_layer="provider" if last["status"]=="provider_failed" else last.get("attempts",[{}])[-1].get("failure_layer")
        records=[]
        for e in evaluations:
            rep=e["report"]; calls=rep.get("provider_metrics",{}).get("calls",[])
            records.append({**{k:v for k,v in e.items() if k!="report"},
                "attempts":[{k:a.get(k) for k in ("index","status","failure_layer","diagnostic","compiled","executed")} for a in rep.get("attempts",[])],
                "provider_terminal":{k:calls[-1].get(k) for k in ("status","error","content_characters","reasoning_content_characters","finish_reason","usage")} if calls else None})
        rows.append(dict(id=t["id"],group=t["group"],model_sha256=t["model_sha256"],action=action,reason=reason,
            modification_direction=direction,terminal_failure_layer=last_layer,evaluations=records,
            artifact_metadata=metadata,impossibility_proven=False))
    counts=dict(collections.Counter(r["action"] for r in rows))
    expected={"retry_static_check":152,"retry_provider":94,"retry_numerical_comparison":11,"hold_pipeline":33,"already_recovered":4}
    if counts!=expected or len(rows)!=294: raise ValueError("Unexpected scope: "+str(counts))
    sources=runtime_sources()
    if digest(sources)!=RUNTIME: raise ValueError("Current runtime drift")
    return sealed(dict(format="poseidon-targeted-failure-review-r153",parents=parents,source_hashes=sources,
        runtime_source_sha256=RUNTIME,rows=rows,counts=counts,reviewed_task_ids=294,planned_retry_tasks=RETRY_COUNT,
        historical_scores_unchanged=True,classified_retry_is_not_guaranteed_success=True,
        new_paid_calls=0,new_encrypted_executions=0,
        provider_documentation=dict(url="https://api-docs.deepseek.com/quick_start/pricing/",checked_date="2026-09-27",
            model="deepseek-flash",advertised_backend="DeepSeek-V4.1-Flash",max_output="384K",
            limitation="Alias is not an immutable model version; compare separate cohorts.")))

def authorize():
    from workspace_paths import RESULTS
    if REVIEW.exists(): raise ValueError("Review exists; do not overwrite")
    r=review();exclusive_json(REVIEW,r)
    a=sealed(dict(format="poseidon-targeted-user-authorization-r153",user_instruction=USER_AUTHORIZATION,
        interpretation="One bounded retry evaluation for the diagnosed recoverable failed cases. No repeated campaign, installation or parameter change.",
        review=str(REVIEW),review_binding=r["binding"],review_sha256=sha(REVIEW),
        task_ids=[x["id"] for x in r["rows"] if x["action"].startswith("retry_")],
        runtime_source_sha256=RUNTIME,maximum_generations=1028,maximum_http_attempts=4112,
        max_repairs=3,provider_retries=3,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,
        per_shard_retained_mib=4096,total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200,
        no_explicit_currency_cap=True,duplicate_billing_possible=True,old_authorizations_not_reused=True))
    exclusive_json(RESULTS/AUTH_NAME,a)
    return r,a

def document(path):
    p=Path(path)
    if p.is_symlink() or p.name==".env": raise ValueError("Unsafe document")
    value=strict_file(p,32*1024**2);check_binding(value);return value

def approved(proposal, authorization):
    from workspace_paths import RESULTS
    if Path(proposal)!=REVIEW or Path(authorization)!=RESULTS/AUTH_NAME: raise ValueError("Wrong authorization path")
    p=document(proposal);a=document(authorization)
    if p["binding"]!="77c78805e8431ba8cf3fcf0364df15e125d0e4ef11c9034acbd9338970d8f177" or a["binding"]!="ecf4852e6000eaeb81747341a5427283b7ef2fba03b9b275287c721e1ed1f0e9": raise ValueError("Unapproved retry binding")
    if a["user_instruction"]!=USER_AUTHORIZATION or a["review_binding"]!=p["binding"] or a["review_sha256"]!=sha(REVIEW): raise ValueError("Authorization identity")
    if runtime_sources()!=p["source_hashes"] or digest(p["source_hashes"])!=RUNTIME: raise ValueError("Runtime drift")
    ids=[r["id"] for r in p["rows"] if r["action"].startswith("retry_")]
    if len(ids)!=RETRY_COUNT or len(set(ids))!=RETRY_COUNT or a["task_ids"]!=ids: raise ValueError("Retry scope")
    for k,v in dict(maximum_generations=1028,maximum_http_attempts=4112,max_repairs=3,provider_retries=3,
        api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,per_shard_retained_mib=4096,
        total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200).items():
        if a[k]!=v: raise ValueError("Authorization budget changed")
    for f,h in p["parents"].items():
        if sha(Path(f))!=h: raise ValueError("Prior evidence changed")
    return p,a

def tools_sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob("*.py")}

def build(proposal,authorization,output):
    from workspace_paths import RESULTS
    from hecate_python_env import VENV
    from unified_graph_contract import validate_request,prepare
    p,a=approved(proposal,authorization)
    if output.is_symlink() or output!=RESULTS/QUEUE_NAME: raise ValueError("Exact fresh queue path required")
    index_path=RESULTS/"stage2-agent-campaign-v7-r130/index.json"
    idx=document(index_path); original={}
    for ref in idx["shards"]:
        path=index_path.parent/ref["file"]
        if Path(ref["file"]).name!=ref["file"] or sha(path)!=ref["sha256"]: raise ValueError("Historical plan hash")
        for s in document(path)["cases"]: original[s["id"]]=s
    sources=runtime_sources();proposals=tools_sources()
    probe_path=RESULTS/"stage2-recovery-native-probe-r147/report.json";probe=document(probe_path)
    if probe["source_hashes"]!=sources or probe["passed"]!=2 or not probe["two_native_slots_observed"] or probe["min_available_mib"]<1024 or probe["new_paid_calls"]!=0: raise ValueError("Concurrent native proof")
    if probe["runner_sha256"]!=sha(Path(__file__).with_name("probe_recovery_native.py")): raise ValueError("Probe runner drift")
    for f,h in probe["parents"].items():
        if sha(Path(f))!=h: raise ValueError("Native evidence drift")
    specs=[]
    for row in sorted((x for x in p["rows"] if x["action"].startswith("retry_")),key=lambda x:(x["action"]!="retry_provider",x["id"])):
        s=dict(original[row["id"]]);old=s["request"]
        guidance="explicit-v4" if s["group"]=="directed_construction" else "explicit-v3"
        if s["model_sha256"]!=row["model_sha256"]: raise ValueError("Frozen model mismatch")
        r=prepare(s["model"],old["compiler_profile_sha256"],old["compiler_configuration"],s.get("exercise"),
            construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"),
            helper_profile=s.get("helper_profile"),helper_exercise=s.get("required_helpers"),chunk_period=s.get("chunk_period"),
            generation_guidance=guidance)
        validate_request(r)
        if guidance=="explicit-v3" and r!=old: raise ValueError("Free/helper request changed")
        args=list(s["candidate_arguments"]);pos=args.index("--unified-guidance");args[pos+1]=guidance
        s.update(request=r,request_id=r["request_id"],candidate_arguments=args,
            original_evaluation_identity=s["evaluation_identity"],original_status="failed",
            retry_reason=row["reason"],retry_action=row["action"],prior_evaluations=row["evaluations"])
        s["evaluation_identity"]=identity(s,sources);specs.append(s)
    shards=[]
    for i in range(10):
        cases=specs[i::10];n=len(cases)
        if not 1<=n<=48: raise ValueError("Shard task bound")
        shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=cases,source_hashes=sources,
            proposal_files=proposals,paid_configuration=PAID,
            limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=4*n,maximum_http_attempts=16*n),
            authorization_binding=a["binding"],targeted_review_binding=p["binding"],
            executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
            shared_scheduling=dict(api_workers=10,native_workers=2,version=1),old_claims_preserved=True,
            automatic_retry=False,stage2_complete=False))
        if len(json.dumps(shard).encode())>8*1024**2: raise ValueError("Shard bytes")
        shards.append(dict(plan=shard,output=str(RESULTS/(output.name+"-shard-%02d"%i)),audit=str(RESULTS/(output.name+"-shard-%02d-audit"%i))))
    return sealed(dict(format="poseidon-targeted-retry-queue-r153",proposal=str(proposal),authorization=str(authorization),
        proposal_sha256=sha(proposal),authorization_sha256=sha(authorization),source_hashes=sources,proposal_files=proposals,
        native_probe=str(probe_path),native_probe_sha256=sha(probe_path),prior_seconds=0,
        claim_registry=str(RESULTS/"stage2-targeted-retry-claims-r153"),authorization_binding=a["binding"],
        shards=shards,planned=RETRY_COUNT,maximum_generations=1028,maximum_http_attempts=4112,
        max_wall_seconds=43200,max_retained_mib=32768,min_free_mib=4096,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,
        admission_available_mib=3072,stop_available_mib=512,automatic_restart=False,stage2_complete=False))

def verify(plan,output):
    check_binding(plan)
    if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output): raise ValueError("Frozen targeted plan changed")
    return plan

if __name__=="__main__":
    r,a=authorize()
    print(json.dumps(dict(review_binding=r["binding"],counts=r["counts"],authorization_binding=a["binding"],paid_calls=0)))
