"""Freeze the approved 27-case operator batch; no provider calls in preparation."""
import json,re
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,LIMITS,identity
from stage2_agent_pilot_plan import PAID,sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed
from benchmark_runner import strict_file
def document(path):
    value=strict_file(path,32*1024**2)
    return value
from semantic_benchmark_execution import runtime_sources
NAME="stage3-extrema-live-r231"
REVIEW=ROOT/"docs/baseline/stage3-extrema-live-proposal-r230.json"
AUTH_NAME="stage3-extrema-authorization-r231.json"
def tools_sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob("*.py")}
def build(proposal,auth_path,output):
    from workspace_paths import RESULTS as R
    from hecate_python_env import VENV
    from seal_cpu_golden import KEY_BUILD,BUILD
    from component_contract import reconstruct_request,runner_options
    from deepseek_provider import public_request,Config
    from unified_graph_contract import prepare
    from compiler_configuration import PROFILE_SHA256,configuration
    if proposal!=REVIEW or auth_path!=R/AUTH_NAME or output!=R/NAME:raise ValueError("Exact fresh paths required")
    p=document(proposal);auth=document(auth_path);check_binding(p);check_binding(auth)
    if not auth.get("approved") or auth["proposal_binding"]!=p["binding"] or auth["proposal_sha256"]!=sha(proposal):
        raise ValueError("Approval mismatch")
    if p["maximum_generations"]!=108 or p["maximum_http_attempts"]!=432 or len(p["cases"])!=27:
        raise ValueError("Approved scope")
    expected=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
                  max_repairs=3,provider_retries=3,max_tokens=384000,api_timeout_seconds=1200)
    if PAID!=expected:raise ValueError("Provider configuration drift")
    Config(model=PAID["model"],service_provider=PAID["provider"],reasoning_effort=PAID["reasoning_effort"],
           max_calls=4,max_tokens=384000,timeout_seconds=1200,stream=True,provider_retries=3)
    sources=runtime_sources()
    if sources!=p["source_hashes"]:raise ValueError("Approved runtime source drift")
    unit=document(R/"stage3-extensions-r228/unit.json")
    compat=document(R/"stage3-compatibility-r230/unit-r215.json")
    if unit["tests"]!=74 or any(unit[k] for k in ("failed","errors","skipped")):raise ValueError("Unit gate")
    if compat["run"]!=249 or any(compat[k] for k in ("failures","errors","skipped")):raise ValueError("Compatibility gate")
    if compat["source_hashes"]!=sources:raise ValueError("Compatibility binding drift")
    if document(R/"stage3-compatibility-r230/compatibility-r215.json")["old_requests_exact"]!=2208:
        raise ValueError("Legacy request compatibility")
    manifest=ROOT/"scripts/baseline/cases/operator-extension-v4/manifest.json"
    if sha(manifest)!=p["manifest_sha256"]:raise ValueError("Model manifest changed")
    proof=document(R/"stage3-extrema-offline-r229/report.json")
    proved={x["id"]:x for x in proof["rows"] if x["status"]=="passed"}
    if set(proved)!={c["id"] for c in p["cases"]}:raise ValueError("Offline population mismatch")
    for c in p["cases"]:
        row=proved[c["id"]]
        if row["model_sha256"]!=c["model_sha256"] or row["source_sha256"]!=c["source_sha256"]:
            raise ValueError("Offline model identity")
        rp=Path(row["evidence"])/"report.json"
        if sha(rp)!=row["report_sha256"]:raise ValueError("Offline proof changed")
        rr=document(rp)
        if rr["status"]!="passed" or not all(rr["attempts"][-1][k] for k in ("compiled","executed","numerically_correct")):
            raise ValueError("Offline numerical proof")
    specs=[]
    for c in p["cases"]:
        model=document(ROOT/c["model_path"])
        req=prepare(model,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),generation_guidance="explicit-v9")
        if digest(model)!=c["model_sha256"] or req["request_id"]!=c["request_id"]:raise ValueError("Request identity")
        if reconstruct_request(req)!=req or public_request(req)!=req:raise ValueError("Public request changed")
        args=[*runner_options(req),"--live","--provider","deepseek","--model","deepseek-flash",
              "--reasoning-effort","high","--stream","--api-timeout","1200","--max-tokens","384000",
              "--max-repairs","3","--provider-retries","3"]
        s=dict(id=c["id"],track="free",model=model,model_sha256=c["model_sha256"],request=req,
               request_id=req["request_id"],candidate_arguments=args,max_repairs=3,new_generation=True)
        s["evaluation_identity"]=identity(s,sources);specs.append(s)
    ts=tools_sources();shards=[]
    for index in range(4):
        items=specs[index::4]
        shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=items,source_hashes=sources,
            proposal_files=ts,paid_configuration=PAID,
            limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,
                        maximum_generations=len(items)*4,maximum_http_attempts=len(items)*16),
            executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
            shared_scheduling=dict(api_workers=4,native_workers=1,version=1),
            authorization_binding=auth["binding"],automatic_retry=False))
        shards.append(dict(plan=shard,output=str(R/(NAME+"-shard-%02d"%index)),
                           audit=str(R/(NAME+"-shard-%02d-audit"%index))))
    guard={str(BUILD/f):sha(BUILD/f) for f in ("bin/hecate-opt","lib/libSEAL_HEVM.so","lib/libHecateFrontend.so")}
    checker=KEY_BUILD/"libseal_artifact_parameters.so"
    return sealed(dict(format="poseidon-extrema-queue-r231",proposal=str(proposal),authorization=str(auth_path),
        proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=ts,
        compiler_guard=guard,checker_guard={str(checker):sha(checker)},native_probe=str(rp),native_probe_sha256=sha(rp),
        prior_seconds=0,prior_generations=0,prior_http_attempts=0,deadline_epoch=auth["authorized_at"]+43200,
        old_batches_not_rewritten=True,claim_registry=str(R/"stage3-extrema-claims-r231"),shards=shards,
        planned=27,maximum_generations=108,maximum_http_attempts=432,max_wall_seconds=43200,
        max_retained_mib=32768,min_free_mib=4096,api_workers=4,native_workers=1,compile_jobs=2,link_jobs=1,
        automatic_restart=False))
def verify(plan,output):
    check_binding(plan)
    if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen execution plan changed")
