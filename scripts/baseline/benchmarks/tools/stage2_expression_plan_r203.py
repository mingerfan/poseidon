"""Fresh explicitly approved one-case retest, separate from all prior quotas."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,LIMITS
from stage2_agent_pilot_plan import PAID,sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources as base_tools_sources
NAME="stage2-expression-live-r203"
REVIEW=ROOT/"docs/baseline/stage2-expression-live-proposal-r203.json"
AUTH_NAME="stage2-expression-authorization-r203.json"
def tools_sources():
    return dict(base_tools_sources(),**{f:sha(ROOT/f) for f in (
        "scripts/stage2_expression_job_r203.py","scripts/stage2_expression_report_r204.py",
        "scripts/baseline/seal_keys/artifact_parameters.cpp","scripts/baseline/seal_keys/artifact_scale_probe.cpp",
        "scripts/baseline/seal_keys/CMakeLists.txt")})
def build(proposal,auth_path,output):
    from workspace_paths import RESULTS as R
    from hecate_python_env import VENV
    from seal_cpu_golden import KEY_BUILD
    from component_contract import reconstruct_request,runner_options
    from deepseek_provider import public_request
    if proposal!=REVIEW or auth_path!=R/AUTH_NAME or output!=R/NAME:raise ValueError("Exact fresh retest paths")
    p=document(proposal);auth=document(auth_path)
    if (auth.get("approved") is not True or auth["proposal_binding"]!=p["binding"] or
        auth["proposal_sha256"]!=sha(proposal) or auth["maximum_generations"]!=4 or auth["maximum_http_attempts"]!=16):
        raise ValueError("Explicit one-case authorization mismatch")
    if p["maximum_generations"]!=4 or p["maximum_http_attempts"]!=16 or len(p["cases"])!=1:raise ValueError("Scope")
    expected_provider=dict(PAID);expected_provider["name"]=expected_provider.pop("provider")
    if p["provider"]!=expected_provider:raise ValueError("Provider configuration changed")
    sources=runtime_sources()
    if sources!=p["source_hashes"]:raise ValueError("Approved runtime source drift")
    for f,h in p["parents"].items():
        if sha(Path(f))!=h:raise ValueError("Offline proof changed")
    unit=json.loads((R/"stage2-expression-repairs-r202/unit.json").read_text())
    if unit["run"]!=184 or any(unit[k] for k in ("failures","errors","skipped")):raise ValueError("Offline unit gate")
    c=p["cases"][0];q=c["request"]
    if c["id"]!="free_bench_helper_0113" or c["model_id"]!="bench_helper_0113":raise ValueError("Case changed")
    if reconstruct_request(q)!=q or public_request(q)!=q or digest(q["model"])!=c["model_sha256"]:raise ValueError("Request/model changed")
    args=[*runner_options(q),"--live","--provider","deepseek","--model","deepseek-flash",
          "--reasoning-effort","high","--stream","--api-timeout","1200","--max-tokens","384000",
          "--max-repairs","3","--provider-retries","3"]
    spec=dict(id=c["id"],track="free",model=q["model"],model_sha256=c["model_sha256"],request=q,
              request_id=q["request_id"],candidate_arguments=args,max_repairs=3,new_generation=True)
    spec["evaluation_identity"]=digest(dict(task_id=spec["id"],model_sha256=spec["model_sha256"],
        request_id=spec["request_id"],runtime_source_sha256=digest(sources),paid_configuration=PAID))
    guard=json.loads((R/"stage2-expression-repairs-r199/before.json").read_text())["compiler"]
    if len(guard)!=93 or not all(sha(Path(f))==h for f,h in guard.items()):raise ValueError("Compiler drift")
    ts=tools_sources()
    shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=[spec],source_hashes=sources,
        proposal_files=ts,paid_configuration=PAID,
        limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=4,maximum_http_attempts=16),
        executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
        shared_scheduling=dict(api_workers=1,native_workers=1,version=1),
        authorization_binding=auth["binding"],automatic_retry=False))
    checker=KEY_BUILD/"libseal_artifact_parameters.so";probe=R/"stage2-expression-repairs-r201/offline/report.json"
    return sealed(dict(format="poseidon-expression-retest-r203",proposal=str(proposal),authorization=str(auth_path),
        proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),
        source_hashes=sources,proposal_files=ts,compiler_guard=guard,
        checker_guard={str(checker):sha(checker)},native_probe=str(probe),native_probe_sha256=sha(probe),
        prior_seconds=0,prior_generations=0,prior_http_attempts=0,deadline_epoch=auth["authorized_at"]+43200,
        old_batches_not_rewritten=True,claim_registry=str(R/"stage2-expression-claims-r203"),
        shards=[dict(plan=shard,output=str(R/(NAME+"-shard-00")),audit=str(R/(NAME+"-shard-00-audit")))],
        planned=1,maximum_generations=4,maximum_http_attempts=16,max_wall_seconds=43200,max_retained_mib=32768,
        min_free_mib=4096,api_workers=1,native_workers=1,compile_jobs=2,link_jobs=1,automatic_restart=False))
def verify(plan,output):
    check_binding(plan)
    if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen retest changed")
