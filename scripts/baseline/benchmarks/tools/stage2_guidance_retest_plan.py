"""Frozen six-failure retest proposal after r99. Preparation never calls a provider."""
import argparse,copy,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,canonical
from benchmark_runner import strict_file,dump
from stage2_agent_pilot_plan import check_binding,sha,PAID
IDS=("free_bench_composition_0004","free_bench_helper_0047","construct_000_0",
     "construct_057_0","construct_134_0","fixed_Poly_Default_2")

def build_plan():
    from semantic_benchmark_execution import runtime_sources
    from unified_graph_contract import prepare,validate_request,EXPLICIT_GUIDANCE
    from platform_config import identity,require_python_packages
    import numpy as np,torch
    require_python_packages(torch,np)
    paths={k:ROOT/("docs/baseline/"+v) for k,v in (
      ("pilot","stage2-agent-pilot-r97.json"),("audit","stage2-agent-pilot-results-r98.json"),
      ("guidance","stage2-guidance-acceptance-r99.json"))}
    parent={k:strict_file(p,8*1024**2) for k,p in paths.items()}
    for value in parent.values():check_binding(value)
    assert parent["pilot"]["binding"]=="886c8e779247b5e388d93d2063512218cbf4dd92aa36b617aff30343e1f1cde0"
    assert parent["audit"]["binding"]=="c93338364edc094c47937caffddc6a986eba9151d117c73cc349167c79a1090d"
    assert parent["guidance"]["binding"]=="df766da177c4c0d748763bf3bafc01d12637b101b21ec1f0cea49c8ad6a4f939"
    assert {r["id"] for r in parent["audit"]["rows"] if r["status"]=="failed"}==set(IDS)
    sources=runtime_sources()
    if digest(sources)!=parent["guidance"]["source_after"]:
        raise ValueError("Source drift since scoped r99 acceptance")
    for name,hsh in parent["guidance"]["evidence"].items():
        if sha(Path(name))!=hsh:raise ValueError("r99 evidence changed")
    assert parent["guidance"]["tests"]==dict(passed=23,failed=0,skipped=0,
       initial_test_failure=parent["guidance"]["tests"]["initial_test_failure"])
    assert parent["guidance"]["offline_execution"]["passed"]==3
    cases=[]
    for old in parent["pilot"]["cases"]:
        if old["id"] not in IDS:continue
        item=copy.deepcopy(old);r=old["request"]
        request=prepare(r["model"],r["compiler_profile_sha256"],r.get("compiler_configuration"),
            r.get("construction_exercise",{}).get("id"),
            construction_profile=r.get("construction_profile"),constant_policy=r["constant_origins"].get("policy"),
            helper_profile=r.get("upstream_helpers",{}).get("profile"),
            helper_exercise=r.get("upstream_exercise",{}).get("required_helpers"),
            generation_guidance=EXPLICIT_GUIDANCE)
        validate_request(request)
        assert canonical({k:v for k,v in request.items() if k not in ("request_id","generation_guidance")})==canonical({k:v for k,v in r.items() if k!="request_id"})
        item.update(request=request,request_id=request["request_id"],request_bytes=len(canonical(request)),
                    previous_request_id=r["request_id"],previous_status="failed",status="prepared_not_run")
        item["candidate_arguments"]+=["--unified-guidance",EXPLICIT_GUIDANCE]
        cases.append(item)
    plan=copy.deepcopy(parent["pilot"])
    plan.update(format="poseidon-stage2-guidance-retest-v1",cases=cases,source_hashes=sources,
        platform=identity(),approved=False,paid_calls=0,
        batch_entrypoint="scripts/baseline/benchmarks/tools/run_stage2_guidance_retest.py",
        limits=dict(parent["pilot"]["limits"],maximum_generations=24,maximum_http_attempts=96),
        selection={"basis":"All six r98 failures retained; no passing cases repeated or substitutes",
                   "case_ids":list(IDS)},
        parents={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p),binding=parent[k]["binding"])
                 for k,p in paths.items()},
        limitations=parent["pilot"]["limitations"]+[
            "r99 is scoped offline acceptance; no evidence that guidance improves live Agent success yet.",
            "Retest results remain a separate version; do not overwrite r97 failures or combine attempts as new models."])
    names=[
      "stage2_guidance_retest_plan.py","run_stage2_guidance_retest.py","audit_stage2_guidance_retest.py",
      "stage2_agent_pilot_plan.py","stage2_agent_provenance.py","audit_stage2_live_candidate.py"]
    plan["proposal_files"]={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n) for n in names}
    plan.pop("binding",None);plan["binding"]=digest(plan)
    return plan

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve existing proposal; use a new path")
    plan=build_plan();dump(a.output,plan)
    print(json.dumps({"binding":plan["binding"],"cases":len(plan["cases"]),"paid_calls":0}))
