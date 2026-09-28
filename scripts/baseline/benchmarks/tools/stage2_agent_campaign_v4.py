"""Freeze all existing Stage2 Agent tasks into bounded shards; never call a provider.

Task definitions stay independent of current preparation success. No automatic
retry, generation, SDK build, credential access or historical-result promotion.
"""
import argparse,hashlib,json,os,shlex,sys
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,canonical
from benchmark_runner import strict_file,dump,load
from stage2_agent_pilot_plan import PAID,LIMITS,check_binding,sha,SUITE
INVENTORY=ROOT/"docs/baseline/stage2-agent-coverage-ledger-r101.json"
GROUPS={"free_primary":1200,"free_supplement":92,"free_gap_context":30,
        "directed_construction":627,"directed_helper":60,"directed_fixed_polynomial":21}


def identity(spec,sources):
    return digest(dict(task_id=spec["id"],model_sha256=spec["model_sha256"],
                       request_id=spec["request_id"],runtime_source_sha256=digest(sources),
                       paid_configuration=PAID))


def partition(specs,max_cases=48,max_payload_bytes=6*1024**2):
    if type(max_cases) is not int or not 1<=max_cases<=48:
        raise ValueError("Shard case budget")
    if type(max_payload_bytes) is not int or not 1<=max_payload_bytes<=6*1024**2:
        raise ValueError("Shard payload budget")
    if len({s["id"] for s in specs})!=len(specs):raise ValueError("Duplicate task")
    shards=[];current=[];size=0
    for spec in specs:
        n=len(canonical(spec))
        if n>max_payload_bytes:raise ValueError("Single prepared task exceeds shard payload")
        if current and (len(current)>=max_cases or size+n>max_payload_bytes):
            shards.append(current);current=[];size=0
        current.append(spec);size+=n
    if current:shards.append(current)
    return shards


def classify_error(error):
    # A preparation limitation never proves a mathematical/DSL/backend impossibility.
    return dict(status="request_preparation_blocked",failure_layer="request_preparation",
                diagnostic=type(error).__name__+": "+str(error)[:2000],
                agent_support_disproved=False)


def task_arguments(spec):
    args=["--compiler-configuration",spec["compiler_configuration"]]
    if spec.get("construction_profile")=="hecate-unified-public-v1":args+=["--unified-profile","public-v1"]
    if spec.get("exercise"):args+=["--unified-exercise",spec["exercise"]]
    if spec.get("helper_profile"):
        args+=["--unified-helpers",spec["helper_profile"]]
        for name in spec["required_helpers"]:args+=["--unified-helper-exercise",name]
    args+=["--unified-guidance","explicit-v3","--live","--provider",PAID["provider"],"--model",PAID["model"],
           "--reasoning-effort",PAID["reasoning_effort"],"--stream","--api-timeout",str(PAID["api_timeout_seconds"]),
           "--max-tokens",str(PAID["max_tokens"]),"--max-repairs",str(PAID["max_repairs"]),
           "--provider-retries",str(PAID["provider_retries"])]
    return args


def definitions():
    inventory=strict_file(INVENTORY,16*1024**2);check_binding(inventory)
    if inventory["binding"]!="fcf0b1109b3e0870d29323c9f9adf0addf21bcd1c7a05cef9374d80ddd19582e":
        raise ValueError("Frozen task inventory changed")
    if Counter(t["group"] for t in inventory["tasks"])!=Counter(GROUPS):raise ValueError("Task denominator changed")
    # Verify only definition files here; historical execution is not a current pass.
    parents={str(INVENTORY):sha(INVENTORY)}
    def read(path,limit=16*1024**2):
        path=Path(path)
        expected=inventory["parents"].get(str(path))
        if expected is None or sha(path)!=expected:raise ValueError("Frozen definition changed: "+str(path))
        parents[str(path)]=expected
        return strict_file(path,limit)
    read(SUITE/"index.json")
    rows,index=load(SUITE,check_sources=False)
    for shard in index["shards"]:read(SUITE/shard["file"],1024**2)
    ledger=read(SUITE/"coverage.json",4*1024**2)
    math=read(SUITE/ledger["components"]["math_contexts"]["file"])
    models={r["model"]["id"]:r["model"] for r in rows}
    models.update({r["model"]["id"]:r["model"] for r in math["supplemental_models"]})
    directed={r["id"]:r for r in ledger["directed_tasks"]}
    helpers={r["id"]:r for r in ledger["helper_directed_tasks"]}
    fixture=Path(__file__).with_name("fixed_polynomial_cases.py")
    if sha(fixture)!=inventory["parents"][str(fixture)]:raise ValueError("Fixed polynomial definitions changed")
    parents[str(fixture)]=sha(fixture)
    from fixed_polynomial_cases import cases
    fixed={x["id"]:x for x in cases()}
    specs=[]
    for task in inventory["tasks"]:
        group=task["group"]
        s=dict(id=task["id"],group=group,track="free" if group.startswith("free_") else
               "directed_helper" if group in ("directed_helper","directed_fixed_polynomial") else "directed_construction",
               origin={k:task[k] for k in ("topology","split","requirement","frozen_task_sha256","acceptance_kind") if k in task},
               compiler_configuration="seal-cpu-eva-w45-v1")
        if group=="free_gap_context":
            from model_decomposition import decompose
            original=read(task["definition"]["file"]);model,record=decompose(original)
            if record["binding_sha256"]!=task["definition"]["decomposition_binding"]:raise ValueError("Gap decomposition changed")
        elif group=="directed_helper":
            h=helpers[task["id"]];model=h["model"]
            if h["task_sha256"]!=task["frozen_task_sha256"]:raise ValueError("Helper task changed")
            s.update(helper_profile=h["profile"],required_helpers=h["required_helpers"],compiler_configuration=h["compiler_configuration"])
            if h.get("chunk_period") is not None:s["chunk_period"]=h["chunk_period"]
        elif group=="directed_fixed_polynomial":
            h=fixed[task["id"]];model=h["model"]
            s.update(helper_profile=h["profile"],required_helpers=[h["helper"]],compiler_configuration="seal-cpu-eva-w40-v1")
        else:
            model=models[task["model_id"]]
            if group=="directed_construction":
                t=directed[task["id"]]
                if t["task_sha256"]!=task["frozen_task_sha256"]:raise ValueError("Construction task changed")
                s.update(exercise=t["exercise"],construction_profile=t["profile"])
        if digest(model)!=task["model_sha256"]:raise ValueError("Model hash changed")
        s.update(model=model,model_sha256=task["model_sha256"])
        specs.append(s)
    if len({s["id"] for s in specs})!=2030:raise ValueError("Task set changed")
    return specs,parents,inventory["binding"]


def prepare_all(output):
    from hecate_python_env import VENV
    from workspace_paths import RESULTS
    from platform_config import require_python_packages,identity as platform_identity
    from semantic_benchmark_execution import runtime_sources
    from compiler_configuration import configuration,PROFILE_SHA256
    from unified_graph_contract import prepare,validate_request
    import numpy as np,torch
    require_python_packages(torch,np)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure Python required")
    if output.exists() or output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result directory required")
    sources=runtime_sources();specs,parents,inventory_binding=definitions()
    output.mkdir();ready=[];states=[]
    for s in specs:
        state=dict(id=s["id"],group=s["group"],model_sha256=s["model_sha256"],status="not_run")
        try:
            public=s.get("construction_profile")=="hecate-unified-public-v1"
            request=prepare(s["model"],PROFILE_SHA256,configuration(s["compiler_configuration"]),s.get("exercise"),
                            construction_profile=s["construction_profile"] if public else None,
                            helper_profile=s.get("helper_profile"),helper_exercise=s.get("required_helpers"),
                            chunk_period=s.get("chunk_period"),generation_guidance="explicit-v3")
            validate_request(request)
        except ValueError as e:
            state.update(classify_error(e))
        else:
            s.update(request=request,request_id=request["request_id"],candidate_arguments=task_arguments(s),status="prepared_not_run")
            if s.get("chunk_period") is not None:s["candidate_arguments"]+=["--unified-chunk-period",str(s["chunk_period"])]
            s["evaluation_identity"]=identity(s,sources);ready.append(s)
            state.update(status="prepared_not_run",request_id=s["request_id"],evaluation_identity=s["evaluation_identity"])
        states.append(state)
    shards=partition(ready);refs=[]
    names=("stage2_agent_campaign.py","stage2_agent_campaign_v4.py","run_stage2_agent_campaign_v4.py","campaign_live_state.py",
           "audit_stage2_campaign_shard_v2.py","campaign_request_files.py","campaign_candidate_failures.py",
           "stage2_agent_pilot_plan.py","stage2_guidance_retest_plan.py",
           "run_stage2_guidance_retest.py","stage2_agent_provenance.py","audit_stage2_live_candidate.py")
    proposal={str(Path(__file__).with_name(n).relative_to(ROOT)):sha(Path(__file__).with_name(n)) for n in names}
    for n,cases_ in enumerate(shards):
        plan=dict(format="poseidon-stage2-agent-campaign-shard-v1",generation_guidance="explicit-v3",source_hashes=sources,proposal_files=proposal,
                  definition_parents=parents,inventory_binding=inventory_binding,platform=platform_identity(),
                  shard_index=n,cases=cases_,paid_configuration=PAID,
                  limits=dict(LIMITS,maximum_generations=4*len(cases_),maximum_http_attempts=16*len(cases_)),
                  paid_calls=0,approved=False,executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
                  scope="Offline frozen requests only. No dispatch or reuse of historical pass status.",
                  configuration_policy="Free and construction: existing w45. Helpers: frozen per-task configuration. Fixed polynomials: existing w40.")
        plan["binding"]=digest(plan)
        from run_stage2_agent_campaign_v4 import validate_plan
        validate_plan(plan,plan["binding"])
        if len(canonical(plan))>8*1024**2:raise ValueError("Shard plan file budget")
        p=output/("shard-%03d.json"%n);dump(p,plan)
        refs.append(dict(file=p.name,sha256=sha(p),binding=plan["binding"],case_ids=[s["id"] for s in cases_]))
    if runtime_sources()!=sources:raise ValueError("Runtime drift while preparing")
    for name,hsh in parents.items():
        if sha(Path(name))!=hsh:raise ValueError("Definition drift while preparing")
    inventory=strict_file(INVENTORY,16*1024**2)
    relations=[dict(id=x["id"],layer=x["layer"],scope=x["scope"],backend_blocker=x["backend_blocker"],
                    required_distinct_contexts=x["required_distinct_contexts"],
                    contexts=[dict(task_id=c["task_id"],topology=c["topology"]) for c in x["contexts"]],
                    current_status="backend_blocked" if x["agent_status"]=="backend_blocked" else
                    "static_controls_separate" if x["layer"]=="rejection" else "not_evaluated")
               for x in inventory["partitions"]]
    assert len(relations)==401
    index=dict(format="poseidon-stage2-agent-campaign-index-v1",inventory_binding=inventory_binding,
               runtime_source_sha256=digest(sources),definition_parents=parents,states=states,shards=refs,
               semantic_partitions=relations,bootstrap_blocked_partitions=[x["id"] for x in relations if x["current_status"]=="backend_blocked"],
               counts=dict(Counter(x["status"] for x in states)),planned=len(specs),paid_calls=0,
               agent_evaluation_complete=False,stage2_complete=False,
               limits="Each shard <=48 tasks and <=8MiB; execution not started or authorized by this file.",
               historical_passes_promoted=0)
    index["binding"]=digest(index);dump(output/"index.json",index)
    print(json.dumps({k:index[k] for k in ("binding","runtime_source_sha256","planned","counts","paid_calls")}))
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    from hecate_python_env import VENV,enter_nix
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
    prepare_all(a.output);return 0
if __name__=="__main__":raise SystemExit(main())
