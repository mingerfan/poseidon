"""Nine existing composite tasks, explicit-v2/w45. Offline preparation only."""
import copy,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,canonical
from benchmark_runner import load,strict_file
from stage2_agent_pilot_plan import check_binding,sha,PAID,LIMITS,SUITE

def build_plan():
 from semantic_benchmark_execution import runtime_sources
 from compiler_configuration import configuration,PROFILE_SHA256
 from unified_graph_contract import prepare,validate_request
 from platform_config import identity,require_python_packages
 from hecate_python_env import VENV
 import numpy as np,torch
 require_python_packages(torch,np)
 accepted_path=ROOT/"docs/baseline/stage2-composite-guidance-r104.json"
 accepted=strict_file(accepted_path,8*1024**2);check_binding(accepted)
 sources=runtime_sources()
 assert digest(sources)==accepted["source_after"]
 assert accepted["w45_scoped_offline_acceptance"] and accepted["tests"]["passed"]==14
 for name,hsh in accepted["evidence"].items():assert sha(Path(name))==hsh
 rows,index=load(SUITE,check_sources=False);models={r["model"]["id"]:r for r in rows}
 ledger=strict_file(SUITE/"coverage.json",4*1024**2)
 required=("array.arithmetic","array.mutation","array.view_copy")
 tasks=[t for t in ledger["directed_tasks"] if t["requirement"] in required]
 assert len(tasks)==9
 for f in required:assert len({models[t["model_id"]]["topology"] for t in tasks if t["requirement"]==f})==3
 cases=[]
 for t in tasks:
  m=models[t["model_id"]];assert t["profile"]=="hecate-unified-public-v1"
  config=configuration("seal-cpu-eva-w45-v1")
  request=prepare(m["model"],PROFILE_SHA256,config,t["exercise"],construction_profile=t["profile"],generation_guidance="explicit-v2")
  validate_request(request)
  args=["--compiler-configuration",config["name"],"--unified-profile","public-v1","--unified-exercise",t["exercise"],
        "--unified-guidance","explicit-v2","--live","--provider","deepseek","--model",PAID["model"],
        "--reasoning-effort","high","--stream","--api-timeout","1200","--max-tokens","384000","--max-repairs","3","--provider-retries","3"]
  cases.append(dict(id=t["id"],track="directed_construction",model=m["model"],model_sha256=digest(m["model"]),
                    request=request,request_id=request["request_id"],request_bytes=len(canonical(request)),candidate_arguments=args,
                    compiler_configuration=config["name"],origin=dict(task_sha256=t["task_sha256"],requirement=t["requirement"],topology=m["topology"],split=m["split"]),status="prepared_not_run"))
 names=["stage2_composite_agent_plan.py","run_stage2_composite_agent.py","audit_stage2_guidance_retest.py",
        "stage2_agent_pilot_plan.py","stage2_agent_provenance.py","audit_stage2_live_candidate.py"]
 files={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n) for n in names}
 files.update({str(p.relative_to(ROOT)):sha(p) for p in (accepted_path,SUITE/"coverage.json",SUITE/"index.json")})
 plan=dict(format="poseidon-composite-agent-plan-r105-v1",source_hashes=sources,proposal_files=files,platform=identity(),
           corpus_model_set_sha256=index["model_set_sha256"],cases=cases,paid_configuration=PAID,
           limits=dict(LIMITS,maximum_generations=36,maximum_http_attempts=144),paid_calls=0,approved=False,
           executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
           batch_entrypoint="scripts/baseline/benchmarks/tools/run_stage2_composite_agent.py",
           selection="All three existing topology contexts for each of three public array composites. No outcome filtering.",
           parent_acceptance_binding=accepted["binding"],
           outbound_content=["public model and weights","frozen layout and DSL rules including component instructions","response schema","specified construction requirements","candidate code and restricted repair diagnostics"],
           never_outbound=["test inputs","reference answers","golden DSL","credentials","repository","FHE secret keys"],
           limitations=["Uses existing w45 explicitly; not a same-configuration causal comparison with r100 w40",
                        "Historical failed tasks retained. Three contexts per aggregate are not all combinations.",
                        "No currency hard cap. Finite 36 generation /144 HTTP /3600s /1GiB boundaries.",
                        "User authorized paid testing; execution still requires this frozen binding."])
 assert runtime_sources()==sources
 plan["binding"]=digest(plan);return plan
