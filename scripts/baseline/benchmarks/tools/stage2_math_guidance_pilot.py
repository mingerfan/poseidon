"""Freeze and CLI-check the six-case logical-semantics pilot. No provider calls."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
from stage2_agent_campaign import BASE,ROOT,PAID,LIMITS,definitions,identity
from run_stage2_agent_campaign_v4 import task_arguments,validate_plan
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
from benchmark_runner import dump,strict_file
from run_stage2_guidance_retest import evidence_path
IDS=("free_bench_boundary_0011","free_bench_boundary_0030","free_bench_boundary_0036",
     "free_bench_boundary_0044","free_bench_composition_0000","free_bench_graph_0012")

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned Python")
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from platform_config import identity as platform_identity,require_python_packages
 from unified_graph_contract import prepare,validate_request
 from compiler_configuration import PROFILE_SHA256,configuration
 import numpy as np,torch
 require_python_packages(torch,np)
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh pilot directory")
 specs,parents,inventory_binding=definitions();lookup={s["id"]:s for s in specs};sources=runtime_sources();cases=[]
 for case_id in IDS:
  spec=dict(lookup[case_id])
  if spec["track"]!="free":raise ValueError("This pilot expects free tasks")
  request=prepare(spec["model"],PROFILE_SHA256,configuration(spec["compiler_configuration"]),generation_guidance="explicit-v3")
  validate_request(request)
  spec.update(request=request,request_id=request["request_id"],candidate_arguments=task_arguments(spec),status="prepared_not_run")
  spec["evaluation_identity"]=identity(spec,sources);cases.append(spec)
 names=("stage2_agent_campaign.py","stage2_agent_pilot_plan.py","stage2_guidance_retest_plan.py",
  "run_stage2_guidance_retest.py","stage2_agent_provenance.py","audit_stage2_live_candidate.py","campaign_live_state.py",
  "run_stage2_agent_campaign_v4.py","audit_stage2_campaign_shard_v2.py","campaign_request_files.py",
  "campaign_candidate_failures.py","stage2_math_guidance_pilot.py")
 proposal={str(Path(__file__).with_name(n).relative_to(ROOT)):sha(Path(__file__).with_name(n)) for n in names}
 plan=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",generation_guidance="explicit-v3",source_hashes=sources,
  proposal_files=proposal,definition_parents=parents,inventory_binding=inventory_binding,platform=platform_identity(),
  shard_index=0,cases=cases,paid_configuration=PAID,
  limits=dict(LIMITS,maximum_generations=24,maximum_http_attempts=96),paid_calls=0,approved=False,
  executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
  scope="Two non-full-period rotations, one prior transparent failure, reduction and two small compatibility models; new request/runtime binding, old failures retained"))
 validate_plan(plan,plan["binding"])
 a.output.mkdir();dump(a.output/"plan.json",plan)
 rows=[]
 for spec in cases:
  case=a.output/spec["id"];case.mkdir();dump(case/"model.json",spec["model"])
  command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--case",str(case/"model.json"),
   "--prepare","--unified-guidance","explicit-v3","--compiler-configuration",spec["compiler_configuration"],"--inside"]
  with (case/"prepare.log").open("x") as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=90).returncode
  folder=evidence_path(case/"prepare.log",RESULTS)
  if code or folder is None:raise ValueError("CLI prepare failed")
  from campaign_request_files import read_request
  read_request(folder/"request.json",spec["request"])
  report=strict_file(folder/"report.json",8*1024**2)
  if report["status"]!="request_prepared_not_generated" or report["agent_calls"]!=0:raise ValueError("Prepare unexpectedly executed")
  rows.append(dict(id=spec["id"],cli_prepare="passed",evidence=str(folder),request_id=spec["request_id"]))
 if runtime_sources()!=sources:raise ValueError("Runtime changed")
 result=sealed(dict(format="poseidon-mathematical-guidance-pilot-prepared-v1",plan_binding=plan["binding"],rows=rows,
  cli_prepared=len(rows),paid_calls=0,encrypted_executions=0,stage2_complete=False))
 dump(a.output/"report.json",result)
 print(json.dumps(dict(plan_binding=plan["binding"],prepared=len(rows),maximum_generations=24,maximum_http_attempts=96,paid_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
