"""Read-only audit of unchanged Agent-source replay, with honest attribution."""
import argparse,copy,json,os,shlex,shutil,sys,time
from pathlib import Path
from replay_scope import *
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--inside",action="store_true")
args=parser.parse_args()
if not args.inside:
 from hecate_python_env import enter_nix,VENV
 supervision.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,keep_env=(),environment="poseidon-cpu"))
from hecate_python_env import VENV
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"replay_audit_pinned_environment")
import numpy as np
from candidate_bundle import check_bundle,check_provenance
from audit_unified_candidate import verify_candidate
from generation_receipt import verify as verify_generation
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from numerical_results import compare
from program_execution import package_execution_metadata
from pytorch_source_contract import restore_outputs
import saved_outputs
from saved_outputs import declared_outputs
p=verify();plan=read(RUN/"plan.json");report=read(RUN/"report.json")
bound(plan);bound(report);supervision.no_active()
require(report["status"]=="complete"and report["proposal_binding"]==p["binding"]==plan["proposal_binding"]and
 not supervision.same_process(plan["owner"]),"replay_audit_terminal_owner")
counts=terminal(report["rows"],p["cases"]);require(all(report[k]==v for k,v in counts.items()),"replay_audit_denominator")
observed={str(PROPOSAL):sha(PROPOSAL),str(RUN/"plan.json"):sha(RUN/"plan.json"),
 str(RUN/"report.json"):sha(RUN/"report.json"),str(Path(__file__)):sha(Path(__file__)),
 str(Path(saved_outputs.__file__)):sha(Path(saved_outputs.__file__))}
rows=[];groups=values=0;maximum=0.;absolute_sum=0.;compile_count=encrypted_count=0
bycase={c["id"]:c for c in p["cases"]}
for row in report["rows"]:
 case=bycase[row["id"]]
 require(row["paid_generations"]==row["http_attempts"]==0,"replay_audit_unpaid_case")
 if row["status"]!="passed":
  rows.append(dict(id=row["id"],status=row["status"],failure=row.get("failure")));continue
 require(row["new_agent_generation"]is False and row["source_unchanged"]is True and row["input_groups"]==8,
  "replay_audit_honest_source")
 folder=RUN/row["id"];task=Path(row["task_directory"]);require(task==Path(case["source"]["qualification"]["source_task_directory"]),"audit_qualified_task_identity");state=read(task/"state.json");meta=read(task/"task.json")
 bound(meta);q=read(case["source"]["request_file"])
 require(state["status"]=="succeeded"and state["generations"]==1 and state["http_attempts"]==0 and
  not any(v.get("event")=="http_reserved"for v in state["ledger"]),"replay_audit_no_transport")
 require(meta["request"]==q and meta["identity"]==p["stage_identity"],"replay_audit_current_request_identity")
 check_provenance(meta["model_provenance"],q["model"])
 package=Path(row["package"]);manifest=check_bundle(package)
 require(manifest["agent_generated"]is False and manifest["binding"]==row["package_binding"]and
  sha(package/"candidate.py")==case["origin"]["candidate_sha256"]==row["candidate_sha256"],
  "replay_audit_exact_original_candidate")
 origin=verify_generation(read(case["origin"]["receipt_file"]),q,(package/"candidate.py").read_text(),
  generation_identity_sha256=case["origin"]["generation_identity_sha256"])
 require(origin["receipt_binding"]==row["original_generation_receipt_binding"],"replay_audit_genuine_origin")
 require(read(package/"model-provenance.json")==meta["model_provenance"],"replay_audit_source_provenance")
 qual=read(task/"validation-0.json");require(qual==row["qualification"],"replay_audit_qualification_receipt")
 evidence=Path(qual["evidence"]);initial=verify_candidate(evidence);attempt=read(evidence/"attempt-00/report.json")
 require(initial["request_id"]==q["request_id"]and initial["comparison"]==attempt["comparison"]and
  attempt["compiled"]is True and attempt["executed"]is True and attempt["execution"]["backend"]=="poseidon_cpu"and
  attempt["execution"]["executor_secret_key"]is False,"replay_audit_true_backend_qualification")
 compile_count+=1;encrypted_count+=1
 for rel,want in initial["files"].items():observed[str(evidence/rel)]=want
 comparisons=[attempt["comparison"]]
 model=q["model"];tree=package_execution_metadata(package,manifest,model)["output_structure"]
 for name,n in(("new_inputs",3),("repeat",1)):
  phase=read(folder/(name+".json"));require(phase==row[name]and phase["status"]=="passed","replay_audit_phase")
  execution=phase["execution"]
  require(execution["encrypted_execution"]is True and execution["executor_secret_key"]is False and
   execution["agent_calls"]==execution["compiler_calls"]==0 and execution["backend"]=="poseidon_cpu",
   "replay_audit_package_reexecution")
  require(execution.get("output_structure")==tree,"replay_audit_output_tree")
  if tree is not None:
   require(origin_scope.canonical(execution["structured_outputs"])==origin_scope.canonical([
    restore_outputs(declared_outputs(output,[o["name"]for o in model["outputs"]]),tree,allow_none=True)for output in execution["outputs"]]),"replay_audit_saved_output_structure")
  refs=[]
  for i in range(n):
   rng=np.random.default_rng(20261008+i)
   inputs={v["name"]:rng.uniform(-.2,.2,v["shape"]).astype(np.float64)for v in model["inputs"]}
   a=math_ref(model,inputs);b=torch_ref(model,inputs)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
   refs.append(np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]]))
  actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)
   for o in model["outputs"]])for v in execution["outputs"]])
  compared=compare(actual,np.asarray(refs,dtype=np.float64),1e-5,1e-4)
  require(compared==phase["comparison"]and compared["passed"],"replay_audit_independent_reexecution_reference")
  precheck=execution["resource_precheck"]
  require(precheck["status"]=="assessed"and precheck["failure"]is None and
   read(Path(execution["evidence"])/"output/resource-precheck.json")==precheck,"replay_audit_resources")
  comparisons.append(compared);encrypted_count+=1
  for file in Path(execution["evidence"]).rglob("*"):
   if file.is_file():require(not file.is_symlink(),"replay_audit_regular_evidence");observed[str(file)]=sha(file)
 for compared,n in zip(comparisons,(4,3,1)):
  require(compared["passed"]and compared["atol"]==1e-5 and compared["rtol"]==1e-4,"replay_audit_frozen_tolerance")
  delta=np.abs(np.asarray(compared["actual"])-np.asarray(compared["reference"]))
  groups+=n;values+=int(delta.size);maximum=max(maximum,float(delta.max()));absolute_sum+=float(delta.sum())
 cache=read(folder/"cache.json")
 require(cache["status"]=="succeeded"and cache["package"]["reused"]is True and
  cache["package"]["binding"]==manifest["binding"]and cache["generations"]==cache["http_attempts"]==0,
  "replay_audit_own_source_cache")
 require(row["compared_values"]==sum(c["compared_values"]for c in comparisons),"replay_audit_value_denominator")
 for file in(folder/"case-report.json",task/"task.json",task/"state.json",task/"validation-0.json",
  package/"manifest.json",package/"candidate.py",folder/"new_inputs.json",folder/"repeat.json",folder/"cache.json"):
  observed[str(file)]=sha(file)
 rows.append(dict(id=row["id"],status="passed",candidate_sha256=row["candidate_sha256"],
  original_generation_receipt_binding=origin["receipt_binding"],new_agent_generation=False,package_binding=manifest["binding"],
  input_groups=8,compared_values=row["compared_values"],maximum_absolute_error=row["maximum_absolute_error"]))
require(groups==counts["passed"]*8 and compile_count<=BOUNDS["max_compiler_calls"]and
 encrypted_count<=BOUNDS["max_encrypted_calls"],"replay_audit_execution_denominator")
global_bytes=supervision.measure(RESULTS)
require(global_bytes<=BOUNDS["global_result_bytes"]and global_bytes-plan["initial_bytes"]<=BOUNDS["new_result_bytes"]and
 shutil.disk_usage(WORK).free>=BOUNDS["minimum_free_bytes"],"replay_audit_storage_ceiling")
for path,want in observed.items():require(sha(path)==want,"replay_audit_evidence_changed")
data=dict(format="poseidon-static-broadcast-replay-acceptance-v1",audit_status="passed",**counts,
 proposal_binding=p["binding"],stage_identity=p["stage_identity"],rows=rows,evidence_hashes=observed,
 input_groups=groups,compared_values=values,maximum_absolute_error=maximum if values else None,
 mae=absolute_sum/values if values else None,compiler_qualifications=compile_count,encrypted_calls=encrypted_count,
 new_agent_generation=False,origin="unchanged genuine Agent DSL; separately bound scripted replay packages",
 resource_observation=dict(global_result_bytes=global_bytes,new_bytes=global_bytes-plan["initial_bytes"]),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),backend_modified=False,parameter_modified=False,network_modified=False)
data["binding"]=digest(data);out=ROOT/"docs/baseline/agent-static-broadcast-replay-acceptance-r1431.json"
with out.open("x")as f:json.dump(data,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:data[k]for k in("audit_status","benchmark_status","planned","passed","failed","not_run",
 "input_groups","compared_values","maximum_absolute_error","mae","binding")}))
