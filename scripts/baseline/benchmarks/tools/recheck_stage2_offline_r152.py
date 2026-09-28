"""Revalidate all retained offline requirements plus current static gates; no historical FHE rebinding."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature,validate
from benchmark_runner import strict_file,dump,load
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def bound(p):
 d=strict_file(p,16*1024**2)
 need(digest({k:v for k,v in d.items() if k!="binding"})==d["binding"],"Report identity: "+str(p))
 return d
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 if a.output.exists():p.error("Preserve prior evidence")
 from hecate_python_env import VENV,enter_nix
 if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 need(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure environment")
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from benchmark_semantics import features
 from rejection_benchmark import tasks as rejection_tasks,run_task
 sources=runtime_sources();runner_hash=sha(Path(__file__));start=time.monotonic();D=ROOT/"docs/baseline"
 from retained_acceptance_context_r152 import context
 lineage=context();historical_sha=lineage["historical_runtime_sha256"]
 # The joint verifier checks retained dependency artifacts, security gates and current source compatibility.
 process=subprocess.run([str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("recheck_stage2_joint_r152.py")),"--output",str(a.output.with_suffix(".joint.json"))],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
 need(process.returncode==0,"Joint verifier: "+process.stderr[-1000:])
 joint=bound(D/"stage2-status-r92.json")
 rechecked=bound(a.output.with_suffix(".joint.json"))
 stripped={k:v for k,v in rechecked.items() if k not in ("binding","runner_sha256","revalidation_context")}
 expected={k:v for k,v in joint.items() if k not in ("binding","runner_sha256")}
 need(stripped==expected,"Historical joint facts changed")
 parents={str(D/"stage2-status-r92.json"):sha(D/"stage2-status-r92.json")}
 def get(path,binding=True):
  parents[str(path)]=sha(path)
  return bound(path) if binding else strict_file(path,16*1024**2)
 linkage=get(D/"stage2-contract-linkage-r90.json")
 need(linkage["current_source_sha256"]==historical_sha,"Current contract source")
 suite=BASE/"benchmarks/semantic-v2-ledger-r38"
 models,index=load(suite,check_sources=False);model_by={r["model"]["id"]:r for r in models}
 ledger=get(suite/"coverage.json",False)
 need(len(ledger["requirements"])==401 and len({r["id"] for r in ledger["requirements"]})==401,"Frozen denominator")
 for item in ledger["components"].values():need(sha(suite/item["file"])==item["sha256"],"Ledger component hash")
 overlap=get(suite/"legacy-overlap.json",False)
 need(overlap["legacy_cases"]==96 and not overlap["overlaps"] and not overlap["normalization_errors"],"Legacy anchors")
 plaintext=get(RESULTS/"semantic-v1-plaintext-release/report.json",False)
 need((plaintext["planned"],plaintext["passed"],plaintext["failed"],plaintext["skipped"],plaintext["tests"])==(1200,1200,0,0,19200),"Dual-reference denominator")
 for name,hsh in plaintext["result_hashes"].items():
  q=RESULTS/"semantic-v1-plaintext-release"/(name+".json");need(sha(q)==hsh,"Plaintext record hash")
  row=strict_file(q,1024**2)
  need(row["model_sha256"]==model_by[name]["model_sha256"] and row["status"]=="passed" and row["tests"]==16 and row["binding"]==plaintext["binding"],"Plaintext model/probe identity")
 mathematical=get(D/"stage2-context-supplement-r48.json")
 maths={r["id"]:r for r in mathematical["partitions"]}
 historical_base=get(RESULTS/"benchmark-r41-independent-audit/report.json",False)
 construction_report=get(RESULTS/"benchmark-r43-directed-independent/report.json",False)
 helpers=get(D/"stage2-helper-mapping-r49.json");recovered=get(D/"stage2-helper-recovered-r93.json")
 need(recovered["current_runtime_sha256"]==historical_sha,"Recovered helper source")
 helper_records=dict(helpers["records"])
 for r in recovered["rows"]:
  need(r["task_id"] not in helper_records,"Duplicate recovered helper")
  helper_records[r["task_id"]]=dict(r,actual_frontend_checked=True,finite_return_influence_checked=True,numerical=True)
 need(set(helper_records)=={r["id"] for r in ledger["helper_directed_tasks"]},"All 60 frozen helper tasks accounted")
 helper_proofs={}
 for parent in helpers["input_reports"]:
  folder=Path(parent["batch"]);need(sha(folder/"report.json")==parent["report_sha256"],"Helper parent hash")
  batch=get(folder/"report.json",False)
  for record in batch["records"]:helper_proofs["upstream_"+record["case_id"]]=record
 for r in recovered["rows"]:helper_proofs[r["task_id"]]=r
 need(set(helper_proofs)==set(helper_records),"Helper artifact proofs incomplete")
 negative_model=get(suite/"semantic-component-model_negatives.json",False)
 model_negatives={}
 for t in negative_model["tasks"]:
  positive=model_by[t["positive_model_id"]]["model"];need(digest(positive)==t["positive_model_sha256"],"Negative control positive identity")
  validate(positive);need(digest(t["negative_model"])==t["negative_model_sha256"],"Negative fixture identity")
  try:validate(t["negative_model"])
  except ValueError as error:need(str(error)==t["observed_rejection"],"Wrong model rejection")
  else:raise ValueError("Model negative accepted")
  model_negatives[t["requirement"]]=dict(id=t["id"],task_sha256=t["task_sha256"],positive=True,negative=True,scope="current static model gate")
 def checker(folder,count):
  report=get(folder/"report.json",False);plan=get(folder/"plan.json",False)
  need(report["binding"]==plan["binding"] and (report["passed"],report["failed"],report["skipped"])==(count,0,0),"Counterexample outcome")
  need(sha(folder/"results.json")==report["results_sha256"],"Counterexample result hash")
  return strict_file(folder/"results.json",8*1024**2)
 trace_negatives={r["requirement"]:r for r in checker(RESULTS/"construction-trace-counterexamples",209)}
 helper_negatives=checker(RESULTS/"helper-call-negatives-000",48)+checker(RESULTS/"helper-call-negatives-001",28)
 need(all(r["positive_contract_accepted"] and r["call_omission_rejected"] and r["state"]=="passed" for r in helper_negatives),"Helper negative scope")
 need({r["task_id"] for r in helper_negatives}==set(helper_records),"Helper negative coverage")
 static={t["id"]:run_task(t,rejection_tasks()) for t in rejection_tasks()}
 need(len(static)==48 and all(r["status"]=="passed" and r["positive_control_accepted"] and r["negative_rejected"] for r in static.values()),"Current static rejection coverage")
 gates=get(RESULTS/"stage2-gates-r51/report.json")
 need(sha(RESULTS/"stage2-gates-r51/compiler-contexts.json")==gates["files"]["compiler-contexts.json"],"Compiler contexts hash")
 compiler=strict_file(RESULTS/"stage2-gates-r51/compiler-contexts.json",8*1024**2)
 proofs={};checked_files=set()
 def retained(folder,hashes):
  for name,hsh in hashes.items():
   rel=Path(name);path=folder/rel
   need(not rel.is_absolute() and ".." not in rel.parts and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()),"Evidence path")
   key=(str(path),hsh)
   if key not in checked_files:need(sha(path)==hsh,"Changed retained evidence");checked_files.add(key)
 def execution(proof):
  folder=Path(proof["evidence"]);need(folder.parent==RESULTS,"Execution location")
  key=str(folder)
  if key in proofs:return proofs[key]
  retained(folder,proof["files"])
  r=strict_file(folder/"report.json",8*1024**2)
  need(r["status"]=="passed" and r["agent_calls"]==0 and not r["llm_generation_validated"],"Historical manual pass required")
  attempt=next(x for x in r["attempts"] if x.get("status")=="passed")
  need(all(attempt.get(n) for n in ("compiled","executed","numerically_correct")) and attempt["trace"]["frontend"]=="real_Hecate" and attempt["execution"]["encrypted_execution"],"Complete FHE chain")
  cmp=attempt["comparison"];need(cmp["passed"] and (cmp["atol"],cmp["rtol"])==(1e-5,1e-4),"Frozen numerical gate")
  out=folder/("attempt-%02d"%attempt["index"])/"output";retained(out,attempt["artifact_hashes"])
  need(strict_file(folder/"key-cleanup-outcome.json",65536)["complete"] and not (folder/"private-keys").exists(),"Key cleanup")
  request=strict_file(folder/"request.json",4*1024**2)
  need(digest(request["model"])==proof["model_sha256"],"Executed model identity")
  proofs[key]=dict(evidence=key,report_sha256=sha(folder/"report.json"),model_sha256=proof["model_sha256"],
                  topology=signature(request["model"],True),historical_source_hashes=r["source_hashes"],
                  comparison_passed=True,new_execution=False)
  return proofs[key]
 def audit_record(path,expected=None):
  if expected:need(sha(path)==expected,"Positive audit hash")
  record=strict_file(path,8*1024**2);execution(record);return record
 def math_record(context):
  name=context.get("model_id",context.get("id"))
  if "audit_sha256" in context:
   options=[Path(b["batch"])/name/"fhe-audit.json" for b in mathematical["batches"]]
   path=next((x for x in options if x.is_file() and sha(x)==context["audit_sha256"]),None)
   need(path is not None,"Diagnostic context audit")
   return audit_record(path,context["audit_sha256"])
  path=RESULTS/"benchmark-r41-independent-audit"/(name+".json")
  if path.is_file():return audit_record(path,historical_base["record_hashes"][path.name])
  path=RESULTS/"benchmark-r43-stage2-closure"/(name+".json")
  need(path.is_file(),"Historical mathematical context audit")
  return audit_record(path)
 tasks={r["id"]:r for r in ledger["directed_tasks"]};hparts={r["id"]:r for r in helpers["helper_partitions"]}
 matrix=[]
 for req in ledger["requirements"]:
  ident=req["id"];layer=req["layer"]
  need(req.get("source_basis") and req.get("applicable_contracts"),"Missing source/contract: "+ident)
  bases=[]
  for basis in req["source_basis"]:
   q=ROOT/basis["path"];need(q.is_file(),"Source basis missing")
   bases.append(dict(basis,current_sha256=sha(q)))
  item=dict(id=ident,layer=layer,source_basis=bases,applicable_contracts=req["applicable_contracts"],
            claim_scope=req.get("claim_scope",req.get("evidence_scope")),acceptance_levels=req["acceptance_levels"],
            original_positive_examples=req.get("positive_examples",[]),original_negative_examples=req.get("negative_examples",[]),
            status="pending",contexts=[],negative_checks=[],blocker=None)
  if layer=="model":
   for context in maths[ident]["contexts"]:
    if context["state"]!="verified":continue
    record=math_record(context);request=strict_file(Path(record["evidence"])/"request.json",4*1024**2)
    need(ident in features(request["model"]) and signature(request["model"],True)==context["topology"],"Math partition not exercised")
    item["contexts"].append(dict(context,evidence=record["evidence"],model_sha256=record["model_sha256"]))
   need(len({r["topology"] for r in item["contexts"]})>=3,"Missing mathematical contexts")
   item["negative_checks"]=[model_negatives[ident]];item["status"]="accepted_historical_numeric"
  elif layer=="construction":
   for example in req["positive_examples"]:
    tid=example["id"];task=tasks[tid]
    record=audit_record(RESULTS/"benchmark-r43-directed-independent"/(tid+".json"),construction_report["record_hashes"][tid+".json"])
    need(record["task_sha256"]==task["task_sha256"] and record["requirement"]==ident and record["model_sha256"]==task["model_sha256"],"Construction identity")
    coverage=record["coverage"];need(coverage["actual_frontend_checked"],"Missing actual construction trace")
    need(task["acceptance_kind"] in ("numeric_influence","mixed","structure_only"),"Unknown acceptance kind")
    if task["acceptance_kind"]!="structure_only":need(coverage["finite_influence_checked"],"Missing numeric return intervention")
    else:need(coverage["structural_only"] and not coverage["finite_influence_checked"],"Structural item cannot claim numeric contribution")
    item["contexts"].append(dict(task_id=tid,topology=example["topology"],evidence=record["evidence"],
                                 acceptance_kind=task["acceptance_kind"]))
   need(len({r["topology"] for r in item["contexts"]})>=3,"Missing construction contexts")
   negative=trace_negatives[ident];need(negative["positive_gate_passed"] and negative["negative_gate_rejected"],"Missing construction negative")
   item["negative_checks"]=[negative];item["status"]="accepted_historical_construction"
  elif layer in ("upstream_helper","upstream_helper_layout"):
   part=hparts[ident]
   if part["blocker"]:
    item.update(status="backend_blocked",blocker=part["blocker"])
   else:
    for tid in req["directed_task_ids"]:
     r=helper_records[tid];need(r["actual_frontend_checked"] and r["finite_return_influence_checked"] and r["numerical"],"Helper proof")
     need(sha(Path(r["evidence"])/"report.json")==r["report_sha256"],"Helper report identity")
     execution(helper_proofs[tid])
     item["contexts"].append(dict(task_id=tid,topology=r["topology"],evidence=r["evidence"],historical_runtime_sha256=r["historical_runtime_sha256"]))
    need(len({r["topology"] for r in item["contexts"]})>=3,"Missing helper contexts: "+ident+" count="+str(len(item["contexts"])))
    item["negative_checks"]=[r for r in helper_negatives if r["task_id"] in req["directed_task_ids"]]
    need(item["negative_checks"],"Missing helper negatives");item["status"]="accepted_historical_helper"
   item["capability_scope"]=part["scope"]
  elif layer=="compiler":
   for r in compiler["examples"][ident]:
    evidence=RESULTS/r["evidence_relative_to_results"]
    need(sha(evidence/"report.json")==r["report_sha256"] and r["actual_encrypted_execution_recorded"],"Compiler example")
    retained(evidence/"attempt-00/output",r["artifact_hashes"])
    item["contexts"].append(r)
   need(len({r["topology"] for r in item["contexts"]})>=3,"Compiler contexts")
   item["negative_checks"]=req["negative_examples"]
   need(gates["unit_failed"]==0 and gates["unit_skipped"]==0,"Compiler rejection audit")
   item["status"]="accepted_historical_artifact"
  elif layer=="rejection":
   item["negative_checks"]=[static[n] for n in req["task_ids"]]
   need(len(item["negative_checks"])==3,"Rejection contexts");item["status"]="accepted_current_static"
  else:raise ValueError("Unknown requirement layer")
  matrix.append(item)
  need(time.monotonic()-start<240,"Completion audit budget")
 # API partitions are separately scoped: a pure utility is not a ciphertext program;
 # factory metaclasses and trusted I/O are not directly callable candidate instructions.
 api=[];outside={"trusted_framework_io","frontend_metaclass_and_operator_factory","defined_but_unregistered_inplace",
                 "trusted_framework_introspection","frontend_expression_handle","plaintext_diagnostic_utility","mutable_maximum_hook"}
 allowed={"mapped_bounded_entry_partition","backend_blocked","public_preprocessing_verified","outside_candidate_computation",
          "cipher_dependency_verified","candidate_fixed_polynomial_verified","trusted_closure_factory_verified","internal_direct_return_mapped",
          "configured_compiler_capacity_blocked","intentional_rejection_verified","bounded_dependency_execution_verified","native_function_semantics_mapped"}
 for r in joint["api_dispositions"]:
  need(r["state"] in allowed and sha(ROOT/r["source"])==r["source_sha256"],"Unclassified or changed API")
  if r["state"]=="outside_candidate_computation":need(r["role"] in outside,"Undocumented API exclusion")
  if r["state"]=="public_preprocessing_verified":need(r["public_contexts"]>=3,"Public preprocessing contexts")
  if r["state"]=="internal_direct_return_mapped":need(r["contexts"]>=3,"Internal contexts")
  route=("not_a_candidate_computation" if r["state"]=="outside_candidate_computation" else
         "backend_blocked" if r["state"] in ("backend_blocked","configured_compiler_capacity_blocked") else
         "trusted_public_preparation" if r["state"]=="public_preprocessing_verified" else
         "explicit_fixed_polynomial_profile" if r["state"]=="candidate_fixed_polynomial_verified" else
         "bound_helper_profile" if r["state"]=="mapped_bounded_entry_partition" else
         "implicit_frontend_or_trusted_helper_dependency")
  api.append(dict(r,candidate_route=route,unrestricted_direct_api_access=False))
 need(len(api)==107 and not joint["remaining_api"],"Public API coverage mapping")
 need(joint["original_queue_latest"]["all_planned_cases_accounted_for"] and joint["original_queue_latest"]["ready_unrun"]==0,"Original queue")
 need(joint["current_regression"]["failed"]==0 and linkage["all_legacy_contracts_accepted"],"Compatibility")
 need(linkage["free_request_preflight"]["counts"]=={"ready":1200},"Rule-independent request preparation")
 need(runtime_sources()==sources and sha(Path(__file__))==runner_hash,"Current source changed")
 need(all(sha(Path(n))==h for n,h in parents.items()),"Evidence changed during audit")
 counts=dict(Counter(r["status"] for r in matrix))
 ready=len(matrix)==401 and all(r["status"].startswith("accepted_") or r["status"]=="backend_blocked" for r in matrix)
 result=dict(format="poseidon-stage2-offline-acceptance-v1",scope="Fixed source and explicit finite semantic partitions; not unrestricted Python, all helper parameters, all programs, or formal equivalence",
  authoritative_definition=str(suite/"coverage.json"),definition_sha256=sha(suite/"coverage.json"),
  current_runtime_sha256=digest(sources),parents=parents,requirements=matrix,requirement_counts=counts,upstream_api=api,
  corpus=dict(models=1200,unique_signatures=index["unique_signatures"],topology_groups=index["topology_groups"],small_models=index["small_models"],
              split_counts=index["split_counts"],dual_reference_passed=1200,probes=19200,legacy_anchors=96),
  helper_tasks=dict(frozen=60,verified=60,recovered_without_reexecution=3),
  original_primary_status=joint["original_queue_latest"]["current_primary_counts"],
  current_compatibility=joint["contract_linkage"],current_regression=joint["current_regression"],
  fixed_polynomial_candidates=joint["new_polynomial_candidates"],
  retained_failed_cases="All original preflight/compiler/numerical and infrastructure failures remain in bound reports; a passing alternative never erases them",
  offline_engineering_acceptance_complete=ready,agent_evaluation_complete=False,
  stage2_complete=False,paid_calls=0,new_encrypted_executions=0,
  remaining_for_full_stage2=["Separately authorized real Agent free-generation evaluation","Separately authorized real Agent directed-construction evaluation"],
  mathematical_negative_controls_rechecked=len(model_negatives),static_negative_controls_rechecked=len(static),
  validated_historical_execution_directories=len(proofs),retained_files_verified=len(checked_files),
  runner_sha256=sha(Path(__file__)),seconds=time.monotonic()-start)
 result["format"]="poseidon-retained-offline-acceptance-r152"
 result["historical_source_lineage"]=lineage
 result["current_compatibility"]=lineage
 result["current_regression"]=dict(lineage["retained_regression"],new_tests_run=0,scope="Retained exact-current-source tests")
 result["parents"][str(a.output.with_suffix(".joint.json"))]=sha(a.output.with_suffix(".joint.json"))
 result["binding"]=digest(result);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("requirements","upstream_api","parents","current_compatibility")}));return int(not ready)
if __name__=="__main__":raise SystemExit(main())
