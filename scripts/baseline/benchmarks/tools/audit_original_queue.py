"""Join original failed/blocked model outcomes without replacing any model."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import load,strict_file,dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def bound(p):
 d=strict_file(p,8*1024**2)
 need(digest({k:v for k,v in d.items() if k!="binding"})==d["binding"],"Binding drift")
 return d
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Preserve prior report")
 from workspace_paths import RESULTS
 suite=BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
 frozen,index=load(suite,check_sources=False)
 models={r["model"]["id"]:r for r in frozen};need(len(models)==1200,"Frozen model denominator")
 supplement=strict_file(BASE/"benchmarks/model-contexts-draft-v1/tasks.json",4*1024**2)
 models.update({r["model"]["id"]:r for r in supplement["supplemental_models"]})
 original={}
 sources={}
 retries=[*("stage2-original-blockers-r58-"+str(i) for i in range(3)),
  "stage2-original-blockers-r59-2","stage2-original-blockers-r60-w40",
  *("stage2-ready-originals-r78-"+str(i) for i in range(3)),
  "stage2-original-compile-failures-w40-r80"]
 all_attempts=[]
 for batch in retries:
  folder=RESULTS/batch;plan=bound(folder/"plan.json");report=bound(folder/"report.json")
  need(report["plan_binding"]==plan["binding"],"Retry parent drift")
  sources[batch]=dict(path=str(folder),plan_sha256=sha(folder/"plan.json"),report_sha256=sha(folder/"report.json"),binding=report["binding"])
  selected={r["model"]["id"]:r for r in plan["models"]}
  need({r["id"] for r in report["rows"]}==set(selected),"Dropped or added original model")
  for row in report["rows"]:
   name=row["id"]
   need(row["model_sha256"]==models[name]["model_sha256"]==selected[name]["model_sha256"],"Original model changed")
   need(row["reference_status"]=="passed" and row["reference_probes"]==16,"Independent reference checks")
   if batch.startswith("stage2-original-blockers-r58"):
    need(name not in original,"Duplicate original blocker")
    original[name]=dict(id=name,model_sha256=row["model_sha256"],original_kind="preflight_blocked",
      state="ready_unrun" if row["preflight_status"]=="ready" else "rule_preflight_blocked",
      preflight_reason=row["preflight_reason"],attempts=[])
   if batch.endswith("r80"):
    need(name not in original and name in ("bench_helper_0040","bench_helper_0044"),"Original compiler-failure identity")
    original[name]=dict(id=name,model_sha256=row["model_sha256"],original_kind="compiler_failed",
      state="ready_unrun",preflight_reason=None,attempts=[])
   if row["execution_status"] in ("not_selected","blocked_preflight"):continue
   evidence=Path(row["evidence"])
   need(evidence.parent==RESULTS and not evidence.is_symlink(),"Candidate result location")
   need(sha(evidence/"report.json")==row["report_sha256"] and sha(folder/(name+".log"))==row["log_sha256"],"Candidate/report/log drift")
   actual=strict_file(evidence/"report.json",8*1024**2)
   need(actual["agent_calls"]==0 and not actual["llm_generation_validated"],"Incorrect Agent claim")
   need(not (evidence/"private-keys").exists(),"Unexpected retained keys")
   model=strict_file(evidence/"model.json",131072)
   need(digest(model)==row["model_sha256"],"Executed original model identity")
   retained={str(f.relative_to(evidence)):sha(f) for f in evidence.rglob("*") if f.is_file()}
   attempt=dict(id=name,batch=batch,compiler_configuration=plan["compiler_configuration"],
    status=row["execution_status"],failure_layer=row.get("failure_layer"),
    evidence=str(evidence),report_sha256=row["report_sha256"],files=retained,helper_execution_not_implied=True)
   if row["execution_status"]=="passed":
    audit_path=folder/(name+".audit.json");need(sha(audit_path)==row["audit_sha256"],"Independent candidate audit")
    audit=strict_file(audit_path,8*1024**2)
    for rel,hsh in audit["files"].items():need(sha(evidence/rel)==hsh,"Audited artifact identity")
    c=audit["comparison"];need(c["passed"] and (c["atol"],c["rtol"])==(1e-5,1e-4),"Numerical threshold")
    attempt.update(audit_sha256=row["audit_sha256"],comparison={k:c[k] for k in ("compared_values","mae","max_absolute_error")})
    original[name]["state"]="rechecked_encrypted_pass"
   else:
    layers={r.get("failure_layer") for r in actual["attempts"] if r.get("status")=="failed"}
    need(layers=={"compiler"} and row.get("failure_layer") in (None,"compiler"),"Unexpected original failure layer")
    attempt["failure_layer"]="compiler";attempt["failure_layer_source"]="actual_attempt_report"
    logs=list(evidence.rglob("compile.log"))
    need(len(logs)==1 and "'earth.mul' op failed to infer returned types" in logs[0].read_text(),"Compiler failure evidence")
    attempt["compiler_log"]=str(logs[0].relative_to(evidence))
    if original[name]["state"]!="rechecked_encrypted_pass":original[name]["state"]="compiler_failed"
   original[name]["attempts"].append(attempt);all_attempts.append(attempt)
 need(len(original)==114 and sum(r["original_kind"]=="preflight_blocked" for r in original.values())==112,"Original queue denominator")
 need(not any(r["state"]=="ready_unrun" for r in original.values()),"Ready original model still unexecuted")
 rows=sorted(original.values(),key=lambda r:r["id"])
 primary=[r for r in rows if r["id"].startswith("bench_")];extra=[r for r in rows if not r["id"].startswith("bench_")]
 need(len(primary)==112 and len(extra)==2,"Original primary/supplement partition")
 counts=Counter(r["state"] for r in primary)
 # 1,088 original independent passes remain historical; inspect record identities.
 historical=RESULTS/"benchmark-r41-independent-audit";old=strict_file(historical/"report.json",4*1024**2)
 need(old["independently_verified"]==1088 and old["remaining"]==0,"Historical audit incomplete")
 old_ids=set()
 for name,hsh in old["record_hashes"].items():
  need(Path(name).name==name and sha(historical/name)==hsh,"Historical independent record drift")
  record=strict_file(historical/name,8*1024**2);model_id=Path(name).stem
  need(record["model_sha256"]==models[model_id]["model_sha256"] and model_id not in original,"Historical model identity/overlap")
  old_ids.add(model_id)
 need(len(old_ids)==1088 and len(old_ids|{r["id"] for r in primary})==1200,"Full planned denominator")
 result=dict(format="poseidon-original-queue-audit-v1",sources=sources,rows=rows,
  frozen_index_sha256=sha(suite/"index.json"),historical_audit_sha256=sha(historical/"report.json"),
  historical_passes=1088,original_queue=114,original_preflight_blockers=112,original_compile_failures=2,
  current_primary_counts=dict(historical_encrypted_pass=1088,**counts),
  current_supplement_blockers=dict(Counter(r["state"] for r in extra)),
  execution_attempts=len(all_attempts),rechecked_passes=sum(r["state"]=="rechecked_encrypted_pass" for r in rows),
  ready_unrun=0,new_models_added=0,old_passes_rebound=False,paid_calls=0,
  all_planned_cases_accounted_for=True,all_models_passed=False,runner_sha256=sha(Path(__file__)))
 result["binding"]=digest(result);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("rows","sources")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
