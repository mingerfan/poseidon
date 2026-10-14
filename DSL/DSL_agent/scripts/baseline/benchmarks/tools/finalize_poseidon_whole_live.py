"""Finalize the bounded whole-model Agent batch without API or execution."""
import json,sys,hashlib,shlex
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from hecate_python_env import VENV,WORK,enter_nix
 import os
 if "--inside" not in sys.argv:
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=300,environment="poseidon-cpu")
 from run_poseidon_whole_live_v2 import document,verify,seal,write
 from agent_response_provenance import provider_accounting,verify_answers
 from audit_unified_candidate import verify_candidate,verify_files
 from candidate_contract import strict_json
 from unified_graph_contract import validate_candidate
 from semantic_benchmark_execution import verify_numerical_report
 from retained_artifact_usage import size_bytes
 out=WORK/"results/poseidon-whole-models-live-r452";target=WORK/"results/poseidon-whole-models-live-acceptance-r453"
 if target.exists():raise ValueError("Fresh final report required")
 plan=document(out/"plan.json");verify(plan);run=document(out/"report.json");rows=[];parents={}
 for row in run["rows"]:
  ev=Path(row["evidence"]);report_path=ev/"report.json";report=json.loads(report_path.read_text())
  if sha(report_path)!=row["report_sha256"]:raise ValueError("Executed report changed")
  c=next(c for c in plan["cases"] if c["id"]==row["id"])
  request=json.loads((ev/"request.json").read_text())
  if report["request_id"]!=c["request_id"] or request["request_id"]!=c["request_id"]:raise ValueError("Approved request drift")
  accounting=provider_accounting(report,plan["provider"]);verify_numerical_report(report,"poseidon-cpu-eva-w45-v1")
  verify_files(ROOT/"scripts/baseline",report["source_hashes"]);verify_files(ev,report["frozen_hashes"])
  received=accounting["received"];attempts=report["attempts"]
  if len(received)!=len(attempts):raise ValueError("Received responses/attempts differ")
  details=[];usage={}
  for a,call in zip(attempts,received):
   d=ev/("attempt-%02d"%a["index"]);raw=(d/"response.txt").read_text()
   candidate=strict_json(raw)
   if len(raw)!=call["content_characters"] or candidate["request_id"]!=c["request_id"] or candidate["hecate_source"]!=(d/"candidate.py").read_text():raise ValueError("Candidate differs from Agent response")
   validate_candidate(candidate,request)
   feedback=json.loads((d/"feedback.json").read_text())
   if feedback!=report["loop"]["feedback_history"][a["index"]]:raise ValueError("Retained feedback changed")
   diag=None
   if a.get("failure_layer")=="compiler":
    lines=(d/"compile.log").read_text().splitlines();diag="\n".join(lines[-3:])
   details.append(dict(index=a["index"],status=a.get("status"),failure_layer=a.get("failure_layer"),
     real_frontend=a.get("trace",{}).get("frontend")=="real_Hecate",compiled=bool(a.get("compiled")),
     encrypted_execution=bool(a.get("executed")),numerically_correct=bool(a.get("numerically_correct")),
     response_sha256=sha(d/"response.txt"),source_sha256=sha(d/"candidate.py"),compiler_tail=diag))
   for k,v in call.get("usage",{}).items():
    if type(v) is int:usage[k]=usage.get(k,0)+v
  result=dict(row,attempts=details,usage=usage,first_attempt_passed=report["loop"]["first_attempt_passed"],
     repairs_used=report["loop"]["repairs_used"],transport_retries=accounting["http_attempts"]-accounting["generations"])
  if row["status"]=="passed":
   provenance=verify_answers(ev,report,request,plan["provider"])
   audit=verify_candidate(ev,live_approval=plan["provider"])
   result["provenance"]=provenance;result["numerical_comparison"]=audit["comparison"]
   manifest=json.loads((Path(row["program"])/"manifest.json").read_text())
   if not manifest["agent_generated"] or manifest["validation"]["level"]!="numerical" or manifest["binding"]!=row["package_binding"]:raise ValueError("Program acceptance/provenance drift")
  rows.append(result);parents[str(report_path)]=sha(report_path)
 source=plan["runtime_source_hashes"]
 if not all(sha(ROOT/p)==h for p,h in source.items()):raise ValueError("Final source drift")
 report=seal(dict(format="poseidon-whole-models-live-acceptance-v1",rows=rows,passed=sum(r["status"]=="passed" for r in rows),
  failed=sum(r["status"]=="failed" for r in rows),not_run=run["not_run"],skipped=0,
  first_attempt_passed=sum(r["status"]=="passed" and r["first_attempt_passed"] for r in rows),
  successful_after_repair=sum(r["status"]=="passed" and not r["first_attempt_passed"] for r in rows),
  generations=sum(r["generation_attempts"] for r in rows),http_attempts=sum(r["http_attempts"] for r in rows),
  transport_retries=sum(r["transport_retries"] for r in rows),maximum_generations=12,maximum_http_attempts=48,
  compiled_attempts=sum(a["compiled"] for r in rows for a in r["attempts"]),
  encrypted_executions=sum(a["encrypted_execution"] for r in rows for a in r["attempts"]),
  compared_values=sum(r.get("comparison",{}).get("compared_values",0) for r in rows),
  max_absolute_error=max(r.get("comparison",{}).get("max_absolute_error",0) for r in rows),
  reload_values=sum(r.get("reload_comparison",{}).get("compared_values",0) for r in rows),
  reload_max_absolute_error=max(r.get("reload_comparison",{}).get("max_absolute_error",0) for r in rows),
  numeric_tolerance=plan["numeric_tolerance"],parent_binding=plan["binding"],original_authorization=plan["authorization_reuse"],
  source_hashes=source,application_source_hashes=plan["source_hashes"],backend_identity=plan["backend_identity"],
  original_reports=parents,developer_source_fixes_only=True,
  developer_failures_preserved=["poseidon-whole-models-live-r450","poseidon-whole-models-live-r451"],
  provider_generated_program_rewritten=False,new_paid_calls_by_finalizer=0,
  execution_seconds=run["seconds"],cost_usd=None,
  scope="Small explicit-approximation models; not complete operator, network accuracy or ResNet20 acceptance"))
 target.mkdir();write(target/"report.json",report,True)
 docs=ROOT/"docs/baseline/poseidon-whole-models-agent-acceptance-r453.json";write(docs,report,True)
 print(json.dumps({k:report[k] for k in ["passed","failed","not_run","skipped","generations","http_attempts","transport_retries","compared_values","max_absolute_error","reload_values","reload_max_absolute_error","binding"]}),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
