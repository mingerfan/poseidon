"""Reconcile terminal retained evidence without rerunning any provider request."""
import argparse,json,os,shlex,sys
from pathlib import Path
from stage2_agent_campaign import BASE,ROOT
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed,resume_rows
from campaign_request_files import read_request
from run_stage2_guidance_retest import evidence_path,terminal_failure_layer

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--batch",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned Python required")
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from audit_stage2_live_candidate import verify_candidate,verify_files
 from stage2_agent_provenance import provider_accounting
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct output required")
 if a.batch.is_symlink() or a.batch.resolve().parent!=RESULTS.resolve():raise ValueError("Direct original batch required")
 plan=strict_file(a.batch/"plan.json",8*1024**2);check_binding(plan)
 report=strict_file(a.batch/"report.json",8*1024**2);check_binding(report)
 if report["plan_binding"]!=plan["binding"] or runtime_sources()!=plan["source_hashes"]:raise ValueError("Source or plan mismatch")
 for name,hsh in plan["proposal_files"].items():
  if sha(ROOT/name)!=hsh:raise ValueError("Original proposal changed")
 parents={str(a.batch/n):sha(a.batch/n) for n in ("plan.json","report.json")}
 rows=list(report["rows"]);done={r["id"] for r in rows};recovered=[]
 for spec in plan["cases"]:
  c=a.batch/spec["id"];launch=c/"launch.json"
  if not launch.exists():continue
  launch_data=strict_file(launch,131072);check_binding(launch_data)
  command=[plan["executable"],"-B",plan["entrypoint"],"--case",str(c/"model.json"),*spec["candidate_arguments"]]
  if launch_data["command"]!=command or launch_data["plan_binding"]!=plan["binding"]:raise ValueError("Launch identity")
  for proc in Path("/proc").glob("[0-9]*/cmdline"):
   try:actual=proc.read_bytes().rstrip(b"\0").split(b"\0")
   except (FileNotFoundError,ProcessLookupError,PermissionError):continue
   if actual==[x.encode() for x in command]:raise ValueError("Candidate still running")
  if spec["id"] in done:continue
  folder=evidence_path(c/"run.log",RESULTS)
  if folder is None:raise ValueError("Uncertain launch: no evidence")
  candidate=strict_file(folder/"report.json",8*1024**2)
  if candidate["status"]!="passed":raise ValueError("Recovery requires separately audited terminal success")
  read_request(folder/"request.json",spec["request"])
  if digest(strict_file(folder/"model.json",131072))!=spec["model_sha256"]:raise ValueError("Model changed")
  audit=verify_candidate(folder,plan["paid_configuration"])
  accounting=provider_accounting(candidate,plan["paid_configuration"])
  if len(accounting["received"])!=len(candidate["attempts"]):raise ValueError("Incomplete responses")
  if "Candidate status: passed" not in (c/"run.log").read_text():raise ValueError("Missing terminal log")
  for file in (launch,c/"process.json",c/"run.log",folder/"report.json",folder/"request.json"):
   parents[str(file)]=sha(file)
  rows.append(dict(id=spec["id"],track=spec["track"],evaluation_identity=spec["evaluation_identity"],
   status="runner_reported_pass_pending_audit",exit_code=None,exit_code_note="Original exit code was not retained; independent terminal evidence audited",
   evidence=str(folder),report_sha256=sha(folder/"report.json"),agent_calls=accounting["http_attempts"],
   generations=accounting["generations"],failure_layer=terminal_failure_layer(candidate),
   recovered_without_provider_call=True))
  recovered.append(dict(id=spec["id"],audit=audit))
 if not recovered:raise ValueError("No terminal uncheckpointed candidate")
 # Validate all old rows and every launched item before allowing any successor.
 review_report=sealed(dict(plan_binding=plan["binding"],rows=rows,seconds=report["seconds"]))
 resume_rows(plan,review_report,a.batch)
 if sum(r.get("generations",0) for r in rows)>plan["limits"]["maximum_generations"]:raise ValueError("Generation budget")
 if sum(r.get("agent_calls",0) for r in rows)>plan["limits"]["maximum_http_attempts"]:raise ValueError("HTTP budget")
 for file,hsh in parents.items():
  if sha(Path(file))!=hsh:raise ValueError("Recovery evidence drift")
 a.output.mkdir()
 record=sealed(dict(format="poseidon-terminal-recovery-v1",original_plan_binding=plan["binding"],parents=parents,
  source_hashes=plan["source_hashes"],rows=rows,seconds=report["seconds"],recovered=recovered,
  paid_calls=0,encrypted_executions=0,reason="Explicit parallel scheduling handoff after terminal candidate; original wall budget retained",
  runner_sha256=sha(Path(__file__))))
 dump(a.output/"recovery.json",record)
 successor=dict(plan);successor.pop("binding")
 successor["proposal_files"]=dict(plan["proposal_files"])
 for name in ("run_stage2_agent_campaign_v2.py","audit_stage2_campaign_shard_v2.py","campaign_request_files.py","recover_parallel_terminal.py"):
  q=Path(__file__).with_name(name);successor["proposal_files"][str(q.relative_to(ROOT))]=sha(q)
 successor["recovery"]=dict(record=str(a.output/"recovery.json"),sha256=sha(a.output/"recovery.json"))
 successor["binding"]=digest(successor);dump(a.output/"plan.json",successor)
 seed=sealed(dict(format="poseidon-stage2-agent-campaign-run-v1",plan_binding=successor["binding"],
  rows=rows,seconds=report["seconds"],failure=None,passed=0,independent_audit_complete=False,
  not_run_or_interrupted=[s["id"] for s in plan["cases"] if s["id"] not in {r["id"] for r in rows}],
  recovery_binding=record["binding"],interrupted_call_count_may_be_unknown=False,
  generations_in_completed_reports=sum(r.get("generations",0) for r in rows),
  http_attempts_in_completed_reports=sum(r.get("agent_calls",0) for r in rows),stage2_complete=False))
 dump(a.output/"report.json",seed);dump(a.output/"progress.json",rows)
 print(json.dumps(dict(binding=successor["binding"],recovered=[x["id"] for x in recovered],
  completed=len(rows),remaining=len(plan["cases"])-len(rows),seconds_used=report["seconds"],paid_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
