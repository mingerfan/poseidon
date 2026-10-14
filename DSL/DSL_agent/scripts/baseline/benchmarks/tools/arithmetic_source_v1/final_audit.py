"""Read-only audit of separate historical and current v21 acceptance bindings."""
import hashlib,json,pathlib,subprocess,time,shutil
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
RESULTS=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
NAMES=["agent-exact-arithmetic-v21-source-acceptance-v1","agent-exact-arithmetic-v21-mask-repair-v1","agent-exact-arithmetic-v21-mask-support-v1"]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def require(ok,code):
 if not ok:raise ValueError(code)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def size(p):return int(subprocess.check_output(["du","-sb",str(p)],text=True,timeout=20).split()[0])
def audit():
 before=read(HERE/"before.json")
 for key in ["protected_sources","historical_request_files"]:
  for n,h in before[key].items():require(sha(pathlib.Path(n)if pathlib.Path(n).is_absolute()else ROOT/n)==h,"protected_history_changed")
 require(sha(ROOT/before["old_proposal_path"])==before["old_proposal_sha256"],"old_pending_proposal_changed")
 require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()==before["branch"],"branch_changed")
 require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==before["head"],"head_changed")
 subprocess.run(["git","diff","--check"],cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
 candidate_check=read(HERE/"current-candidate-check.json")
 require(candidate_check["source_sha256"]==sha(HERE/"current_candidate_check.py")and
         candidate_check["schedule_sha256"]==sha(ROOT/"scripts/baseline/capacity_algebraic_schedule.py"),"candidate_check_changed")
 require(candidate_check["compiler_calls"]==candidate_check["encrypted_calls"]==candidate_check["provider_calls"]==0,"candidate_check_scope")
 controls=read(HERE/"controls-finalcurrent.json")
 require(controls["status"]=="passed"and controls["tests_run"]==357 and controls["failed"]==controls["errors"]==controls["skipped"]==0,"current_controls")
 for n,h in controls["tested_sources"].items():require(sha(ROOT/"scripts/baseline"/n)==h,"control_source_changed")
 cli=read(ROOT/"scripts/baseline/benchmarks/tools/source_manifest_v1/controls-v21-compatibility.json")
 require(cli["status"]=="passed"and cli["tests_run"]==59 and cli["failed"]==cli["errors"]==cli["skipped"]==0,"cli_controls")
 require(cli["source_sha256"]==sha(ROOT/"scripts/prepare_pytorch_source.py"),"cli_binding")
 meta=read(HERE/"cli-v21-metadata-check.json")
 require(meta["status"]=="passed"and len(meta["checked_fixtures"])==10 and
         meta["source_executed"]is False and meta["default_source_format"]=="poseidon-trusted-pytorch-v18","metadata_controls")
 for r in meta["checked_fixtures"]:require(r["manifest_sha256"]==sha(HERE/"fixtures"/r["case"]/"manifest.json"),"metadata_fixture_changed")
 phases=[];plans=[read(RESULTS/n/"plan.json")for n in NAMES]
 for index,name in enumerate(NAMES):
  out=RESULTS/name;plan=plans[index];report=read(out/"report.json")
  require(plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"}),"plan_binding")
  require(report["binding"]==digest({k:v for k,v in report.items()if k!="binding"}),"report_binding")
  require(report["plan_binding"]==plan["binding"]and report["status"]in("passed","failed"),"phase_terminal")
  require(plan["deadline_wall"]==plan["created_wall"]+plan["scope"]["wall_seconds"]and
          report["seconds"]<=plan["scope"]["wall_seconds"],"phase_budget")
  require(report["provider_calls"]==report["http_attempts"]==0 and report["new_agent_generation"]is False,"offline_origin")
  require(report["tolerance"]==dict(atol=1e-5,rtol=1e-4),"frozen_tolerance")
  changed=[]
  for n,h in plan["sources"].items():
   if sha(ROOT/n)!=h:
    require(index<2 and n in ["scripts/baseline/capacity_algebraic_schedule.py",
     "scripts/baseline/test_capacity_neutral_factors.py","scripts/baseline/test_capacity_nested_masks.py"],"unexpected_historical_source_change")
    changed.append(dict(path=n,historical_sha256=h,current_sha256=sha(ROOT/n)))
    if n.endswith("capacity_algebraic_schedule.py"):
     saved=HERE/"source-before"/("capacity_algebraic_schedule.py"if index==0 else"capacity_algebraic_schedule-mask-v3.py")
     require(sha(saved)==h,"historical_scheduler_snapshot")
  for n,h in plan["fixtures"].items():require(sha(ROOT/n)==h,"fixture_changed")
  for c,h in plan["controls_sha256"].items():require(sha(HERE/("controls-"+c+".json"))==h,"old_controls_changed")
  if index:require(plan["parent_report_sha256"]==sha(RESULTS/NAMES[0]/"report.json"),"parent_history_changed")
  packages=[];rows=[]
  for r in report["rows"]:
   require(r["status"]in("passed","failed"),"case_terminal")
   case=out/r["case"];require(sha(case/"rule_candidate.py")==r["candidate_sha256"],"candidate_changed")
   validation=read(out/"tasks"/r["task_id"]/"validation-0.json")
   require(validation["paid_calls"]==0 and validation["new_agent_generation"]is False,"validation_origin")
   if r["status"]=="passed":
    require(r["source_captured"]and r["compiled"]and r["encrypted_execution"]and r["reuse_passed"],"case_acceptance")
    pkg=pathlib.Path(r["package"]);m=read(pkg/"manifest.json")
    require(m["binding"]==r["package_binding"]and m["validation"]["level"]=="numerical"and
            m["validation"]["encrypted_execution"]is True and m["agent_generated"]is False,"package_scope")
    for n,v in m["files"].items():require(sha(pkg/n)==(v if type(v)is str else v["sha256"]),"package_artifact_changed")
    require({"artifacts/lowered.earth.mlir","artifacts/lowered.ckks.mlir","artifacts/operator-spec.json",
             "artifacts/bound/plan.json","artifacts/bound/bundle/manifest.json","artifacts/validation.json"}<=set(m["files"]),"pipeline_artifacts")
    require(r["reuse_compiler_calls"]==r["reuse_agent_calls"]==0,"reuse_calls")
    comparison=read(case/"new-input-comparison.json");require(comparison["passed"]and comparison["compared_values"]==r["reuse_values"],"reuse_numerical")
    execution=read(case/"new-input-execution.json")
    require(execution["status"]=="executed"and execution["encrypted_execution"]is True and
            execution["agent_calls"]==execution["compiler_calls"]==0 and execution["executor_secret_key"]is False,"reuse_execution_scope")
    packages.append(dict(case=r["case"],files=len(m["files"]),manifest_sha256=sha(pkg/"manifest.json"),binding=m["binding"]))
   rows.append(dict(case=r["case"],status=r["status"],compiled=r["compiled"],encrypted_execution=r["encrypted_execution"],
                    reuse_passed=r["reuse_passed"],candidate_sha256=r["candidate_sha256"]))
  endpoint=set(plans[index+1]["existing_results"])if index+1<len(plans)else{p.name for p in RESULTS.iterdir()}
  added=endpoint-set(plan["existing_results"]);new_bytes=sum(size(RESULTS/n)for n in added)
  require(new_bytes<=plan["scope"]["new_bytes"],"phase_artifact_budget")
  phases.append(dict(name=name,report_sha256=sha(out/"report.json"),report_binding=report["binding"],plan_binding=plan["binding"],
                    status=report["status"],passed=report["passed"],failed=report["failed"],sources=len(plan["sources"]),
                    historical_source_differences=changed,fixtures=len(plan["fixtures"]),packages=packages,rows=rows,new_artifact_bytes=new_bytes,
                    **{k:report[k]for k in ["source_captures","compilations","encrypted_calls","input_groups","compared_values","maximum_absolute_error","seconds"]}))
 require(phases[-1]["status"]=="passed"and phases[-1]["passed"]==4 and phases[-1]["failed"]==0,"current_acceptance")
 selected={}
 for row in candidate_check["rows"]:
  if row["source_unchanged"]and row["backend_identity_unchanged"]:selected[row["case"]]=row
 require(len(selected)==10,"current_candidate_compatibility_gap")
 protected=before["protected_sources"]
 actual=size(RESULTS);reserve=shutil.disk_usage(RESULTS).free
 require(actual<=32*1024**3 and reserve>=4*1024**3,"global_storage_boundary")
 result=dict(format="poseidon-exact-arithmetic-source-final-audit-v1",status="passed",phases=phases,
  protected_sources_checked=len(protected),protected_sources_changed=0,historical_requests_checked=len(before["historical_request_files"]),
  old_proposal_preserved=True,old_proposal_status="superseded_pending",final_sources_checked=len(plans[-1]["sources"]),
  unique_models=10,new_v21_models=8,legacy_models=2,current_candidate_matches_by_historical_phase=selected,
  current_static_tests=416,failed=0,errors=0,skipped=0,provider_calls=0,http_attempts=0,
  all_result_bytes=actual,reserve_bytes=reserve,new_artifact_bytes=sum(p["new_artifact_bytes"]for p in phases),
  tolerance=dict(atol=1e-5,rtol=1e-4),backend="poseidon_cpu",observed_wall=time.time(),
  compiler_calls=sum(p["compilations"]for p in phases),encrypted_calls=sum(p["encrypted_calls"]for p in phases),
  input_groups=sum(p["input_groups"]for p in phases),compared_values=sum(p["compared_values"]for p in phases),
  maximum_absolute_error=max(p["maximum_absolute_error"]for p in phases))
 result["binding"]=digest(result)
 return result
if __name__=="__main__":
 result=audit()
 with(HERE/"final-audit.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({k:result[k]for k in["status","unique_models","current_static_tests","protected_sources_checked","all_result_bytes",
  "new_artifact_bytes","compiler_calls","encrypted_calls","input_groups","compared_values","maximum_absolute_error","provider_calls"]}))
