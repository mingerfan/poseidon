"""Read-only final evidence audit for the v20 scalar partition source acceptance."""
import hashlib,json,pathlib,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OUT=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-scalar-partition-v20-source-acceptance-v1")
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def require(ok,code):
 if not ok:raise ValueError(code)
def size(p):return int(subprocess.check_output(["du","-sb",str(p)],text=True,timeout=20).split()[0])
def audit():
 old=read(HERE/"before.json");plan=read(OUT/"plan.json");report=read(OUT/"report.json")
 require(report["status"]=="passed"and report["passed"]==8 and report["failed"]==0,"terminal_acceptance")
 require(plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"}),"plan_binding")
 require(report["binding"]==digest({k:v for k,v in report.items()if k!="binding"}),"report_binding")
 for key in ["protected_sources","historical_request_files"]:
  for name,expected in old[key].items():
   p=pathlib.Path(name)if pathlib.Path(name).is_absolute()else ROOT/name
   require(sha(p)==expected,"protected_source_or_history_changed")
 require(sha(ROOT/old["old_proposal_path"])==old["old_proposal_sha256"],"historical_proposal_changed")
 for name,expected in {**plan["sources"],**plan["fixtures"]}.items():
  require(sha(ROOT/name)==expected,"frozen_source_changed")
 packages=[];evidence_dirs=set()
 for row in report["rows"]:
  require(row["source_captured"]and row["compiled"]and row["encrypted_execution"]and row["reuse_passed"],"case_scope")
  pkg=pathlib.Path(row["package"]);m=read(pkg/"manifest.json")
  require(m["binding"]==row["package_binding"]and m["validation"]["level"]=="numerical"and
   m["validation"]["encrypted_execution"]is True and m["agent_generated"]is False,"package_scope")
  for name,item in m["files"].items():
   expected=item if type(item)is str else item["sha256"]
   require(sha(pkg/name)==expected,"package_artifact_changed")
  names=set(m["files"])
  require({"artifacts/lowered.earth.mlir","artifacts/lowered.ckks.mlir","artifacts/operator-spec.json",
   "artifacts/bound/plan.json","artifacts/bound/bundle/manifest.json","artifacts/validation.json"}<=names,"pipeline_artifacts")
  require(row["reuse_compiler_calls"]==row["reuse_agent_calls"]==0,"reuse_calls")
  validation=read(OUT/"tasks"/row["task_id"]/"validation-0.json")
  def paths(v):
   if type(v)is dict:
    for child in v.values():yield from paths(child)
   elif type(v)is list:
    for child in v:yield from paths(child)
   elif type(v)is str and "/results/candidate-replay-"in v:
    yield pathlib.Path(v.split("/results/candidate-replay-")[0]+"/results/candidate-replay-"+v.split("/results/candidate-replay-")[1].split("/")[0])
  evidence_dirs.update(paths(validation))
  packages.append(dict(case=row["case"],manifest_sha256=sha(pkg/"manifest.json"),binding=m["binding"],files=len(m["files"])))
 require(plan["deadline_wall"]==plan["created_wall"]+900 and report["seconds"]<=900,"original_batch_budget")
 for name,count in[("finalnew",39),("compatibility",186)]:
  control=read(HERE/("controls-"+name+".json"))
  require(control["status"]=="passed"and control["tests_run"]==count and
          control["failed"]==control["errors"]==control["skipped"]==0,"controls")
  require(sha(HERE/("controls-"+name+".json"))==plan["controls_sha256"][name],"controls_binding")
 cli=read(ROOT/"scripts/baseline/benchmarks/tools/source_manifest_v1/controls-v20-compatibility.json")
 require(cli["status"]=="passed"and cli["tests_run"]==59 and cli["failed"]==cli["errors"]==cli["skipped"]==0,"cli_controls")
 require(cli["source_sha256"]==sha(ROOT/"scripts/prepare_pytorch_source.py"),"cli_source_binding")
 check=read(HERE/"cli-v20-metadata-check.json")
 require(check["status"]=="passed"and len(check["checked_fixtures"])==8 and
         check["source_executed"]is False and check["default_source_format"]=="poseidon-trusted-pytorch-v18","v20_cli_metadata")
 for row in check["checked_fixtures"]:
  require(row["manifest_sha256"]==sha(HERE/"fixtures"/row["case"]/"manifest.json"),"cli_fixture_binding")
 added=[p for p in OUT.parent.iterdir()if p.name not in plan["existing_results"]]
 result=dict(format="poseidon-scalar-partition-v20-final-audit-v1",status="passed",packages=packages,
  protected_sources_checked=len(old["protected_sources"]),historical_requests_checked=len(old["historical_request_files"]),
  old_proposal_preserved=True,old_proposal_status="superseded_pending",
  frozen_sources_checked=len(plan["sources"]),frozen_fixtures_checked=len(plan["fixtures"]),
  new_artifact_bytes=sum(size(p)for p in added),all_result_bytes=size(OUT.parent),evidence_directories=len(evidence_dirs),
  report_sha256=sha(OUT/"report.json"),report_binding=report["binding"],paid_calls=0,
  source_format="explicit v20 plus legacy v19/v18",observed_wall=time.time())
 require(result["new_artifact_bytes"]<=plan["scope"]["new_bytes"]and
  result["all_result_bytes"]<=plan["scope"]["global_bytes"],"artifact_budget")
 result["binding"]=digest(result)
 return result
if __name__=="__main__":
 v=audit()
 with(HERE/"final-audit.json").open("x")as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({k:v[k]for k in ["status","protected_sources_checked","historical_requests_checked",
  "frozen_sources_checked","new_artifact_bytes","all_result_bytes","evidence_directories"]}))
