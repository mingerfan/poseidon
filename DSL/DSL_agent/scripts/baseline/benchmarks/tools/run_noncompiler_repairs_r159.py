"""Non-compiler checker fixes: tests, original-response replays, and model inventory."""
import argparse,json,os,sys,shlex,subprocess,unittest,time
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
from benchmark_runner import dump
from benchmark_graph import digest,validate
from run_stage2_guidance_retest import evidence_path

def export_models(results):
 parent=results/"stage2-targeted-pipeline-diagnosis-r153.json"
 diagnosis=json.loads(parent.read_text());ids={x["id"] for x in diagnosis["rows"]}
 folder=ROOT/"docs/baseline/compiler-blocked-models-r159";folder.mkdir()
 (folder/"models").mkdir();index_path=results/"stage2-agent-campaign-v7-r130/index.json"
 index=json.loads(index_path.read_text());rows=[]
 for ref in index["shards"]:
  p=index_path.parent/ref["file"];assert sha(p)==ref["sha256"]
  for case in json.loads(p.read_text())["cases"]:
   ident=case["id"]
   if ident not in ids:continue
   m=case["model"];assert digest(m)==case["model_sha256"]
   if ident.startswith("free_supplement_"):group="affine_maxpad"
   else:
    n=int(ident.rsplit("_",1)[1])
    group="maxpad" if 32<=n<=39 else "max" if 40<=n<=47 else "relu" if 105<=n<=111 else "silu"
   file=folder/"models"/(m["id"]+".json");dump(file,m)
   item=next(x for x in diagnosis["rows"] if x["id"]==ident)
   row=dict(task_id=ident,model_id=m["id"],group=group,
      input_shapes=[x["shape"] for x in m["inputs"]],output_shapes=validate(m)["output_shapes"],
      chebyshev_degrees=[len(m["constants"][n["inputs"][1]])-1 for n in m["nodes"] if n["op"]=="polynomial"],
      negate_before_pool=any(n["op"]=="negate" for n in m["nodes"]),
      affine_preprocess={k:v for k,v in m["constants"].items() if k.startswith("context_")},
      compiler_configuration=case["compiler_configuration"],frozen_model_sha256=case["model_sha256"],
      file=str(file.relative_to(ROOT)),file_sha256=sha(file),origin_shard=str(p),origin_shard_sha256=sha(p),
      compiler_diagnostic_count=len(item["compiler_diagnostics"]),invalid_artifact_count=len(item["artifact_metadata"]),
      helper_call_required=bool(case["request"].get("upstream_helpers")),
      status="historical_compiler_capacity_or_artifact_rejection_not_reexecuted")
   rows.append(row)
 assert {x["task_id"] for x in rows}==ids and len(rows)==33
 result=sealed(dict(parent=str(parent),parent_sha256=sha(parent),index_sha256=sha(index_path),rows=rows,
   compiler_modified=False,models_modified=False,new_compiler_trials=0))
 dump(folder/"index.json",result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true");a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=1800)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from candidate_contract import strict_json,validate_candidate
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure environment")
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output required")
 a.output.mkdir();before=runtime_sources();frozen_path=RESULTS/"stage2-noncompiler-before-r159.json"
 frozen=json.loads(frozen_path.read_text())
 def unchanged():
  assert runtime_sources()==before,"Agent source drift"
  for p,h in frozen.items():assert sha(Path(p))==h,"Compiler/runtime changed: "+p
 unchanged();dump(a.output/"plan.json",sealed(dict(source_hashes=before,compiler_freeze=str(frozen_path),
  compiler_freeze_sha256=sha(frozen_path),new_paid_calls=0,native_concurrency=1,seconds=1800)))
 mods=["test_typed_witness_repairs","test_failure_repairs","test_unified_generation_guidance","test_unified_directed_guidance","test_seal_cpu_golden.SealGateTests",
  "test_unified_public_coverage","test_unified_public","test_unified_public_arithmetic","test_unified_public_unary","test_unified_public_storage","test_unified_public_views","test_unified_lambda_binding"]
 with (a.output/"tests.log").open("x") as f:res=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(mods))
 info=dict(run=res.testsRun,failed=len(res.failures),errors=len(res.errors),skipped=len(res.skipped))
 dump(a.output/"tests.json",info);print("tests",info,flush=True)
 if not res.wasSuccessful():return 1
 inventory=export_models(RESULTS);dump(a.output/"compiler-model-inventory.json",inventory)
 prior=RESULTS/"stage2-failure-repairs-r157-v3/static-review.json";review=json.loads(prior.read_text())
 checked=[];prepared=[]
 for old in review["rows"]:
  if old["status"]!="still_static_rejected":continue
  response=Path(old["response"]);assert sha(response)==old["response_sha256"]
  request_path=response.parents[1]/"request.json";assert sha(request_path)==old["request_sha256"]
  request=strict_json(request_path.read_text());candidate=strict_json(response.read_text())
  try:
   validate_candidate(candidate,request);status="static_passed";error=None
  except (ValueError,TypeError,KeyError,IndexError) as e:status="static_rejected";error=type(e).__name__+": "+str(e)
  checked.append(dict(id=old["id"],status=status,diagnostic=error,previous_diagnostic=old["diagnostic"],
    source_sha256=__import__("hashlib").sha256(candidate["hecate_source"].encode()).hexdigest(),
    original_response=str(response),original_response_sha256=sha(response),request_id=request["request_id"]))
  if status=="static_passed":prepared.append((old,request))
 assert len(checked)==88
 dump(a.output/"static-review.json",sealed(dict(parent_sha256=sha(prior),source_hashes=before,rows=checked)))
 print("newly static passed",[x[0]["id"] for x in prepared],flush=True)
 assert len(prepared)<=8,"Review unexpectedly large new executable set before admission"
 rows=[]
 for old,request in prepared:
  unchanged();ident=old["id"];work=a.output/ident;work.mkdir();dump(work/"model.json",request["model"])
  response=Path(old["response"]);dump(work/"responses.json",[response.read_text()])
  cmd=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
      "--unified-profile","public-v1","--unified-exercise",request["construction_exercise"]["id"],
      "--unified-guidance",request["generation_guidance"]["version"],"--compiler-configuration","seal-cpu-eva-w45-v1",
      "--replay",str(work/"responses.json"),"--max-repairs","0"]
  with (work/"run.log").open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
  folder=evidence_path(work/"run.log",RESULTS)
  if folder is None:raise ValueError("Missing replay evidence")
  report=json.loads((folder/"report.json").read_text());read_request(folder/"request.json",request)
  assert report["agent_calls"]==0
  row=dict(id=ident,status=report["status"],evidence=str(folder),report_sha256=sha(folder/"report.json"),
    unchanged_candidate_source=True,unchanged_request=True,new_agent_generation=False,
    failure_layer=report["attempts"][-1].get("failure_layer"),diagnostic=report["attempts"][-1].get("diagnostic"))
  if not code and report["status"]=="passed":
   row["audit"]=verify_candidate(folder);row["max_absolute_error"]=row["audit"]["comparison"]["max_absolute_error"]
  rows.append(row);dump(a.output/"progress.json",rows);print(ident,row["status"],row["diagnostic"],flush=True)
  if row["failure_layer"] in ("environment","integrity","key_setup"):raise ValueError("Infrastructure stop")
 unchanged()
 result=sealed(dict(source_hashes=before,tests=info,compiler_files_unchanged=len(frozen),compiler_modified=False,
  compiler_inventory_binding=inventory["binding"],static_rechecked=88,static_remaining=sum(x["status"]!="static_passed" for x in checked),
  rows=rows,passed=sum(x["status"]=="passed" for x in rows),failed=sum(x["status"]!="passed" for x in rows),
  new_paid_calls=0,new_agent_generation=False,parents={str(a.output/n):sha(a.output/n) for n in ["plan.json","tests.json","static-review.json","compiler-model-inventory.json"]}))
 dump(a.output/"report.json",result);print(json.dumps({k:result[k] for k in ["passed","failed","static_remaining","binding"]}),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
