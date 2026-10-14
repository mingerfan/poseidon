"""Three-context trusted source validation with exact logical coordinate proofs."""
import json,sys,time,shlex,shutil,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2]
sys.path[:0]=[str(B/"benchmarks/tools/generic_capacity_backend_v10"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from application_backend import PoseidonBackend
from hecate_python_env import VENV,enter_nix
def verify():
 p=q.read(HERE/"capture-scope.json");q.bound(p)
 for path,want in p["frozen_files"].items():q.require(q.sha(path)==want,"capture_source:"+path)
 q.require(PoseidonBackend().identity_manifest()==q.read(HERE/"identity.json"),"sdk")
 return p
def save(path,value):
 value=dict(value);value["binding"]=q.digest(value);q.write(path,value);return value
def inside():
 from pytorch_model_intake import run
 from candidate_bundle import load_model_provenance
 from benchmark_graph import validate
 p=verify();out=Path(p["campaign"]);started=time.monotonic();rows=[]
 def boundary():
  verify();q.require(time.monotonic()-started<p["budget"]["wall_seconds"],"wall")
  q.require(live.measure(out)<=p["budget"]["new_artifact_bytes"]and
    live.measure(live.RESULTS)<=p["budget"]["global_result_bytes"],"storage")
  q.require(shutil.disk_usage(out).free>=p["budget"]["reserve_bytes"],"disk_reserve")
 try:
  for c in p["cases"]:
   boundary();folder=out/c["id"];result=run(c["manifest"],folder,resource_profile="general-tensors-v1",trusted=True)
   model=q.read(folder/"model.json");report=q.read(folder/"reference-report.json");binding=q.read(folder/"decomposition.json")
   record=report["capture_metadata"]["public_value_expressions"]
   proof=record["logical_view_normalization"];q.bound(proof)
   q.require(record["source_intake_version"]=="poseidon-trusted-pytorch-v35"and proof["production_active"]is True and
    proof["normalized_sha256"]==q.digest(model)and proof["original_sha256"]==q.digest(record["base_capture"]["lowered_graph"]),
    "logical_view_authority")
   q.require(record["logical_view_reference"]["groups"]==16 and report["finite_reference"]["lowered_sha256"]==q.digest(model),"new_reference")
   before=q.read(c["old_graph"]);q.require(model["inputs"]==before["inputs"]and model["constants"]==before["constants"]and
     model["outputs"]==before["outputs"],"frozen_math_declarations")
   for metric in ("max_reference_tolerance_ratio","max_normalization_tolerance_ratio"):
    q.require(report["finite_reference"][metric]<=1,"math_torch_comparison")
   try:
    provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
    intake=dict(status="accepted",canonical_bytes=len(q.canonical(provenance)),provenance_sha256=q.digest(provenance))
   except ValueError as e:
    intake=dict(status="blocked",diagnostic=str(e)[:200],layer="source_provenance")
   row=dict(case=c["id"],status="passed",source_format="poseidon-trusted-pytorch-v35",capture=result,proof_binding=proof["binding"],
    original_nodes=proof["original_nodes"],normalized_nodes=proof["normalized_nodes"],model_sha256=q.digest(model),
    source_intake=intake,groups=16,values=report["finite_reference"]["compared_values"],
    ratio=report["finite_reference"]["max_reference_tolerance_ratio"],old_manifest_sha256=c["old_manifest_sha256"])
   rows.append(row);save(out/(c["id"]+"-receipt.json"),row);print(json.dumps(row),flush=True)
  boundary()
 except BaseException as e:
  save(out/"report.json",dict(status="stopped",scope_binding=p["binding"],cases=rows,passed=len(rows),failed=1,
   not_run=len(p["cases"])-len(rows)-1,phase="source_capture",error_type=type(e).__name__,diagnostic=str(e)[:240],
   provider_calls=0,compiler_calls=0,encrypted_calls=0));raise
 report=save(out/"report.json",dict(format="poseidon-logical-view-source-reference-v1",status="passed",scope_binding=p["binding"],
  sdk=p["sdk"],cases=rows,passed=len(rows),failed=0,not_run=0,skipped=0,
  input_groups=sum(r["groups"]for r in rows),output_values=sum(r["values"]for r in rows),
  source_intake_accepted=sum(r["source_intake"]["status"]=="accepted"for r in rows),
  provider_calls=0,compiler_calls=0,encrypted_calls=0,wall_seconds=time.monotonic()-started))
 print(json.dumps({k:v for k,v in report.items()if k!="cases"}));return 0
def outer():
 p=verify();live.no_active();out=Path(p["campaign"]);q.require(not out.exists(),"fresh_campaign")
 q.require(live.measure(live.RESULTS)+p["budget"]["new_artifact_bytes"]<=p["budget"]["global_result_bytes"]and
  shutil.disk_usage(live.RESULTS).free>=p["budget"]["reserve_bytes"]+p["budget"]["new_artifact_bytes"],"initial_storage")
 out.mkdir(mode=0o700)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 tree=live.OwnedTree().start()
 try:code=enter_nix(command,seconds=p["budget"]["wall_seconds"],environment="poseidon-cpu")
 finally:background=tree.finish()
 q.require(not background,"owned_background");verify()
 save(out/"terminal.json",dict(exit_code=code,finished=time.time(),scope_binding=p["binding"]))
 return code
if __name__=="__main__":
 if sys.argv[1:]==["--execute"]:raise SystemExit(outer())
 if sys.argv[1:]==["--inside"]:raise SystemExit(inside())
 raise SystemExit("capture.py --execute")
