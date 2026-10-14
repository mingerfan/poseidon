"""Bounded qualification of the explicit V22 application intake; no Provider."""
import argparse,copy,hashlib,json,os,sys,time,unittest,io,subprocess,shutil
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from benchmark_graph import require,digest
from hecate_python_env import WORK,VENV,enter_nix
def inside():
 from pytorch_ordered_contract import source_identity,encode_weight_files
 from pytorch_model_intake import run,package_format
 from pytorch_weight_package import export_state,read
 from candidate_bundle import check_provenance
 import fixture_factory,torch
 output=WORK/"results/agent-public-tensor-v22-source-v2";require(not output.exists(),"Fresh source results")
 output.mkdir();start=time.time();deadline=start+590
 def storage():
  used=sum(p.stat().st_size for p in output.rglob("*")if p.is_file())
  global_used=max(int(subprocess.check_output(["du",arg,"-s","-B1",str(WORK/"results")],text=True,timeout=30).split()[0])
                  for arg in ("--apparent-size","--count-links"))
  require(used<=128*1024**2 and global_used<=32*1024**3 and shutil.disk_usage(WORK).free>=4*1024**3,
          "Source storage boundary")
  return dict(source_bytes=used,global_bytes=global_used,free_bytes=shutil.disk_usage(WORK).free)
 storage_before=storage()
 cases=json.loads((HERE/"source-cases.json").read_text())
 for name in ("boolean_only","ordinary"):
  target=HERE/"generated-fixtures-v2"/name;target.mkdir(parents=True)
  weights=export_state(getattr(fixture_factory,name)(),target,resource_profile="general-tensors-v1")
  raw=(HERE/"fixture_factory.py").read_bytes();(target/"model.py").write_bytes(raw)
  spec=dict(format="poseidon-trusted-pytorch-v22",files={"model.py":hashlib.sha256(raw).hexdigest()},
   entry="model."+name,inputs=[dict(name="x",shape=[2,3])],outputs=["output"],
   approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
   factory=dict(args=[],kwargs={}),dtype_policy="strict-float64-v1",weights=weights)
  (target/"manifest.json").write_text(json.dumps(spec,sort_keys=True,indent=2))
  cases.append(dict(case=name,manifest=str(target/"manifest.json"),original_manifest=None))
 frozen=source_identity();plan=dict(format="poseidon-public-tensor-v22-source-plan-v1",source_hashes=frozen,
  cases=cases,source_seconds=590,source_disk_bytes=128*1024**2,paid_calls=0,backend_changes=0)
 plan["binding"]=digest(plan);(output/"plan.json").write_text(json.dumps(plan,sort_keys=True,indent=2))
 results=[];checks=[]
 names=["test_pytorch_public_boolean_arithmetic","test_pytorch_public_tensor_division",
        "test_pytorch_exact_arithmetic","test_pytorch_source_contract","test_application_capabilities"]
 stream=io.StringIO();rr=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
 (output/"control-tests.log").write_text(stream.getvalue())
 controls=dict(tests=rr.testsRun,failed=len(rr.failures),errors=len(rr.errors),skipped=len(rr.skipped))
 print("CONTROLS",json.dumps(controls),flush=True)
 for item in cases:
  require(time.time()<deadline and source_identity()==frozen,"Source time or integrity boundary")
  name=item["case"];case=output/name
  try:
   result=run(item["manifest"],case,resource_profile="general-tensors-v1",trusted=True)
   spec=json.loads(Path(item["manifest"]).read_text())
   high=json.loads((case/"original.json").read_text());model=json.loads((case/"model.json").read_text())
   bound=json.loads((case/"decomposition.json").read_text());ref=json.loads((case/"reference-report.json").read_text())
   arrays,raw_files=read(Path(item["manifest"]).parent,spec["weights"],resource_profile="general-tensors-v1")
   files={n:Path(item["manifest"]).parent.joinpath(n).read_text()for n in spec["files"]}
   provenance=dict(original=high,decomposition=bound,reference_report=ref,trusted_pytorch=dict(
    format=package_format(spec),manifest=spec,files=files,
    weight_files=encode_weight_files(spec,raw_files,resource_profile="general-tensors-v1")))
   check_provenance(provenance,model)
   (case/"model-provenance.json").write_text(json.dumps(provenance,allow_nan=False))
   # Reload is data-only; original source files are not imported in the caller.
   check_provenance(json.loads((case/"model-provenance.json").read_text()),model)
   altered=copy.deepcopy(provenance);record=altered["reference_report"]["capture_metadata"]["public_tensor_arithmetic"]
   record["original_public_state"][next(iter(record["original_public_state"]))]["sha256"]="0"*64
   record["binding"]=digest({k:v for k,v in record.items()if k!="binding"})
   try:check_provenance(altered,model)
   except (ValueError,KeyError):checks.append(dict(case=name,check="raw_state_tamper",status="passed"))
   else:raise ValueError("Raw state tampering was accepted")
   results.append(dict(case=name,status="passed",source=result,provenance_sha256=digest(provenance)))
  except Exception as error:results.append(dict(case=name,status="failed",error=type(error).__name__,reason=str(error)))
  print("CASE",name,results[-1]["status"],results[-1].get("reason",""),flush=True)
  require(source_identity()==frozen,"Source implementation changed")
  storage()
 # The old scalar-only contract must keep rejecting the unchanged tensor divisor.
 old=HERE/"fixtures/float32_vector"/"legacy-v21.json"
 spec=json.loads((old.parent/"manifest.json").read_text());spec["format"]="poseidon-trusted-pytorch-v21"
 old.write_text(json.dumps(spec,sort_keys=True,indent=2))
 try:run(old,output/"old-v21-rejection",resource_profile="general-tensors-v1",trusted=True)
 except ValueError as error:
  log=(output/"old-v21-rejection/capture.log").read_text()
  require("Public scalar divisor dtype and shape"in log,"Old refusal was at an unexpected layer")
  checks.append(dict(case="float32_vector",check="legacy_v21_tensor_refusal",status="passed"))
 else:raise ValueError("Old V21 behavior changed")
 report=dict(format="poseidon-public-tensor-v22-source-report-v1",plan_binding=plan["binding"],
  source_identity=frozen,cases=results,controls=controls,extra_checks=checks,
  passed=sum(x["status"]=="passed"for x in results),failed=sum(x["status"]=="failed"for x in results),
  compiler_calls=0,encrypted_calls=0,paid_calls=0,network_configuration_changes=0,
  seconds=time.time()-start,storage_before=storage_before,storage_after=storage())
 report["binding"]=digest(report);(output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2))
 print("REPORT",json.dumps({k:v for k,v in report.items()if k not in ("source_identity","cases","extra_checks")}),flush=True)
 return 0 if report["failed"]==0 and not any(controls[k]for k in ("failed","errors","skipped")) else 1
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");args=p.parse_args()
 if args.inside:raise SystemExit(inside())
 import shlex
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=640,keep_env=(),environment="poseidon-cpu"))
