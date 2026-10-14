"""Relevant regression scope and frozen historical request checks; offline only."""
import hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux"
os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def inside():
 from component_contract import reconstruct_request
 from public_weight_context import request_digest
 from unified_request_storage import compact_request
 from application_identity import current_identity_manifest
 started=time.monotonic()
 names=("test_unified_tensor_axis_guidance","test_binding_feedback",
        "test_unified_poseidon_guidance","test_unified_generation_guidance",
        "test_unified_expression_guidance","test_application_diagnostics",
        "test_earth_failure_diagnostic","test_static_work_feedback",
        "test_validation_adapter","test_agent_component")
 loader=unittest.defaultTestLoader
 suite=unittest.TestSuite(loader.loadTestsFromName(n)for n in names)
 with (HERE/"controls.log").open("x")as log:
  result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 report=dict(format="poseidon-tensor-axis-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
  tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
  failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  modules=list(names),provider_calls=0,encrypted_execution=False,
  seconds=time.monotonic()-started)
 old=json.loads((ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1330.json").read_text())
 history=[]
 for row in old["cases"]:
  p=Path(row["request_file"]);before=p.read_bytes()
  actual=hashlib.sha256(before).hexdigest()
  assert actual==row["request_file_sha256"],"historical_request_bytes_changed"
  request=json.loads(before)
  assert reconstruct_request(request)==request,"historical_request_reconstruction"
  assert request["request_id"]==row["request_id"],"historical_request_binding"
  assert p.read_bytes()==before,"historical_request_changed_during_check"
  history.append(dict(case=row["id"],request_id=request["request_id"],sha256=actual,unchanged=True))
 report["historical_requests"]=history
 identity=current_identity_manifest()
 report["identity_sha256"]=identity["manifest_sha256"]
 report["source_stage_counts"]={s:len(identity[s]["sources"])for s in ("generation","compilation","execution","validation")}
 report["binding"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 (HERE/"controls.final.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({k:report[k]for k in ("status","tests","passed","failed","errors","skipped","seconds","source_stage_counts")}),flush=True)
 if not result.wasSuccessful():
  for test,trace in result.failures+result.errors:print(str(test)+"\n"+trace[-3000:],flush=True)
 return 0 if report["status"]=="passed"else 1
def main():
 if sys.argv[1:]==["--inside"]:return inside()
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(cmd,seconds=180,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(main())
