"""Current strict report protocol; preserve prior producer bindings."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
from source_calibration import run,check_report,implementation
from source_calibration_contract import FORMAT
from test_model_output_contract import model,tensor
from application_backend import PoseidonBackend
old=q.read(HERE/"acceptance.json");q.bound(old)
controls=q.read(HERE/"controls-final.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==69 and old["passed"]==12,"controls_and_native")
O=q.RESULTS/"agent-fx-calibration-acceptance-r1630"
OUT=q.RESULTS/"agent-fx-calibration-current-r1631"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
def hashes(root):
 return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
before=hashes(O);OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest();frozen=implementation()
q.require(sdk["manifest_sha256"]==controls["sdk"]==old["sdk"],"sdk")
scope=dict(format="poseidon-fx-current-scope-v1",old_acceptance_binding=old["binding"],
 controls_binding=controls["binding"],implementation=frozen,sdk=old["sdk"],
 authorization="human_current_plan_all_tests_approved",max_native_workers=2,
 api_calls=0,compiler_calls=0,encrypted_calls=0,
 budget=dict(wall_seconds=600,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
scope["binding"]=q.digest(scope);q.write(HERE/"current-scope.json",scope);q.write(OUT/"scope.json",scope)
historical=[]
producer=dict(frozen);producer["source_calibration.py"]=q.sha(HERE/"before-closed-report-fields.py")
for path in sorted(O.glob("*/inventory/calibration.json"))+sorted(O.glob("*/observation/calibration.json"))+sorted(O.glob("*/observation_*/calibration.json")):
 report=q.read(path);request=q.read(path.parent/"request.json")
 q.require(report["implementation"]==request["implementation"]==producer,"historical_producer_identity")
 check_report(report,request);historical.append(dict(path=str(path),binding=report["binding"]))
q.require(len(historical)==27,"historical_report_count")
parent=O/"gelu_affine";request=q.read(parent/"observation/request.json")
spec=q.read(parent/"source/manifest.json")
graph=model(shape=tuple(spec["inputs"][0]["shape"]),profile="general-tensors-v1")
selections=[
 ("module_v1",dict(format=FORMAT,purpose="calibration",selectors=[
  dict(id="original_input",module="",argument=0,projection=dict(kind="tensor"),domain=None)])),
 ("fx_v1",request["selection"])]
start=time.monotonic();rows=[]
for name,selection in selections:
 report=run(parent/"source/manifest.json",graph,request["samples"],OUT/name,
  expected_manifest_sha256=q.digest(spec),selection=selection,expected_selection_sha256=q.digest(selection),
  input_dtypes=request["input_dtypes"],output_structure=tensor(),trusted=True)
 q.require(report["status"]=="observed"and report["forward_output_sha256"]==q.read(parent/"observation/calibration.json")["forward_output_sha256"],
  "same_original_forward_outputs")
 q.require(implementation()==frozen and PoseidonBackend().identity_manifest()==sdk,"current_source_frozen")
 row=dict(case=name,status="passed",samples=3,native_workers=1,report_binding=report["binding"])
 rows.append(row);print(json.dumps(row),flush=True)
q.require(hashes(O)==before,"historical_native_evidence_unchanged")
report=dict(format="poseidon-fx-current-acceptance-v1",scope_binding=scope["binding"],
 status="passed",passed=2,failed=0,skipped=0,new_native_workers=2,samples=6,
 rechecked_historical_reports=27,historical_native_evidence_unchanged=True,
 old_acceptance_binding=old["binding"],controls_binding=controls["binding"],
 implementation=frozen,sdk=old["sdk"],api_calls=0,compiler_calls=0,encrypted_calls=0,
 cases=rows,historical_reports=historical,seconds=time.monotonic()-start)
report["binding"]=q.digest(report);q.write(HERE/"current-acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k not in("cases","historical_reports","implementation")}),flush=True)
