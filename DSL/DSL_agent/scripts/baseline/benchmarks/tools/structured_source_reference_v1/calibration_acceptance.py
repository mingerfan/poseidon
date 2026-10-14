"""Shared original-call reconstruction through module calibration."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
import numpy as np
from source_calibration import run,implementation
from source_calibration_contract import FORMAT
from application_backend import PoseidonBackend
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed","controls")
OUT=q.RESULTS/"agent-shared-call-calibration-r1639"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+64*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+64*1024**2,"storage")
OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest();frozen=implementation()
scope=dict(format="poseidon-shared-call-calibration-scope-v1",implementation=frozen,sdk=sdk["manifest_sha256"],
 cases=[dict(id="dictionary0",module="linear",argument=0,input_name="x"),
        dict(id="keywords0",module="",argument="other",input_name="y")],
 controls_binding=controls["binding"],authorization="human_current_plan_all_tests_approved",
 max_native_workers=2,api_calls=0,compiler_calls=0,encrypted_calls=0,
 budget=dict(wall_seconds=300,new_bytes=64*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
scope["binding"]=q.digest(scope);q.write(HERE/"calibration-scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(root):return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
rows=[]
for case in scope["cases"]:
 name=case["id"];old=q.RESULTS/"agent-structured-input-live-v1"/name
 task=q.read(old/"task-id.json")["task_id"];parent=old/"tasks"/task
 before=hashes(parent);meta=q.read(parent/"task.json");spec=q.read(parent/"source/manifest.json")
 samples=q.read(q.RESULTS/"agent-native-structured-reference-r1637"/name/"inputs.json")
 selector=dict(format=FORMAT,purpose="calibration",selectors=[
  dict(id="observed",module=case["module"],argument=case["argument"],projection=dict(kind="tensor"),domain=None)])
 report=run(parent/"source/manifest.json",meta["request"]["model"],samples,OUT/name,
  expected_manifest_sha256=q.digest(spec),selection=selector,expected_selection_sha256=q.digest(selector),
  input_dtypes={i["name"]:"float64"for i in spec["inputs"]},
  output_structure=meta["model_provenance"]["decomposition"]["pytorch_capture"]["output_structure"],
  trusted=True)
 values=np.concatenate([np.asarray(x[case["input_name"]],dtype=np.float64).reshape(-1)for x in samples])
 stats=report["observations"]["observed"]
 q.require(stats["minimum"]==float(values.min())and stats["maximum"]==float(values.max())and
  stats["values"]==values.size,"same_named_tensor_routing")
 q.require(hashes(parent)==before and implementation()==frozen and PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 row=dict(case=name,status="passed",samples=3,values=int(values.size),report_binding=report["binding"],native_workers=1)
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(format="poseidon-shared-call-calibration-acceptance-v1",scope_binding=scope["binding"],
 passed=2,failed=0,skipped=0,new_native_workers=2,samples=6,cases=rows,sdk=sdk["manifest_sha256"],
 api_calls=0,compiler_calls=0,encrypted_calls=0)
report["binding"]=q.digest(report);q.write(HERE/"calibration-report.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
