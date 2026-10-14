"""Original structured sources vs unchanged independent mathematical graphs."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=900,environment="poseidon-cpu"))
import numpy as np
from source_precision_reference import run,source_identity,check_report
from task_numerical_assessment import mathematical
from application_backend import PoseidonBackend
from numerical_results import compare
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==97,"controls")
sdk=PoseidonBackend().identity_manifest();frozen=source_identity()
q.require(sdk["manifest_sha256"]==controls["sdk"]and frozen==controls["source_identity"],"tested_identity")
OUT=q.RESULTS/"agent-native-structured-reference-r1637"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json")
names=[g+str(i)for g in ("dictionary","nested","keywords","defaults")for i in range(3)]+[
 "flat_matrix","flat_dot_scalar","flat_scalar_intermediate"]
scope=dict(format="poseidon-native-structured-reference-scope-v1",cases=names,
 authorization="human_current_plan_all_tests_approved",max_native_workers=15,
 controls_binding=controls["binding"],source_identity=frozen,sdk=sdk["manifest_sha256"],
 api_calls=0,compiler_calls=0,encrypted_calls=0,
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 budget=dict(wall_seconds=900,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3))
scope["binding"]=q.digest(scope);q.write(HERE/"scope-r3.json",scope);q.write(OUT/"scope.json",scope)
def hashes(root):return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
start=time.monotonic();rows=[]
for name in names:
 q.require(time.monotonic()-start<900,"wall_budget")
 old=Path(prior["campaign"])/name[5:]if name.startswith("flat_")else q.RESULTS/"agent-structured-input-live-v1"/name
 task=q.read(old/"task-id.json")["task_id"];parent=old/"tasks"/task;original=hashes(parent)
 metadata=q.read(parent/"task.json");spec=q.read(parent/"source/manifest.json")
 model=metadata["request"]["model"];tree=metadata["model_provenance"]["decomposition"]["pytorch_capture"]["output_structure"]
 old_inputs=q.RESULTS/"agent-native-call-reproduction-r1634"/name/"inputs.json"
 if old_inputs.exists():samples=q.read(old_inputs)
 else:
  rng=np.random.default_rng(2026101434)
  samples=[{i["name"]:rng.uniform(-.4,.4,i["shape"]).tolist()for i in model["inputs"]}for _ in range(3)]
 folder=OUT/name;folder.mkdir(mode=0o700);q.write(folder/"inputs.json",samples)
 prior_native=q.RESULTS/"agent-native-structured-reference-r1635"/name/"native"
 reused=(prior_native/"native-reference.json").exists()
 if reused:
  report=q.read(prior_native/"native-reference.json");prior_request=q.read(prior_native/"request.json")
  check_report(report,prior_request)
  q.require(prior_request["samples"]==samples and prior_request["manifest"]==spec and
            report["source_identity"]==frozen,"reusable_native_source_identity")
 else:
  report=run(parent/"source/manifest.json",model,samples,folder/"native",
   expected_manifest_sha256=q.digest(spec),input_dtypes={i["name"]:"float64"for i in model["inputs"]},
   output_structure=tree,resource_profile="general-tensors-v1",trusted=True)
 actual=[];expected=[]
 for sample,got in zip(samples,report["outputs"]):
  reference=mathematical(metadata["model_provenance"]["original"],sample,ideal=True)
  for item in model["outputs"]:
   n=item["name"];a=np.asarray(got[n],dtype=np.float64).reshape(-1);b=np.asarray(reference[n],dtype=np.float64).reshape(-1)
   q.require(a.shape==b.shape,"original_math_shape")
   actual.extend(a.tolist());expected.extend(b.tolist())
 diff=compare(np.asarray(actual).reshape(1,-1),np.asarray(expected).reshape(1,-1),1e-12,1e-12)
 q.require(diff["passed"],"unchanged_independent_math_reference")
 q.require(hashes(parent)==original and source_identity()==frozen and PoseidonBackend().identity_manifest()==sdk,"unchanged_sources_and_sdk")
 row=dict(case=name,status="passed",samples=3,reused_native_output=reused,compared_values=diff["compared_values"],
  maximum_absolute_error=diff["max_absolute_error"],reference_binding=report["binding"],
  source_state_dtypes=report["source_state_dtypes"],output_dtypes=report["output_dtypes"],
  original_source_and_math_graph_unchanged=True)
 rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=128*1024**2,"space")
report=dict(format="poseidon-native-structured-reference-acceptance-v1",scope_binding=scope["binding"],
 status="passed",passed=15,failed=0,skipped=0,samples=45,new_native_workers=sum(not r["reused_native_output"]for r in rows),
 reused_native_outputs=sum(r["reused_native_output"]for r in rows),
 compared_values=sum(r["compared_values"]for r in rows),
 maximum_absolute_error=max(r["maximum_absolute_error"]for r in rows),
 cases=rows,source_identity=frozen,sdk=sdk["manifest_sha256"],controls_binding=controls["binding"],
 api_calls=0,compiler_calls=0,encrypted_calls=0,seconds=time.monotonic()-start,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance-r3.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k not in("cases","source_identity")}),flush=True)
