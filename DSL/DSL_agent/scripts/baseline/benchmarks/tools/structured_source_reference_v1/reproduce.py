"""Readonly original-source call reproductions; old tasks are never executed/relabelled."""
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
import numpy as np
from source_precision_reference import run,source_identity,SourceReferenceError
from benchmark_torch import mathematical
from application_backend import PoseidonBackend
OUT=q.RESULTS/"agent-native-call-reproduction-r1634"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest();frozen=source_identity()
flat=q.read(T/"tensor_contraction_live_v1/live-plan.json")
cases=["dictionary0","nested0","keywords0","defaults1","defaults2","flat_matrix"]
scope=dict(format="poseidon-native-call-reproduction-scope-v1",cases=cases,
 authorization="human_current_plan_all_tests_approved",source_identity=frozen,sdk=sdk["manifest_sha256"],
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 max_native_workers=6,api_calls=0,compiler_calls=0,encrypted_calls=0,
 budget=dict(wall_seconds=600,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3))
scope["binding"]=q.digest(scope);q.write(HERE/"reproduction-scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(root):return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
rows=[]
for name in cases:
 folder=OUT/name;folder.mkdir(mode=0o700)
 old=Path(flat["campaign"])/"matrix"if name=="flat_matrix"else q.RESULTS/"agent-structured-input-live-v1"/name
 task=q.read(old/"task-id.json")["task_id"];parent=old/"tasks"/task
 before=hashes(parent);metadata=q.read(parent/"task.json");spec=q.read(parent/"source/manifest.json")
 request=metadata["request"];model=request["model"]
 tree=metadata["model_provenance"]["decomposition"]["pytorch_capture"]["output_structure"]
 rng=np.random.default_rng(2026101434)
 samples=[{i["name"]:rng.uniform(-.4,.4,i["shape"]).tolist()for i in model["inputs"]}for _ in range(3)]
 q.write(folder/"inputs.json",samples)
 try:
  report=run(parent/"source/manifest.json",model,samples,folder/"native",
      expected_manifest_sha256=q.digest(spec),input_dtypes={i["name"]:"float64"for i in model["inputs"]},
      output_structure=tree,resource_profile="general-tensors-v1",trusted=True)
  worst=0.;values=0
  for sample,got in zip(samples,report["outputs"]):
   expected=mathematical(metadata["model_provenance"]["original"],sample,ideal=True)
   for item in model["outputs"]:
    n=item["name"];a=np.asarray(got[n],dtype=np.float64).reshape(-1);b=np.asarray(expected[n],dtype=np.float64).reshape(-1)
    q.require(a.shape==b.shape,"oracle_shape");worst=max(worst,float(np.max(np.abs(a-b))));values+=a.size
  row=dict(case=name,status="evaluated",mathematical_match=worst<=1e-12,maximum_absolute_error=worst,
           compared_values=values,native_binding=report["binding"])
 except SourceReferenceError as error:
  row=dict(case=name,status="rejected",code=error.code,mathematical_match=False)
 q.require(hashes(parent)==before and source_identity()==frozen and PoseidonBackend().identity_manifest()==sdk,"frozen_and_original")
 rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True)
report=dict(format="poseidon-native-call-reproduction-v1",scope_binding=scope["binding"],cases=rows,
 source_identity=frozen,sdk=scope["sdk"],api_calls=0,compiler_calls=0,encrypted_calls=0,
 original_tasks_unchanged=True)
report["binding"]=q.digest(report);q.write(HERE/"reproduction.json",report);q.write(OUT/"report.json",report)
print(json.dumps(dict(binding=report["binding"],cases=len(rows),rejected=sum(r["status"]=="rejected"for r in rows),
 mismatched=sum(r["status"]=="evaluated"and not r["mathematical_match"]for r in rows))),flush=True)
