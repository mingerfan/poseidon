"""Current SDK/backend identity and bounded isolated-source scope."""
import sys,shlex,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from benchmark_graph import digest
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open("x")as f:json.dump(v,f,indent=2,sort_keys=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
identity=PoseidonBackend().identity_manifest()
assert identity["backend"]==read(HERE/"prechange-identity.json")["backend"]
if(HERE/"identity-snapshot.json").exists():assert read(HERE/"identity-snapshot.json")==identity
else:write(HERE/"identity-snapshot.json",identity)
sdk={}
for key in("generation","compilation","execution","validation"):
 for section in("sources","configuration_files"):
  for name,want in identity[key][section].items():
   path=BASE/name if section=="sources"else ROOT/name
   relative=str(path.relative_to(ROOT))
   assert sha(path)==want
   assert relative not in sdk or sdk[relative]==want
   sdk[relative]=want
rows=[]
for case in read(HERE/"cases.json"):
 folder=Path(case["manifest"]).parent;spec=read(Path(case["manifest"]))
 names=["manifest.json",*spec["files"],*[v["file"]for v in spec["weights"].values()]]
 rows.append(dict(name=case["name"],manifest=case["manifest"],manifest_sha256=sha(Path(case["manifest"])),
  files={str((folder/name).relative_to(ROOT)):sha(folder/name)for name in names}))
prior=read(BASE/"benchmarks/tools/public_constant_intake_v1/capture-plan-routed.json")
plan=dict(format="poseidon-public-join-source-capture-scope-v1",
 approved_by="human_explicit_current_plan_test_approval",source_root=str(ROOT),
 results_root=str(live.RESULTS),campaign=str(live.RESULTS/"agent-public-join-capture-r1478"),
 cases=rows,sdk=identity["manifest_sha256"],sdk_sources=sdk,budget=prior["budget"],
 capture_tool_sha256=sha(HERE/"capture.py"),storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",
 original_models_or_references_modified=False,historical_records_rewritten=False,
 initial_controls_rejections="scripts/baseline/benchmarks/tools/public_join_intake_v1/probe-before.json")
plan["binding"]=digest(plan)
write(HERE/"capture-plan.json",plan)
write(ROOT/"docs/baseline/agent-public-join-capture-scope-r1478.json",plan)
print(json.dumps(dict(sdk=plan["sdk"],binding=plan["binding"],cases=len(rows),sdk_sources=len(sdk),backend_unchanged=True)))
