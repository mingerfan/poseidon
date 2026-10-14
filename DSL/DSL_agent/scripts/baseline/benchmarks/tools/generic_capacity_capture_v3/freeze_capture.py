"""Freeze current source identity without rewriting the original captures."""
from pathlib import Path
import sys,json,hashlib,copy,shlex
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=90,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from benchmark_graph import digest
old=BASE/"benchmarks/tools/generic_capacity_intake_v1"
p=json.loads((old/"capture-plan.json").read_text());identity=PoseidonBackend().identity_manifest()
assert identity["backend"]==json.loads((old/"identity-snapshot.json").read_text())["backend"]
new=copy.deepcopy(p);new.pop("binding")
new["sdk"]=identity["manifest_sha256"];new["sdk_sources"]={}
for stage in ("generation","compilation","execution","validation"):
 for section in ("sources","configuration_files"):
  for name,want in identity[stage][section].items():
   path=BASE/name if section=="sources"else ROOT/name
   assert hashlib.sha256(path.read_bytes()).hexdigest()==want
   new["sdk_sources"][str(path.relative_to(ROOT))]=want
new["campaign"]=str(Path(p["campaign"]).parent/"agent-generic-capacity-capture-r1493")
new["budget"]["per_case_bytes"]=64*1024**2
new["capture_tool_sha256"]=hashlib.sha256((HERE/"capture.py").read_bytes()).hexdigest()
new["prior_capture_bindings"]=[
 json.loads((Path(p["campaign"])/"report.json").read_text())["binding"],
 json.loads((Path(p["campaign"]).parent/"agent-generic-capacity-capture-r1490/report.json").read_text())["binding"]]
new["reason"]="Opt-in public metadata policy changes authenticated source-intake closure; recapture unchanged sources under current identity, retain all old references and failures."
new["binding"]=digest(new)
for path,data in ((HERE/"identity-snapshot.json",identity),(HERE/"capture-plan.json",new),
 (ROOT/"docs/baseline/agent-generic-capacity-reference-rebind-r1493.json",new)):
 with path.open("x")as f:json.dump(data,f,sort_keys=True,indent=2)
print(json.dumps(dict(binding=new["binding"],sdk=new["sdk"],cases=len(new["cases"]))))
