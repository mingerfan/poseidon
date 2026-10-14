"""Fresh isolated source capture under the new source identity; original models remain frozen."""
import sys,pathlib,os,json,hashlib,time,shutil
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import WORK
from pytorch_model_intake import run
from benchmark_graph import canonical,digest
from application_backend import PoseidonBackend
PROPOSAL=json.loads((ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json").read_bytes())
OUT=WORK/"results/agent-public-math-roles-capture-v2"
def inventory(p):return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
def save(x):
 y=dict(x,binding=digest(x));(OUT/"report.json").write_bytes(canonical(y))
def main():
 assert not OUT.exists()
 start=time.monotonic();backend=PoseidonBackend();identity=backend.identity_manifest()
 def check():
  assert time.monotonic()-start<180,"capture batch time"
  assert backend.identity_manifest()==identity,"source drift"
  assert shutil.disk_usage(WORK).free>=4*1024**3,"free disk"
  assert inventory(WORK/"results")<=32*1024**3,"global artifact bytes"
  if OUT.exists():assert inventory(OUT)<=4*1024**3
 check();OUT.mkdir()
 report=dict(format="poseidon-public-math-role-source-recapture-v1",status="running",cases=[],api_calls=0,
  compiler_calls=0,encrypted_executions=0,stage_identity=identity,
  original_proposal_sha256=hashlib.sha256((ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json").read_bytes()).hexdigest(),
  model_or_reference_changed=False)
 save(report)
 for c in PROPOSAL["cases"]:
  check();path=ROOT/c["source_manifest"]
  assert hashlib.sha256(path.read_bytes()).hexdigest()==c["source_manifest_sha256"]
  for n,h in c["source_files"].items():assert hashlib.sha256((path.parent/n).read_bytes()).hexdigest()==h
  outcome=run(path,OUT/c["id"],resource_profile="general-tensors-v1",trusted=True)
  model=json.loads((OUT/c["id"]/"model.json").read_bytes())
  assert digest(model)==c["model_sha256"],"captured mathematical graph changed"
  report["cases"].append(dict(case=c["id"],status="passed",capture=outcome,model_sha256=digest(model),
   reference_report_sha256=hashlib.sha256((OUT/c["id"]/"reference-report.json").read_bytes()).hexdigest()))
  save(report);check()
 report.update(status="complete",passed=len(report["cases"]),failed=0,skipped=0,elapsed_seconds=time.monotonic()-start)
 save(report);print(json.dumps({k:report[k] for k in ("status","passed","failed","elapsed_seconds","api_calls","compiler_calls","encrypted_executions")}))
if __name__=="__main__":main()
