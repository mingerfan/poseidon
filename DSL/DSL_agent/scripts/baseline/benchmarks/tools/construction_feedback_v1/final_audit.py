"""Post-document identity and protected-source verification; no backend/API calls."""
import hashlib,json,os,shlex,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=90,keep_env=(),environment="poseidon-cpu")
 from application_identity import current_identity_manifest
 controls=json.loads((HERE/"controls.json").read_text());baseline=json.loads((HERE/"baseline.json").read_text())
 identity=current_identity_manifest();assert identity==controls["identity"]
 assert all(sha(ROOT/n)==h for n,h in baseline["protected_sources"].items())
 assert all(sha(n)==h for n,h in baseline["historical_requests"].items())
 root=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-construction-guidance-acceptance-v1")
 plan=json.loads((root/"plan.json").read_text())
 assert all(sha(ROOT/n)==h for n,h in plan["sources"].items())
 assert all(sha(n)==h for n,h in plan["historical_files"].items())
 s=json.loads((ROOT/"docs/baseline/agent-construction-guidance-r1337.json").read_text())
 assert s["native_report_sha256"]==sha(root/"report.json")
 assert s["binding"]==hashlib.sha256(json.dumps({k:v for k,v in s.items() if k!="binding"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
 status=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).splitlines()
 r=dict(status="passed",identity_sha256=identity["manifest_sha256"],protected_preserved=len(baseline["protected_sources"]),
  historical_requests_preserved=len(baseline["historical_requests"]),active_source_binding_preserved=True,
  summary_sha256=sha(ROOT/"docs/baseline/agent-construction-guidance-r1337.json"),
  public_markdown_sha256=sha(ROOT/"docs/baseline/agent-construction-guidance-r1337.md"),
  branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
  head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  git_modified=sum(x[:2]!="??" for x in status),git_untracked=sum(x[:2]=="??" for x in status),
  api_calls=0,compiler_calls=0,encrypted_calls=0)
 assert r["branch"]=="lhy-agent-dsl"
 with (HERE/"final-audit.json").open("x") as f:json.dump(r,f,sort_keys=True,indent=2)
 print(json.dumps(r))
 return 0
if __name__=="__main__":raise SystemExit(main())
