"""Current exact offline scope launcher. No approval means no environment entry."""
import hashlib,json,shlex,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
pre=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-pytorch-offline-prepared-r1040")
proposal_path=ROOT/"docs/baseline/agent-common-pytorch-offline-proposal-r1076.json"
receipt=pre/"approved-backend-scope-r1077.json"
if not receipt.is_file():raise RuntimeError("Current offline backend scope requires human approval")
proposal=json.loads(proposal_path.read_text());scope=json.loads(receipt.read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if hashlib.sha256(canonical({k:v for k,v in proposal.items() if k!="binding"})).hexdigest()!=proposal["binding"]:
    raise RuntimeError("Proposal binding changed")
if scope["approval_answer"]!="approved" or scope["proposal_binding"]!=proposal["binding"]:
    raise RuntimeError("Approval binding mismatch")
for section in ("source_hashes","extra_source_hashes","case_hashes","tool_hashes"):
    for name,h in proposal[section].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path)!=h:raise RuntimeError("Frozen offline input changed")
for name,value in proposal["scope"].items():
    if scope.get(name)!=value:raise RuntimeError("Approved scope differs from fixed proposal")
if scope["deadline_wall"]<=time.time():raise RuntimeError("Approved offline deadline exhausted")
if (pre/"job-r1077.log").exists() or (pre.parent/"agent-common-pytorch-backend-r1077").exists():
    raise RuntimeError("Preserve existing run; explicit outcome audit required")
import sys
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import enter_nix,VENV
started=time.monotonic()
prefix='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONPATH='+shlex.quote(str(ROOT/"scripts/baseline"))+' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.quote(str(VENV/"bin/python"))+' -B '
command=prefix+shlex.quote(str(Path(__file__).with_name("job.py")))+" > "+shlex.quote(str(pre/"job-r1077.log"))+" 2>&1"
code=enter_nix(command,seconds=max(1,min(7200,int(scope["deadline_wall"]-time.time()))),environment="poseidon-cpu")
(pre/"launch-terminal-r1077.json").write_text(json.dumps(dict(exit_code=code,seconds=time.monotonic()-started,paid_calls=0,single_nix_environment=True),indent=2)+"\n")
raise SystemExit(code)
