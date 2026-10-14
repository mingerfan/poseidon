"""Wait for the active frozen offline batch; then enter one sequential live environment."""
import json,hashlib,sys,time,shlex
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
pre=work/"agent-common-pytorch-offline-prepared-r1040"
scope=json.loads((pre/"approved-backend-scope-r1082.json").read_text())
terminal=pre/"reuse-continuation-terminal-r1091.json"
while not terminal.is_file():
 if time.time()>scope["deadline_wall"]+30:raise RuntimeError("Original offline deadline exhausted; retain all evidence")
 time.sleep(5)
outcome=json.loads(terminal.read_text())
report=json.loads((work/"agent-common-pytorch-reuse-continuation-r1091/report.json").read_text())
if report["status"]!="complete" or outcome["audit_exit_code"]!=0:
 raise RuntimeError("Offline integrity/acceptance audit incomplete; no API dispatched")
from hecate_python_env import VENV,enter_nix
prep=ROOT/"scripts/baseline/benchmarks/tools/approved_pending_live_prepare_v2/prepare.py"
execute=Path(__file__).with_name("execute.py")
log=pre/"approved-pending-chain-r1094.log"
if log.exists():raise RuntimeError("Existing chain log; never auto-resume")
prefix='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.quote(str(VENV/"bin/python"))+' -B '
command='( '+prefix+shlex.quote(str(prep))+' && '+prefix+shlex.quote(str(execute))+' ) > '+shlex.quote(str(log))+' 2>&1'
code=enter_nix(command,seconds=130200,environment="poseidon-cpu")
with (pre/"approved-pending-chain-terminal-r1094.json").open("x") as f:
 json.dump(dict(exit_code=code,source='Direct human approval of accumulated pending scopes',paid_calls='See immutable per-task ledgers',single_sequential_nix_environment=True),f,indent=2)
raise SystemExit(code)
