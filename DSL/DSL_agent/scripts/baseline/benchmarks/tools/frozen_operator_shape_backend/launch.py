import sys,json,time,shlex
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"scripts/baseline"))
from hecate_python_env import enter_nix,VENV
pre=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-operator-shape-integration-r985")
scope=json.loads((pre/"approved-backend-scope-r992.json").read_text())
assert scope["approval_answer"]=="approved"
assert not (pre/"job-r992.log").exists()
assert not (pre.parent/"agent-operator-shape-backend-r992").exists()
start=time.monotonic()
prefix='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONPATH='+shlex.quote(str(Path.cwd()/"scripts/baseline"))+' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.quote(str(VENV/"bin/python"))+' -B '
command=prefix+shlex.quote(str(Path(__file__).with_name("job.py")))+' > '+shlex.quote(str(pre/"job-r992.log"))+' 2>&1'
code=enter_nix(command,seconds=max(1,min(7200,int(scope["deadline_wall"]-time.time()))),environment="poseidon-cpu")
terminal=dict(exit_code=code,seconds=time.monotonic()-start,paid_calls=0,single_nix_environment=True)
(pre/"launch-terminal-r992.json").write_text(json.dumps(terminal,indent=2)+"\n")
print(json.dumps(terminal));raise SystemExit(code)
