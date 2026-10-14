"""Bounded plaintext/decomposition draft check; no tracing, API or FHE."""
from pathlib import Path
import sys,json,hashlib,shlex,subprocess,time,re
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path[:0]=[str(HERE),str(BASE)]
if sys.argv[1:]!=["--inside"]:
 sys.path.insert(0,str(BASE/"benchmarks/tools/static_broadcast_live_v1"))
 import run as live
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONPATH='+str(HERE)+':scripts/baseline OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from platform_config import require_python_packages
require_python_packages()
import torch,numpy as np
assert torch.version.cuda is None
started=time.monotonic()
with(HERE/"pinned-tests.log").open("x")as stream:
 code=subprocess.run([sys.executable,"-B","-m","unittest","test_plan","test_pinned","-v"],
 stdout=stream,stderr=subprocess.STDOUT,timeout=120).returncode
output=(HERE/"pinned-tests.log").read_text();print(output[-3300:])
record=dict(format="poseidon-tensor-gather-draft-plaintext-v1",status="passed"if code==0 else"failed",
 returncode=code,python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
 tests=int(re.search(r"Ran (\d+) tests",output).group(1)),seconds=time.monotonic()-started,
 mathematical_and_pinned_torch_contexts=6,core_decomposition_contexts=6,
 production_active=False,source_capture_validated=False,api_calls=0,compiler_calls=0,encrypted_calls=0,
 sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob("*.py")},
 log_sha256=hashlib.sha256(output.encode()).hexdigest())
record["binding"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
with(HERE/"pinned-tests.json").open("x")as stream:json.dump(record,stream,sort_keys=True,indent=2)
raise SystemExit(code)
