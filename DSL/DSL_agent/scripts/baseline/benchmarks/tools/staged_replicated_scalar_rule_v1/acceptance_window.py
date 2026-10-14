"""One pinned offline control/backend window; no Provider calls."""
import hashlib, importlib.util, json, os, subprocess, sys, time, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import WORK
from benchmark_graph import canonical, digest, require
controls=WORK/"results/agent-mha-replicated-scalar-controls-v1"
backend=WORK/"results/agent-mha-replicated-scalar-backend-v1"
capture=WORK/"results/agent-mha-ordered-capture-v2"
require(not controls.exists() and not backend.exists(), "Preserve previous evidence")
controls.mkdir()
paths=[HERE/"replicated_scalar_lowering.py",HERE/"tests/test_replicated_scalar_lowering.py",HERE/"backend_acceptance.py",Path(__file__)]
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
spec=importlib.util.spec_from_file_location("test_replicated_scalar_lowering",paths[1])
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
started=time.monotonic()
with (controls/"controls.log").open("w") as log:
 result=unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
report=dict(format="poseidon-replicated-scalar-controls-v1",status="passed" if result.wasSuccessful() else "failed",
 ran=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 files=frozen,elapsed_seconds=time.monotonic()-started,
 log_sha256=hashlib.sha256((controls/"controls.log").read_bytes()).hexdigest(),
 compiler_run=False,encrypted_execution=False,api_calls=0,production_applied=False)
report["binding"]=digest(report)
(controls/"report.json").write_bytes(canonical(report))
print(json.dumps(report,sort_keys=True),flush=True)
if not result.wasSuccessful():
 print((controls/"controls.log").read_text(),flush=True)
 raise SystemExit(1)
require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items()),"Control source drift")
command=[sys.executable,"-B",str(HERE/"backend_acceptance.py"),str(backend),str(capture)]
with (controls/"backend.log").open("w") as log:
 completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1250)
print((controls/"backend.log").read_text(),flush=True)
print("backend_exit",completed.returncode,flush=True)
raise SystemExit(completed.returncode)
