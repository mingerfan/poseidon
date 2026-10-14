"""Static acceptance only: system Python and the existing locked NumPy."""
import hashlib,io,json,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/"targets"),str(HERE),str(ROOT/"scripts/baseline"),
 str(ROOT/"scripts/baseline/benchmarks/tools/static_alias_source_draft_v1"),
 "/home/lhohy/poseidon-work/platforms/aarch64-linux/venvs/hecate-2.0.1-cpu/lib/python3.10/site-packages"]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
import numpy as np
import test_composition as tests
started=time.monotonic()
output=HERE/"acceptance-r1127.json"
if output.exists():raise SystemExit("Fresh acceptance path required")
freeze=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-unstarted-continuation-r1113/plan.json")
p=json.loads(freeze.read_text())
for name,h in p["source_hashes"].items():
 if sha(ROOT/name)!=h:raise SystemExit("Live production freeze changed")
before={str(path.relative_to(ROOT)):sha(path) for path in HERE.rglob("*.py")}
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
(HERE/"test-output-r1127.log").write_text(stream.getvalue())
for name,h in p["source_hashes"].items():
 if sha(ROOT/name)!=h:raise SystemExit("Production freeze changed during acceptance")
report=dict(format="poseidon-staged-alias-composition-acceptance-v2",
 status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,failed=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
 runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch_loaded="torch" in sys.modules),
 source_hashes=before,origin=json.loads((HERE/"origin.json").read_text()),
 contexts=tests.MEASUREMENTS,production_freeze_checks=len(p["source_hashes"]),
 production_intake_integrated=False,real_tracing=0,compilations=0,encrypted_executions=0,
 paid_calls=0,new_dependencies=0,backend_changed=False,
 scope="Isolated Agent alias/registered-operator composition. Independent ordered NumPy vs high-level and lowered references; not PyTorch capture or FHE.")
output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","runtime","production_freeze_checks")}))
if not result.wasSuccessful():print(stream.getvalue());raise SystemExit(1)
