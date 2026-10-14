"""Affected sandbox inventory, closure and resource regression."""
import sys,json,shlex,time,io,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
if "--inside" not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from qualification_scope import write
loader=unittest.defaultTestLoader;suite=loader.loadTestsFromName("test_poseidon_sandbox_policy")
stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
record=dict(status="passed" if r.wasSuccessful() else "failed",tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),paid_calls=0)
write(HERE/"sandbox-controls.json",record)
(HERE/"sandbox-controls.log").write_text(stream.getvalue())
print(json.dumps(record))
if not r.wasSuccessful():print(stream.getvalue())
raise SystemExit(0 if r.wasSuccessful() else 1)
