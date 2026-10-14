"""Capacity, weight-privacy and frozen execution guards."""
import sys,json,unittest,io,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(HERE),str(BASE),str(BASE/"application_planning/tests"),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
if "--inside"not in sys.argv:
 import run as live
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import os
os.environ["PYTHONPATH"]=str(BASE)
suite=unittest.TestSuite()
for name in ("test_public_origin_columns","test_unified_capacity_extended","test_unified_capacity_wide","test_public_weight_context","test_unified_output_aliases","test_qualification_scope"):
 suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
output=io.StringIO();r=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
record=dict(format="poseidon-generic-capacity-controls-v1",status="passed"if r.wasSuccessful()and not r.skipped else"failed",
 tests=r.testsRun,failures=[(str(a),b)for a,b in r.failures],errors=[(str(a),b)for a,b in r.errors],skipped=[(str(a),b)for a,b in r.skipped],
 compiler_calls=0,encrypted_calls=0,provider_calls=0)
with (HERE/"controls.json").open("x")as f:json.dump(record,f,sort_keys=True,indent=2)
print(json.dumps({k:record[k]for k in("status","tests","failures","errors","skipped")}))
if record["status"]!="passed":print(output.getvalue())
raise SystemExit(record["status"]!="passed")
