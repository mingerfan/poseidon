"""One bounded pinned-environment draft check; no backend or Provider calls."""
import json,os,shlex,sys,unittest,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"scripts/baseline";sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=60,keep_env=(),environment="poseidon-cpu"))
start=time.time();suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_public_values.py")
with (HERE/"controls-v2.log").open("w")as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
result=dict(format="poseidon-public-value-draft-controls-v1",tests=r.testsRun,failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
 seconds=time.time()-start,production_integrated=False,provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
 planned_contexts=12,planned_reference_groups=48,validated_contexts=sys.modules["test_public_values"].COUNTERS["contexts"],reference_groups=sys.modules["test_public_values"].COUNTERS["reference_groups"],source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob("*.py")},reference="real CPU float64 Torch compared with independent NumPy public-value normalization")
(HERE/"controls-v2.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result));raise SystemExit(0 if r.wasSuccessful()else 1)
