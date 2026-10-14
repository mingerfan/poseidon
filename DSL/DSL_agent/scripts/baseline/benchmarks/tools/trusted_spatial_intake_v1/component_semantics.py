"""Spatial semantic unit regression under the pinned local runtime."""
import sys,shlex,unittest,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix,WORK
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
      [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
sys.path.insert(0,str(HERE/"component-stage"))
from benchmark_graph import digest
from pytorch_ordered_contract import source_identity
out=WORK/"results"/("agent-spatial-source-semantics-stage-"+str(time.time_ns()));out.mkdir()
suite=unittest.defaultTestLoader.loadTestsFromName("test_pytorch_spatial_source")
with(out/"tests.log").open("x")as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(format="poseidon-spatial-source-semantics-v1",result=str(out),tests=result.testsRun,
    failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    source_identity=source_identity(),api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items()if k!="source_identity"}),flush=True)
raise SystemExit(0 if result.wasSuccessful()else 1)
