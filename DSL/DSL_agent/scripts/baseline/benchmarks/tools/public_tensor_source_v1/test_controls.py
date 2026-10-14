"""Fresh controls for the captured source draft, preserving all earlier records."""
import importlib.util,io,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE));os.environ.update(POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
def inside():
 from benchmark_graph import digest
 from public_tensor_division import sha
 from hecate_python_env import VENV
 import source_tests
 assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
 old=HERE.parent/"public_tensor_division_v1/test_division.py"
 spec=importlib.util.spec_from_file_location("retained_division_controls",old)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module),unittest.defaultTestLoader.loadTestsFromModule(source_tests)])
 path=HERE/"controls-source-v2.json";log=HERE/"controls-source-v2.log"
 assert not path.exists()and not log.exists()
 started=time.monotonic();stream=io.StringIO()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 log.write_text(stream.getvalue())
 record=dict(format="poseidon-public-tensor-source-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
  tests_run=result.testsRun,passed=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
  failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
  original_source_captures_reused=6,production_activation=False,provider_calls=0,compiler_calls=0,encrypted_calls=0,
  draft_tools={p.name:sha(p.read_bytes())for p in HERE.glob("*.py")},retained_controls_sha256=sha(old.read_bytes()),
  retained_controls_on_staging_implementation=True)
 record["binding"]=digest(record)
 with path.open("x")as f:json.dump(record,f,allow_nan=False,indent=2)
 print(json.dumps({k:v for k,v in record.items()if k not in("draft_tools","retained_controls_sha256")}),flush=True)
 return 0 if record["status"]=="passed"else 1
def outer():
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(cmd,seconds=180,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
