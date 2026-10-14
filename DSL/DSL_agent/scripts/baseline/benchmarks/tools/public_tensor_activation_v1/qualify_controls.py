"""Current control suite and byte-identity timing; no model/API/backend execution."""
import argparse,hashlib,io,json,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import VENV,enter_nix
def inside():
 from pytorch_ordered_contract import source_identity,_dependency_names
 names=["test_pytorch_public_boolean_arithmetic","test_pytorch_public_tensor_division",
        "test_pytorch_dependency_syntax","test_pytorch_exact_arithmetic","test_pytorch_source_contract","test_application_capabilities"]
 stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
 (HERE/"controls-current-v1.log").write_text(stream.getvalue())
 timings=[];identities=[]
 _dependency_names.cache_clear()
 for _ in range(5):
  start=time.perf_counter();identities.append(source_identity());timings.append(time.perf_counter()-start)
 assert all(v==identities[0]for v in identities)
 from application_identity import current_identity_manifest
 sdk=current_identity_manifest()
 data=dict(tests=r.testsRun,failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
  current_source_files=len(identities[0]),source_identity_seconds=timings,cache=_dependency_names.cache_info()._asdict(),
  sdk_identity=sdk["manifest_sha256"],current_hashing_retained=True,compiler_calls=0,encrypted_calls=0,provider_calls=0)
 (HERE/"controls-current-v1.json").write_text(json.dumps(data,indent=2))
 print(json.dumps(data));raise SystemExit(0 if r.wasSuccessful()and not r.skipped else 1)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
 if a.inside:inside()
 else:
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  raise SystemExit(enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu"))
