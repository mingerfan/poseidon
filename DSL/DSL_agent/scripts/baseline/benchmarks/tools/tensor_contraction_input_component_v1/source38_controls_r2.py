"""Relevant v38/legacy controls using the existing pinned tracing environment."""
from pathlib import Path
import sys,shlex,subprocess,time,json,re
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];TOOLS=BASE/"benchmarks/tools"
sys.path[:0]=[str(BASE),str(TOOLS/"static_broadcast_live_v1"),str(TOOLS/"gather_composition_live_v1")]
import run as live,qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
    live.no_active()
    from application_backend import PoseidonBackend
    q.write(HERE/"identity-before-controls-r2.json",PoseidonBackend().identity_manifest())
    command='PYTHONPATH=scripts/baseline LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
started=time.monotonic()
modules=["test_tensor_contraction_source","test_tensor_gather_source","test_gather_capabilities",
 "test_application_capabilities","test_provenance_json_transport","test_pytorch_model_intake",
 "test_pytorch_source_contract","test_logical_view_normalization","test_source_cli_package_policy",
 "test_model_source_provenance","test_application_provenance_transport"]
with(HERE/"source38-controls-r2.log").open("x")as stream:
    child_result=subprocess.run([sys.executable,"-B","-m","unittest",*modules,"-v"],
        stdout=stream,stderr=subprocess.STDOUT,timeout=240)
text=(HERE/"source38-controls-r2.log").read_text();match=re.search(r"Ran (\d+) tests",text)
record=dict(format="poseidon-source38-compatibility-controls-v1",status="passed"if child_result.returncode==0 else"failed",
    returncode=child_result.returncode,tests=int(match.group(1))if match else None,modules=modules,
    seconds=time.monotonic()-started,log_sha256=q.sha(HERE/"source38-controls-r2.log"),
    api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=q.digest(record);q.write(HERE/"source38-controls-r2.json",record)
print(json.dumps(record));print(text[-5000:])
raise SystemExit(child_result.returncode)
