"""Pinned, bounded provenance/reference controls. No API or backend execution."""
import os,sys,shlex,unittest,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
if(HERE/"capture-controls-v3.json").exists():raise SystemExit("preserve previous flow controls")
start=time.monotonic();suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_public_capture.py")
with(HERE/"capture-controls-v3.log").open("x")as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
import torch,numpy as np,test_public_capture
data=dict(format="poseidon-public-expression-capture-controls-v1",tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
          counters=test_public_capture.COUNTERS,runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda),
          seconds=time.monotonic()-start,provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,production_intake_connected=True,
          sources={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(BASE/"pytorch_public_expression_capture.py",BASE/"pytorch_public_arithmetic_capture.py",BASE/"pytorch_ordered_contract.py",BASE/"pytorch_ordered_intake.py",BASE/"pytorch_model_intake.py",BASE/"candidate_bundle.py",BASE/"pytorch_static_trace.py",BASE/"pytorch_public_expression_metadata.py",BASE/"pytorch_public_expression_flow.py",BASE/"pytorch_public_value_ops.py",HERE/"test_public_capture.py",HERE/"capture_controls.py")})
(HERE/"capture-controls-v3.json").write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
data['models']=test_public_capture.ROWS
(HERE/'capture-models-v3.json').write_text(json.dumps(test_public_capture.ROWS,sort_keys=True,indent=2)+'\n')
print(json.dumps(data));raise SystemExit(0 if result.wasSuccessful()else 1)
