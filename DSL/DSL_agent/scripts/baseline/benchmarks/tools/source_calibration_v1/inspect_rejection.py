from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];sys.path[:0]=[str(B)]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 raise SystemExit(enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=60,environment="poseidon-cpu"))
from source_precision_reference import prepare_source,checked_samples
from test_model_output_contract import model,tensor
path=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-calibration-acceptance-r1620/gelu_inputs_f32/source/manifest.json")
spec=json.loads(path.read_bytes())
try:
 value,files=prepare_source(path);print("source_prepare: passed")
except Exception as e:print("source_prepare:",type(e).__name__,str(e)[:200])
try:
 checked_samples(spec,model(shape=(2,),profile="general-tensors-v1"),[dict(x=[-.7,.3])],
  {"x":"float32"},tensor(),"general-tensors-v1");print("sample_check: passed")
except Exception as e:print("sample_check:",type(e).__name__,str(e)[:200])
