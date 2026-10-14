import sys,shlex,json,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import enter_nix,VENV
 command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=60,environment="poseidon-cpu"))
from candidate_bundle import load_model_provenance,check_provenance,storage_limit
from application_model_source import prepare
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
import hecate_python_env
p=next((hecate_python_env.WORK/"results").glob("agent-static-broadcast-prepare-*/expand_as__arithmetic/tasks/*/model-import"))
model=json.loads((p/"model.json").read_text())
try:
 provenance=load_model_provenance(p,model,max_bytes=4*1024**2);print("provenance passed")
 options=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
 options=preparation_options(provenance,model,options);options=graph_options(provenance["original"],model,options)
 print("options",options)
 q,ctx=prepare(model,"poseidon-cpu","numerical",options)
 print("request prepared",q["request_id"],storage_limit(q))
 check_provenance(provenance,q["model"]);print("request provenance checked")
except Exception:traceback.print_exc(limit=15);raise
