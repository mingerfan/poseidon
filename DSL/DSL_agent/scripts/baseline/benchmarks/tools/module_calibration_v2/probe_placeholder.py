"""Readonly probes locate where original module metadata is lost."""
from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=120,environment="poseidon-cpu"))
import torch
from test_source_module_calibration import Nested
from pytorch_input_binding import plan,signature_record,normalize as normalize_inputs
from pytorch_static_trace import trace,ALIAS_POLICY
from pytorch_arithmetic_normalization import clone_public_state,normalize_common,normalize
from pytorch_approximation import module_paths
def nodes(m):
 return [dict(name=n.name,target=str(n.target),paths=module_paths(n),meta_keys=list(n.meta))
   for n in m.graph.nodes if n.op=="call_function"and "gelu"in str(n.target)]
m=Nested().eval();m.block.forward=lambda x:torch.nn.functional.gelu(x)
inputs=[dict(name="x",shape=[2])]
spec=dict(inputs=inputs,call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
call_plan=plan(spec,signature_record(m))
baseline=torch.fx.symbolic_trace(m)
fx=trace(m,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY,input_plan=call_plan)
fx,input_flow=normalize_inputs(fx,call_plan)
cloned=clone_public_state(fx)
common,_=normalize_common(fx,inputs,resource_profile="general-tensors-v1",ordered_effects=True)
normalized,_=normalize(common,resource_profile="general-tensors-v1",preserve_unused_state=True,allow_ordered_mutations=True)
retraced=trace(normalized,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
result={label:dict(gelu=nodes(stage),graph=str(stage.graph))for label,stage in(("original",baseline),("static",fx),("cloned",cloned),("common",common),("normalized",normalized),("retraced",retraced))}
import qualification_scope as q
result["binding"]=q.digest(result);q.write(HERE/"metadata-placeholder-probe.json",result)
print(json.dumps(result))
