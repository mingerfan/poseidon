"""Prototype FX semantics/public-state/alias and provenance rejection controls."""
from pathlib import Path
import sys,shlex,json,copy,time
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];TOOLS=BASE/"benchmarks/tools"
sys.path[:0]=[str(HERE),str(BASE),str(TOOLS/"static_broadcast_live_v1"),str(TOOLS/"gather_composition_live_v1")]
import run as live,qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
    live.no_active()
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import torch,numpy as np
from torch.fx.passes.shape_prop import ShapeProp
from pytorch_static_trace import ALIAS_POLICY
from contraction_fx_normalization import normalize,check,metadata,arguments,public_arrays,POLICY
from platform_config import require_python_packages
require_python_packages(torch,np)
class Model(torch.nn.Module):
    def __init__(self,dims,tensor=False,dtype=torch.int64):
        super().__init__()
        if tensor:self.register_buffer("dimensions",torch.tensor(dims,dtype=dtype))
        else:self.dimensions=dims
    def forward(self,a,b):
        return torch.tensordot(a,b,dims=self.dimensions)
def prepared(module,a,b):
    traced=torch.fx.symbolic_trace(module)
    ShapeProp(traced).propagate(a,b)
    for node in traced.graph.nodes:
        meta=node.meta.get("tensor_meta")
        if node.op!="output"and meta is not None and meta.dtype==torch.float64:
            node.meta["poseidon_static_tensor"]=dict(policy=ALIAS_POLICY,shape=list(meta.shape),dtype="float64")
    traced.meta["tensor_contraction_source_policy"]=POLICY
    return traced
cases=q.read(TOOLS/"tensor_contraction_component_v1/report.json")["cases"]
cases=[dict(c,tensor_dimensions=False)for c in cases]+[
    dict(case="tensor_count",input_shapes=[[2,3],[3,2]],dims=1,tensor_dimensions=True),
    dict(case="tensor_axis_pair",input_shapes=[[2,3,2],[2,3]],dims=[[-1,1],[0,1]],tensor_dimensions=True),
    dict(case="tensor_outer",input_shapes=[[2],[2]],dims=0,tensor_dimensions=True)]
started=time.monotonic();rows=[];checks=0
for ordinal,c in enumerate(cases):
    ash,bsh=c["input_shapes"];rng=np.random.default_rng(2026101400+ordinal)
    a=torch.from_numpy(np.asarray(rng.uniform(-.3,.3,ash),dtype=np.float64).reshape(ash))
    b=torch.from_numpy(np.asarray(rng.uniform(-.3,.3,bsh),dtype=np.float64).reshape(bsh))
    model=Model(c["dims"],c["tensor_dimensions"])
    traced=prepared(model,a,b);inputs=[dict(name="a",shape=ash),dict(name="b",shape=bsh)]
    arrays=public_arrays(traced);before={k:v.copy()for k,v in arrays.items()};graph=str(traced.graph)
    normalized,record=normalize(traced,inputs,resource_profile="general-tensors-v1")
    assert len(record["entries"])==1
    expected=model(a,b);actual=normalized(a,b)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert str(traced.graph)==graph
    for key in before:np.testing.assert_array_equal(public_arrays(traced)[key],before[key])
    check(record,inputs,arrays,resource_profile="general-tensors-v1")
    saved_a,saved_b=a.clone(),b.clone()
    actual.add_(.25)
    torch.testing.assert_close(a,saved_a,rtol=0,atol=0);torch.testing.assert_close(b,saved_b,rtol=0,atol=0)
    alter=copy.deepcopy(record);alter["entries"][0]["plan"]["matrix_shapes"][0][0]=True
    alter["binding"]=q.digest({k:v for k,v in alter.items()if k!="binding"})
    try:check(alter,inputs,arrays,resource_profile="general-tensors-v1")
    except ValueError:pass
    else:raise AssertionError("rebound_provenance_type_confusion")
    alter=copy.deepcopy(record);alter["normalized_inventory"][0]["name"]="changed"
    alter["binding"]=q.digest({k:v for k,v in alter.items()if k!="binding"})
    try:check(alter,inputs,arrays,resource_profile="general-tensors-v1")
    except ValueError:pass
    else:raise AssertionError("rebound_inventory")
    rows.append(dict(case=c["case"],status="passed",tensor_dimensions=c["tensor_dimensions"],record_binding=record["binding"],
        output_shape=list(expected.shape),reference_groups=1,public_state_preserved=True,independent_result_storage=True))
    checks+=1
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError):return
    raise AssertionError("expected_rejection")
reject(lambda:arguments((0,0,1),dict(dims=1)))
reject(lambda:arguments((0,),{}))
reject(lambda:arguments((0,0),dict(unknown=1)))
reject(lambda:arguments((0,0),dict(out=torch.zeros(1))))
reject(lambda:metadata("call_function",torch.tensordot,(torch.empty(2),torch.empty(2),torch.tensor(1)),
    {},resolve_value=lambda x:x,resolve_public=lambda x:torch.tensor(1.),resource_profile="general-tensors-v1"))
reject(lambda:metadata("call_function",torch.tensordot,(torch.empty(2),torch.empty(2),torch.tensor(1)),
    {},resolve_value=lambda x:x,resolve_public=lambda x:None,resource_profile="general-tensors-v1"))
# Metadata inference operates only on static shapes, with no actual model evaluation.
meta=metadata("call_function",torch.tensordot,(torch.empty(2,1,device="meta",dtype=torch.float64),
    torch.empty(3,2,device="meta",dtype=torch.float64),1),{},resolve_value=lambda x:x,
    resolve_public=lambda x:None,resource_profile="general-tensors-v1")
assert tuple(meta.shape)==(2,2)and meta.device.type=="meta"
report=dict(format="poseidon-contraction-fx-prototype-controls-v1",status="passed",cases=rows,
    passed=len(rows),failed=0,skipped=0,rejection_cases=6,tamper_checks=2*len(rows),
    metadata_only_check=True,seconds=time.monotonic()-started,
    source_files={p.name:q.sha(p)for p in HERE.glob("*.py")},production_active=False,
    source_capture_validated=False,api_calls=0,compiler_calls=0,encrypted_calls=0)
report["binding"]=q.digest(report);q.write(HERE/"fx-controls.json",report)
print(json.dumps({k:v for k,v in report.items()if k not in("cases","source_files")}))
