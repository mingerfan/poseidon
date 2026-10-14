"""Pinned Torch API observations, independent mathematics and primitive plan differential."""
from pathlib import Path
import sys,shlex,json,time,copy,itertools
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];TOOLS=BASE/"benchmarks/tools"
sys.path[:0]=[str(HERE),str(BASE),str(TOOLS/"static_broadcast_live_v1"),str(TOOLS/"gather_composition_live_v1")]
import run as live,qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
    live.no_active()
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import numpy as np,torch
from platform_config import require_python_packages
from tensor_contraction_plan import plan,check_plan
require_python_packages(torch,np)
started=time.monotonic()
cases=[
("dot",[3],[3],1),
("default_full",[2,3],[2,3],2),
("matrix",[2,3],[3,2],1),
("negative_axes",[2,3,2],[2,3],[[-1,1],[0,1]]),
("nonadjacent",[2,3,2],[3,2,2],[[-3,-1],[-2,-1]]),
("permuted_full",[2,3],[3,2],[[-2,-1],[-1,-2]]),
("outer_vectors",[2],[2],0),
("outer_matrix",[2,2],[2],[[],[]]),
("outer_singleton",[2],[1,2],0),
("broadcast_row",[2,1],[3,2],1),
("broadcast_vector",[1],[3],1),
("broadcast_axes",[2,1,2],[3,1],[[1,2],[0,1]]),
("scalar_outer",[],[2],0),
("rank32_full",[1]*32,[1]*32,32),
("rank32_output",[1]*32,[1]*32,16)]
def mathematical(a,b,dims):
    if type(dims)is int:
        aa=list(range(a.ndim-dims,a.ndim));bb=list(range(dims))
    else:
        aa=[x%a.ndim for x in dims[0]];bb=[x%b.ndim for x in dims[1]]
    af=[i for i in range(a.ndim)if i not in aa];bf=[i for i in range(b.ndim)if i not in bb]
    output_shape=tuple(a.shape[i]for i in af)+tuple(b.shape[i]for i in bf)
    contract_shape=[max(a.shape[i],b.shape[j])for i,j in zip(aa,bb)]
    out=np.zeros(output_shape,dtype=np.float64)
    for coord in itertools.product(*(range(x)for x in output_shape)):
        for contracted in itertools.product(*(range(x)for x in contract_shape)):
            ai=[0]*a.ndim;bi=[0]*b.ndim
            for index,value in zip(af,coord[:len(af)]):ai[index]=value
            for index,value in zip(bf,coord[len(af):]):bi[index]=value
            for i,j,value in zip(aa,bb,contracted):
                ai[i]=0 if a.shape[i]==1 else value
                bi[j]=0 if b.shape[j]==1 else value
            out[coord]+=float(a[tuple(ai)])*float(b[tuple(bi)])
    return out
def primitives(p,a,b):
    arrays={"a":a,"b":b}
    for step in p["steps"]:
        if step["op"]=="matmul":arrays["result"]=arrays["a"]@arrays["b"]
        else:
            name=step["operand"];x=arrays[name];attrs=step.get("attrs",{})
            if step["op"]=="expand":x=np.broadcast_to(x,attrs["shape"])
            elif step["op"]=="permute":x=np.transpose(x,attrs["axes"])
            elif step["op"]=="reshape":x=np.reshape(x,attrs["shape"])
            elif step["op"]=="clone":x=np.array(x,copy=True)
            else:raise AssertionError("unknown_primitive")
            arrays[name]=x
    return arrays["result"]
rows=[];checks=0
for ordinal,(name,ash,bsh,dims)in enumerate(cases):
    p=plan(ash,bsh,dims,resource_profile="general-tensors-v1")
    check_plan(p,ash,bsh,dims,resource_profile="general-tensors-v1")
    maximum=0.
    for mode in ("zeros","signed","random","boundary"):
        rng=np.random.default_rng(2026101300+ordinal)
        def values(sh,offset):
            if mode=="zeros":return np.zeros(sh,dtype=np.float64)
            if mode=="signed":return ((np.arange(np.prod(sh,dtype=int))%5)-2).astype(np.float64).reshape(sh)/3+offset
            if mode=="boundary":return np.full(sh,.75-offset,dtype=np.float64)
            return rng.uniform(-.5,.5,sh).astype(np.float64)
        a,b=np.asarray(values(ash,.03),dtype=np.float64).reshape(tuple(ash)),np.asarray(values(bsh,-.05),dtype=np.float64).reshape(tuple(bsh))
        original=torch.tensordot(torch.from_numpy(a),torch.from_numpy(b),dims=dims).numpy()
        independent=mathematical(a,b,dims);lowered=primitives(p,a,b)
        np.testing.assert_allclose(original,independent,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(original,lowered,atol=1e-12,rtol=1e-12)
        assert tuple(original.shape)==tuple(p["output_shape"])
        maximum=max(maximum,float(np.max(np.abs(original-lowered))));checks+=1
    rows.append(dict(case=name,status="passed",input_shapes=[ash,bsh],dims=dims,
        output_shape=p["output_shape"],plan_binding=p["binding"],groups=4,maximum_absolute_difference=maximum))
reject=[
([2],[2],-1),([2],[2],2),([2],[2],True),([2],[2],"1"),
([2],[2],[[],[0]]),([2],[2],[[0,0],[0,0]]),([2],[2],[[1],[0]]),
([2],[2],[[False],[0]]),([2,3],[4,2],1),([0],[1],0),
([2**40],[1],0),([1]*33,[1],1),([1]*20,[1]*20,0),
([65536,1],[65536,65536],1)]
for a,b,dims in reject:
    try:plan(a,b,dims,resource_profile="general-tensors-v1")
    except ValueError:pass
    else:raise AssertionError("rejection_case_accepted")
p=plan([1],[1],1,resource_profile="general-tensors-v1");altered=copy.deepcopy(p)
altered["matrix_shapes"][0][0]=True
try:check_plan(altered,[1],[1],1,resource_profile="general-tensors-v1")
except ValueError:pass
else:raise AssertionError("boolean_identity_confusion")
# A correct hash for a different plan is not authority for these declared inputs.
altered=copy.deepcopy(p);altered["a_axes"]=[]
altered["binding"]=q.digest({k:v for k,v in altered.items()if k!="binding"})
try:check_plan(altered,[1],[1],1,resource_profile="general-tensors-v1")
except ValueError:pass
else:raise AssertionError("rebound_plan_tamper")
record=dict(format="poseidon-tensor-contraction-prototype-v1",status="passed",cases=rows,contexts=len(rows),
    reference_groups=checks,rejection_cases=len(reject),tamper_cases=2,
    python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
    torch_tensor_tensordot_available=hasattr(torch.Tensor,"tensordot"),
    implementation={p.name:q.sha(p)for p in HERE.glob("*.py")},seconds=time.monotonic()-started,
    reference_tolerance=dict(atol=1e-12,rtol=1e-12),production_active=False,
    source_capture_validated=False,api_calls=0,compiler_calls=0,encrypted_calls=0,
    upstream_source="https://github.com/pytorch/pytorch/blob/v2.0.1/torch/functional.py")
record["binding"]=q.digest(record);q.write(HERE/"report.json",record)
print(json.dumps({k:v for k,v in record.items()if k not in("cases","implementation")}))
