"""Bounded independent reference and retained-provenance audit.
Runs after the frozen capacity batch; no API, compiler or encrypted execution.
"""
import sys,json,time,shlex,hashlib,resource,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
SCOPE=BASE/"benchmarks/tools/generic_capacity_backend_v10"
sys.path[:0]=[str(BASE),str(SCOPE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import qualification_scope as q
import run as live

def check_scope():
    p=q.read(HERE/"scope.json");q.bound(p)
    if p["format"]!="poseidon-generic-capacity-reference-scope-v1":
        raise ValueError("Closed reference scope")
    for path,want in p["frozen_files"].items():
        if q.sha(path)!=want:raise ValueError("Reference dependency changed:"+path)
    original=q.verify("live")
    if original["binding"]!=p["live_plan_binding"]:raise ValueError("Live plan changed")
    root=Path(original["campaign"])
    if not (root/"terminal.json").is_file():raise ValueError("Live terminal required")
    calls=q.calls(q.states(root),52,208)
    if not calls["calls_resolved"]:raise ValueError("Unsettled HTTP")
    return p

if "--inside" not in sys.argv:
    live.no_active()
    check_scope()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=480,environment="poseidon-cpu"))

resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,155))
resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
import numpy as np
import torch
sys.path.insert(0,str(BASE/"benchmarks/tools/generic_view_normalization_v1"))
from normalize import normalize
from benchmark_math import evaluate as math_evaluate
from benchmark_torch import Model
from candidate_bundle import check_provenance
p=check_scope();started=time.monotonic();rows=[]

def boundary():
    if time.monotonic()-started>=470:raise ValueError("Reference audit deadline")
    if live.measure(live.RESULTS)>48*1024**3:raise ValueError("Global artifact budget")
    if shutil.disk_usage(live.RESULTS).free<4*1024**3:raise ValueError("Disk reserve")
    if live.measure(HERE)>128*1024**2:raise ValueError("Reference artifact budget")

def flatten(model,outputs):
    return np.concatenate([np.asarray(outputs[o["name"]],dtype=np.float64).reshape(-1)
        for o in model["outputs"]])

def probe(model,index):
    result={}
    for number,d in enumerate(model["inputs"]):
        count=int(np.prod(d["shape"]));k=np.arange(count,dtype=np.float64)
        if index==0:a=np.zeros(count,dtype=np.float64)
        elif index==1:a=((k%17)-8)/40
        elif index==3:a=np.where((k+number)%2,-.8,.8)
        else:
            rng=np.random.default_rng(20261013+101*index+number)
            a=rng.uniform(-.2,.2,count).astype(np.float64)
            if index>=4:
                a[(index*13+number)%count]+=.071*(1 if index%2 else -1)
        result[d["name"]]=a.reshape(d["shape"])
    return result

def torch_evaluate(module,model,inputs):
    with torch.no_grad():
        out=module(*(torch.from_numpy(inputs[d["name"]].copy())for d in model["inputs"]))
    return {d["name"]:v.detach().numpy().copy()for d,v in zip(model["outputs"],out)}

def metric(a,b):
    if a.shape!=b.shape or not np.all(np.isfinite(a))or not np.all(np.isfinite(b)):
        raise ValueError("Reference shape/finite")
    delta=np.abs(a-b);ratio=delta/(1e-12+1e-12*np.abs(a))
    if not np.all(ratio<=1):raise ValueError("Independent reference mismatch")
    return dict(max_absolute_error=float(np.max(delta)),max_tolerance_ratio=float(np.max(ratio)))

for item in p["models"]:
    boundary()
    model=q.read(item["model"]);normalized,record=normalize(model)
    old_module=Model(model).eval();new_module=Model(normalized).eval()
    names=("original_math_vs_torch","original_math_vs_normalized_math","original_math_vs_normalized_torch")
    maximum={name:dict(max_absolute_error=0.,max_tolerance_ratio=0.)for name in names};values=0
    for i in range(16):
        boundary();inputs=probe(model,i)
        a=flatten(model,math_evaluate(model,inputs))
        comparisons=(flatten(model,torch_evaluate(old_module,model,inputs)),
            flatten(normalized,math_evaluate(normalized,inputs)),
            flatten(normalized,torch_evaluate(new_module,normalized,inputs)))
        for name,b in zip(names,comparisons):
            m=metric(a,b)
            for key,value in m.items():maximum[name][key]=max(maximum[name][key],value)
        values+=a.size
    rows.append(dict(case=item["case"],status="passed",groups=16,values=values,
        original_sha256=q.digest(model),normalized_sha256=q.digest(normalized),
        normalization=record,comparisons=maximum,
        original_input_output_constant_declarations_preserved=True))
boundary()
sys.path.insert(0,str(BASE/"benchmarks/tools/generic_provenance_dag_v1"))
from codec import decode
provenance=[]
for item in p["provenance_packets"]:
    boundary();packet=q.read(item["packet"]);value=decode(packet)
    model=q.read(item["model"]);binding=check_provenance(value,model)
    provenance.append(dict(case=item["case"],status="passed",packet_binding=packet["binding"],
        logical_bytes=packet["logical_bytes"],wire_bytes=Path(item["packet"]).stat().st_size,
        capture_binding=binding["binding_sha256"],trusted_weight_arrays_checked=True,
        source_capture_semantics_checked=True,production_intake_accepted=False,
        production_wire_and_logical_gate_bytes=4*1024**2,production_active=False))
boundary()
result=dict(format="poseidon-generic-capacity-independent-reference-audit-v1",status="passed",
    scope_binding=p["binding"],live_plan_binding=p["live_plan_binding"],cases=rows,
    total_models=len(rows),input_groups=sum(r["groups"]for r in rows),
    compared_values_per_comparison=sum(r["values"]for r in rows),comparison_paths=3,
    reference_threshold=dict(atol=1e-12,rtol=1e-12),
    probe_policy="new deterministic normalization probes: zero/signed/seeded-random/boundary plus 12 asymmetric probes",
    historical_reference_or_inputs_replaced=False,provenance=provenance,
    runtime=dict(python=sys.version_info[:3],numpy=np.__version__,torch=torch.__version__,
        cuda=torch.version.cuda,hip=torch.version.hip),wall_seconds=time.monotonic()-started,
    provider_calls=0,compiler_calls=0,encrypted_calls=0,production_changes=False)
result["binding"]=q.digest(result)
q.write(HERE/"report.json",result)
print(json.dumps({k:v for k,v in result.items()if k not in("cases","provenance")}))
print("provenance",json.dumps(provenance))
