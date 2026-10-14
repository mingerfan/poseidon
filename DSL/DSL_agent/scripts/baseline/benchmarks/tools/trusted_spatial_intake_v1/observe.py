"""Pinned component checks only; no compiler, encryption or model API."""
import sys,json,hashlib,os,shlex,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*(["--baseline"] if "--baseline" in sys.argv else [])])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
MODE="baseline"if "--baseline"in sys.argv else "stage"
if MODE=="stage":sys.path.insert(0,str(HERE/"stage"))
sys.path.insert(0,str(HERE))
import torch,numpy as np,resource,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
from fixtures import cases
from pytorch_ordered_capture import prepare_module
from benchmark_graph import samples,digest
from benchmark_math import evaluate as core
from benchmark_torch import evaluate as independent_torch
from operator_arithmetic_graph import evaluate_reference
assert torch.__version__=="2.0.1"and np.__version__=="1.25.2"
RESULT=HERE/("baseline-observation-r1401"if MODE=="baseline"else "stage-observation-r1401")
RESULT.mkdir()
rows=[];maximum=0.;groups=values=0
for name,module,inputs,outputs in cases():
    if MODE=="baseline"and name not in ("resize_1_linear_0","resize_2_bilinear_0","transpose_1_0"):continue
    row=dict(case=name,status="running");rows.append(row)
    try:
        kwargs={}if MODE=="baseline"else dict(spatial_source_policy="poseidon-pytorch-static-spatial-v1")
        high,lowered,binding,context=prepare_module(module,inputs,outputs,resource_profile="general-tensors-v1",**kwargs)
        assert MODE=="stage","Unexpected public baseline acceptance"
        assert high["format"]=="poseidon-operator-graph-v4"
        local=0.;count=0
        for data in samples(high,16):
            with torch.no_grad():
                actual=module(*(torch.from_numpy(data[i["name"]]).clone()for i in inputs))
                normalized=context["normalized_module"](*(torch.from_numpy(data[i["name"]]).clone()for i in inputs))
            def named(value):
                seq=value if type(value)in(tuple,list)else[value]
                return dict(zip(outputs,[v.detach().numpy()for v in seq]))
            expected=evaluate_reference(high,data)
            for calculated in (named(actual),named(normalized),core(lowered,data),independent_torch(lowered,data)):
                assert set(calculated)==set(expected)
                for key,result in expected.items():
                    diff=np.abs(calculated[key]-result)
                    assert np.all(diff<=2e-12+2e-12*np.abs(result)),name
                    local=max(local,float(diff.max()))
            count+=sum(v.size for v in expected.values())
        row.update(status="passed",reference_groups=16,logical_outputs=count,maximum_difference=local,
            original_graph_sha256=digest(high),lowered_graph_sha256=digest(lowered),decomposition_binding=binding["binding_sha256"])
        folder=RESULT/name;folder.mkdir()
        for file,item in (("original.json",high),("lowered.json",lowered),("binding.json",binding)):
            with(folder/file).open("x")as f:json.dump(item,f,sort_keys=True,indent=2,allow_nan=False)
        maximum=max(maximum,local);groups+=16;values+=count
    except Exception as error:
        row.update(status="rejected"if MODE=="baseline"else"failed",error_type=type(error).__name__,error=str(error))
        with(RESULT/(name+"-failure.txt")).open("x")as f:f.write(traceback.format_exc())
    print(json.dumps(row),flush=True)
report=dict(format="poseidon-spatial-pytorch-draft-observation-v1",mode=MODE,rows=rows,
    passed=sum(x["status"]=="passed"for x in rows),rejected=sum(x["status"]=="rejected"for x in rows),
    failed=sum(x["status"]=="failed"for x in rows),probe_groups=groups,logical_outputs=values,maximum_difference=maximum,
    implementation={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (HERE/"stage").iterdir()if p.is_file()and not p.is_symlink()},
    production_registered=False,api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report)
with(RESULT/"report.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in report.items()if k not in("rows","implementation")}))
raise SystemExit(1 if report["failed"]else 0)
