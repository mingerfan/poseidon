"""Pinned component checks only; no compiler, encryption or model API."""
import sys,json,hashlib,os,shlex,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
import numpy as np,torch,importlib.util
from benchmark_graph import samples,digest
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from operator_arithmetic_graph import decompose,evaluate_reference
from operator_spatial_graph import FORMAT,implementation_sources
assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
reference=import_file("spatial_original_reference",BASE/"benchmarks/tools/static_spatial_decomposition_v1/reference.py")
modules=["test_operator_spatial_graph","test_operator_arithmetic_graph","test_operator_graph_v2","test_model_decomposition"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in modules)
result=unittest.TextTestRunner(verbosity=2).run(suite)
controls=dict(passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),modules=modules)
with(HERE/"unit-controls-final.json").open("x")as f:json.dump(controls,f,indent=2)
if not result.wasSuccessful()or result.skipped:raise SystemExit(1)
rows=[];groups=values=0;maximum=0.;errors=0.;comparisons=0
(HERE/"models").mkdir();(HERE/"lowered").mkdir()
for old in sorted((BASE/"benchmarks/tools/static_spatial_decomposition_v1/models").glob("*.json")):
    model=json.loads(old.read_text());model["format"]=FORMAT
    lowered,binding=decompose(model,resource_profile="general-tensors-v1")
    local=0.;outputs=0
    for inputs in samples(model,16):
        expected=evaluate_reference(model,inputs)
        candidates=[reference.evaluate(model,inputs,torch_reference=True),core_math(lowered,inputs),core_torch(lowered,inputs)]
        for name,x in expected.items():
            outputs+=x.size
            for other in candidates:
                diff=np.abs(x-other[name]);assert np.all(diff<=2e-12+2e-12*np.abs(x))
                local=max(local,float(diff.max()));errors+=float(diff.sum());comparisons+=x.size
    maximum=max(maximum,local);groups+=16;values+=outputs
    for folder,item in(("models",model),("lowered",lowered)):
        with(HERE/folder/(model["id"]+".json")).open("x")as f:json.dump(item,f,sort_keys=True,indent=2)
    rows.append(dict(id=model["id"],original_sha256=digest(model),lowered_sha256=digest(lowered),
        binding=binding,probe_groups=16,logical_outputs=outputs,maximum_difference=local,compiler=False,encrypted=False,agent=False))
report=dict(format="poseidon-spatial-component-math-v1",status="passed",model_contexts=len(rows),
    probe_groups=groups,logical_outputs=values,maximum_difference=maximum,mae=errors/comparisons,
    rows=rows,unit_controls=controls,python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
    implementation_sources=implementation_sources(),api_calls=0,compilations=0,encrypted_calls=0,
    active_graph_format=FORMAT,active_pytorch_format=None,runtime_layout_validated=False,original_benchmark_changed=False)
report["binding"]=digest(report)
with(HERE/"component-math-results.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:v for k,v in report.items()if k not in("rows","implementation_sources")}))
