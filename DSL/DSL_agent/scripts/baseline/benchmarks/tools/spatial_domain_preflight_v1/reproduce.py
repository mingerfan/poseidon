"""Before-change evidence for valid spatial/approximation model rejection."""
import hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(HERE),str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
import numpy as np
from fixtures import models
from operator_arithmetic_graph import decompose,evaluate_reference
from execution_input_domain import check_approximation_domains
from benchmark_graph import digest
rows=[]
cases=models()
for name,model in cases.items():
    decompose(model,resource_profile="general-tensors-v1")
    value=.5 if name=="transpose2_nonlinear"else .25
    batch=dict(x=np.full(model["inputs"][0]["shape"],value,dtype=np.float64))
    reference=evaluate_reference(model,batch,ideal=False)
    assert np.isfinite(reference["out"]).all()
    try:check_approximation_domains(model,[batch])
    except ValueError as error:row=dict(case=name,status="reproduced",error_type=type(error).__name__,reason=str(error))
    else:row=dict(case=name,status="not_reproduced")
    rows.append(dict(row,model_sha256=digest(model),independent_reference_finite=True))
report=dict(format="poseidon-spatial-domain-reproduction-v1",cases=rows,reproduced=sum(r["status"]=="reproduced"for r in rows),
 source_checker_sha256=hashlib.sha256((BASE/"execution_input_domain.py").read_bytes()).hexdigest(),
 compiler_calls=0,encrypted_calls=0,provider_calls=0)
report["binding"]=digest(report)
with(HERE/"reproduction.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:report[k]for k in("binding","reproduced","cases")}),flush=True)
raise SystemExit(not(report["reproduced"]==6))
