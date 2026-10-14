"""Run only mathematics/shape checks in the existing pinned environment."""
import hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside" not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
import numpy as np,torch
from benchmark_graph import samples,digest,signature
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from cases import models
from draft import decompose
from reference import evaluate
from test_draft import SpatialTests
assert os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpatialTests))
status=dict(format="poseidon-spatial-draft-test-result-v1",passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
            active_agent_capability=False,api_calls=0,compiler_calls=0,encrypted_calls=0)
with (HERE/"unittest-result.json").open("x")as f:json.dump(status,f,indent=2);f.write("\n")
if not result.wasSuccessful():raise SystemExit(1)
(HERE/"models").mkdir();(HERE/"lowered").mkdir()
rows=[];signatures=set();topologies=set();groups=values=0;maximum=0.;total=0.;comparisons=0
for m in models():
    lowered,binding=decompose(m);local=0.;local_values=0
    for batch in samples(m,16):
        expected=evaluate(m,batch)
        candidates=[evaluate(m,batch,torch_reference=True),core_math(lowered,batch),core_torch(lowered,batch)]
        for name,reference in expected.items():
            local_values+=reference.size
            for actual in candidates:
                error=np.abs(reference-actual[name])
                assert np.all(error<=2e-12+2e-12*np.abs(reference))
                local=max(local,float(error.max()));total+=float(error.sum());comparisons+=reference.size
    groups+=16;values+=local_values;maximum=max(maximum,local)
    for folder,value in (("models",m),("lowered",lowered)):
        with (HERE/folder/(m["id"]+".json")).open("x")as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
    sig=signature(lowered);top=signature(lowered,topology=True);signatures.add(sig);topologies.add(top)
    rows.append(dict(id=m["id"],original_sha256=digest(m),lowered_sha256=digest(lowered),
                     binding=binding,groups=16,compared_values=local_values,maximum_reference_difference=local,
                     agent_generated=False,compiled=False,encrypted_execution=False))
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
         [HERE/"draft.py",HERE/"cases.py",HERE/"reference.py",HERE/"test_draft.py",Path(__file__),
          *(BASE/n for n in ("model_decomposition.py","model_resources.py","benchmark_graph.py","benchmark_math.py","benchmark_torch.py"))]}
report=dict(format="poseidon-spatial-draft-math-acceptance-v1",status="passed",models=len(rows),
            model_probe_groups=groups,model_output_values=values,reference_comparisons=comparisons,
            maximum_reference_difference=maximum,mae=total/comparisons,
            lowered_structure_signatures=len(signatures),lowered_topology_groups=len(topologies),
            original_frozen_benchmark_changed=False,torch_version=str(torch.__version__),numpy_version=np.__version__,
            python=sys.version.split()[0],torch_cuda=torch.version.cuda,source_hashes=sources,rows=rows,
            unit_tests=status,elapsed_seconds=time.monotonic()-started,
            active_agent_capability=False,runtime_layout_validated=False,agent_generated=False,
            api_calls=0,compiler_calls=0,encrypted_calls=0)
report["binding"]=digest(report)
with (HERE/"math-acceptance.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:report[k]for k in ("status","models","model_probe_groups","model_output_values",
"maximum_reference_difference","lowered_structure_signatures","lowered_topology_groups","binding","elapsed_seconds")}))
