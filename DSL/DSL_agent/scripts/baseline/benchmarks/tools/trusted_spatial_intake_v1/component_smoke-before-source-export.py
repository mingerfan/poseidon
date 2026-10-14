"""Full staged provenance smoke; no compiler, encrypted runtime or Provider."""
import sys,json,shlex,traceback,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
sys.path.insert(0,str(HERE/"component-stage"));sys.path.insert(1,str(HERE))
import torch,numpy as np
from fixtures import cases
from pytorch_weight_package import export_state,read,binding
from pytorch_input_binding import CALL_FORMAT
from pytorch_public_expression_capture import prepare,wrapped_metadata,make_reference,check_reference
from pytorch_ordered_contract import validate_package,capture_binding,package_version
from benchmark_graph import digest
from candidate_bundle import check_provenance
from hecate_python_env import WORK
torch.set_num_threads(1)
out=WORK/"results"/("agent-spatial-source-stage-smoke-"+str(time.time_ns()));out.mkdir()
name,module,inputs,outputs=cases()[0]
source_text=(HERE/"fixtures.py").read_text()+"\\n"+"def build():return cases()[0][1]\\n"
source_text=source_text.replace("\\n","\n")
weights_path=out/"source";weights_path.mkdir();(weights_path/"models.py").write_text(source_text)
weights=export_state(module,weights_path,resource_profile="general-tensors-v1")
arrays,raw=read(weights_path,weights,resource_profile="general-tensors-v1");bound_weights=binding(arrays,weights)
spec=dict(format="poseidon-trusted-pytorch-v27",entry="models.build",factory=dict(args=[],kwargs={}),
    files={"models.py":hashlib.sha256(source_text.encode()).hexdigest()},weights=weights,
    inputs=inputs,outputs=outputs,dtype_policy="strict-float64-v1",
    approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[]),
    call=dict(format=CALL_FORMAT,args=[dict(kind="tensor",name=x["name"])for x in inputs],kwargs={}))
validate_package(spec,{"models.py":source_text})
capture=out/"capture";capture.mkdir()
try:
    record=prepare(module,spec,capture,expected_weight_binding=bound_weights,resource_profile="general-tensors-v1")
    metadata=wrapped_metadata(record)
    report=make_reference(record,spec,metadata,resource_profile="general-tensors-v1",expected_weight_arrays=arrays)
    check_reference(report,record["high_graph"],record["lowered_graph"],spec,resource_profile="general-tensors-v1",
        expected_weight_binding=bound_weights,expected_weight_arrays=arrays)
    lower,decomp=capture_binding(record["high_graph"],spec,resource_profile="general-tensors-v1",
        capture_metadata=metadata,expected_weight_binding=bound_weights,expected_weight_arrays=arrays)
    provenance=dict(original=record["high_graph"],decomposition=decomp,reference_report=report,
        trusted_pytorch=dict(format=package_version(spec),manifest=spec,files={"models.py":source_text},
                            weight_files=__import__("pytorch_ordered_contract").encode_weight_files(spec,raw,resource_profile="general-tensors-v1")))
    check_provenance(provenance,lower)
    (out/"provenance.json").write_text(json.dumps(provenance,allow_nan=False))
    print(json.dumps(dict(status="passed",case=name,result=str(out),finite=record["original_reference"],
                         backend_calls=0,api_calls=0)),flush=True)
except BaseException:
    (out/"failure.txt").write_text(traceback.format_exc());print("RESULT",out,flush=True);raise
