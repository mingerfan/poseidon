"""Bounded layout admission only, preserving every original model."""
import json,pathlib,sys,os,shlex,copy
HERE=pathlib.Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import ROOT,VENV,enter_nix
if "--inside"not in sys.argv:
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=120,keep_env=(),environment="poseidon-cpu"))
from application_policy import prepare
from unified_graph_lowering import candidate_source
from candidate_contract import validate_candidate
from component_contract import reconstruct_request
plan=json.loads((ROOT/"docs/baseline/agent-spatial-offline-proposal-r1396.json").read_text())
rows=[]
for case in plan["cases"]:
    if case["preflight_status"]=="passed":continue
    model=json.loads(pathlib.Path(case["model"]).read_text());attempts=[]
    for period in(8,16,32,64,128,256):
        options=copy.deepcopy(plan["options"]);options["chunk_period"]=period
        try:
            request,_=prepare(model,"poseidon_cpu","numerical",options)
            assert reconstruct_request(request)==request
            source,lowering=candidate_source(request)
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
            attempts.append(dict(period=period,status="static_passed",request_id=request["request_id"],source_bytes=len(source.encode())))
            break
        except(ValueError,TypeError,KeyError)as e:attempts.append(dict(period=period,status="blocked",error=str(e)))
    rows.append(dict(case=case["case"],original_failure=case["preflight_failure"],attempts=attempts))
with(HERE/"layout-admission-r1397.json").open("x")as f:json.dump(dict(rows=rows,compiler_calls=0,encrypted_calls=0,provider_calls=0),f,indent=2)
print(json.dumps(rows))
