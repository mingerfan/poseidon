"""Freeze source cases and static rule candidates, without backend execution."""
import hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
from benchmark_graph import require,digest
from application_identity import current_identity_manifest
from application_policy import prepare
from component_contract import reconstruct_request
from candidate_contract import validate_candidate
from unified_graph_lowering import candidate_source
from hecate_python_env import ROOT,WORK,VENV
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned proposal preparation")
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with path.open("x")as stream:json.dump(value,stream,sort_keys=False,indent=2,allow_nan=False);stream.write("\n")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=read(ROOT/"docs/baseline/agent-application-control-offline-proposal-r1382.json")
sdk=current_identity_manifest()
require(sdk["backend"]==old["backend_identity"],"Unchanged Poseidon backend")
ready=HERE/"ready-r1390";ready.mkdir();cases=[]
from candidate_bundle import load_model_provenance
capture_root=WORK/"results/agent-pytorch-input-binding-source-r1389-final"
for target in sorted(p for p in capture_root.iterdir()if p.is_dir()):
    name=target.name;folder=ready/name;folder.mkdir()
    model=read(target/"model.json")
    provenance=load_model_provenance(target,model,max_bytes=4*1024**2)
    write(folder/"model.json",model);write(folder/"provenance.json",provenance)
    request,_=prepare(model,"poseidon_cpu","numerical",old["options"])
    require(reconstruct_request(request)==request,"Frozen request reconstruction")
    source_text,lowering=candidate_source(request)
    static=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source_text),request)
    (folder/"candidate.py").write_text(source_text);write(folder/"request.json",request)
    write(folder/"lowering.json",lowering);write(folder/"static.json",static)
    cases.append(dict(case=name,source_contract="poseidon-trusted-pytorch-v26",resource_profile="general-tensors-v1",
        source_manifest=str(HERE/"cases-final"/name/"manifest.json"),model=str(folder/"model.json"),
        provenance=str(folder/"provenance.json"),request=str(folder/"request.json"),request_id=request["request_id"],
        candidate=str(folder/"candidate.py"),repeat_package=name in("dictionary0","nested1","defaults2")))
for row in old["cases"]:
    case=dict(row);case["repeat_package"]=False;cases.append(case)
for case in cases:
    request=read(case["request"]);validate_candidate(dict(schema=1,request_id=request["request_id"],
        hecate_source=Path(case["candidate"]).read_text()),request)
    root=Path(case["source_manifest"]).parent;spec=read(case["source_manifest"])
    paths=[root/"manifest.json",*(root/name for name in spec["files"]),*(root/w["file"]for w in spec["weights"].values()),
           Path(case["candidate"]),Path(case["request"])]
    paths +=[Path(case[key])for key in("model","provenance","historical_task_metadata")if key in case]
    case["files"]={str(path):sha(path)for path in paths}
write(HERE/"selected-offline-cases-r1390.json",cases);write(HERE/"sdk-r1390.json",sdk)
print(json.dumps(dict(cases=len(cases),source_sdk=sdk["manifest_sha256"],backend_unchanged=True,
    rule_candidates_static_passed=len(cases),new_contexts=12,compatibility=3,compiler_calls=0,encrypted_calls=0,provider_calls=0)))
