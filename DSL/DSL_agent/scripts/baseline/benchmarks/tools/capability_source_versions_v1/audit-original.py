"""Read-only request, declaration and stage binding audit; no backend execution."""
import sys, shlex, json, copy, hashlib, time, resource, re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
    from hecate_python_env import enter_nix,VENV
    command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
from benchmark_graph import require,digest,canonical
from application_backend import PoseidonBackend
from application_capabilities import input_sources
from component_contract import reconstruct_request
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=read(HERE/"sources-before.json");current=input_sources();comparable=copy.deepcopy(current)
del comparable["trusted_pytorch"]["static_broadcast"]
del comparable["trusted_pytorch"]["spatial_calls"]["versions"]
comparable["trusted_pytorch"]["input_call"]["versions"]=before["trusted_pytorch"]["input_call"]["versions"]
require(comparable==before,"unrelated_capability_declaration_changed")
scope=read(BASE/"benchmarks/tools/static_broadcast_intake_v1/prepared-source-scope.json")
rows=[]
for source in scope["sources"]:
    path=Path(source["request_file"]);request=read(path)
    require(sha(path)==source["request_file_sha256"],"old_request_file_changed")
    require(canonical(reconstruct_request(request))==canonical(request),"old_request_hash_changed")
    rows.append(dict(context=source["context"],request_id=request["request_id"],request_sha256=sha(path),unchanged=True))
before_identity=read(HERE/"identity-before.json");after_identity=PoseidonBackend().identity_manifest()
require(after_identity["backend"]==before_identity["backend"],"backend_or_parameter_identity_changed")
changed={}
for stage in ("generation","compilation","execution","validation"):
    old=before_identity[stage];new=after_identity[stage]
    paths=sorted(k for k in set(old["sources"])|set(new["sources"])if old["sources"].get(k)!=new["sources"].get(k))
    require(set(paths)<= {"application_capabilities.py"},"unexpected_stage_source_change")
    changed[stage]=dict(before_sha256=old["sha256"],after_sha256=new["sha256"],changed_paths=paths)
with(HERE/"identity-after.json").open("x")as stream:json.dump(after_identity,stream,sort_keys=True,indent=2)
tests=read(HERE/"controls-first.json")
require(tests["status"]=="passed"and tests["tests"]==168 and
        tests["failed"]==tests["errors"]==tests["skipped"]==0,"controls_failed")
log=(HERE/"controls-first.log").read_text()
ids=re.findall(r"^test[^ ]* \(([^)]+)\) \.\.\. ok$",log,re.M)
require(len(ids)==168,"control_log_case_count")
record=dict(format="poseidon-source-capability-consistency-audit-v1",status="passed",
    input_call_versions=current["trusted_pytorch"]["input_call"]["versions"],
    spatial_versions=current["trusted_pytorch"]["spatial_calls"]["versions"],
    static_broadcast=current["trusted_pytorch"]["static_broadcast"],
    other_capabilities_unchanged=True,old_source_requests=rows,
    backend_and_parameters_unchanged=True,
    before_sdk=before_identity["manifest_sha256"],after_sdk=after_identity["manifest_sha256"],
    stage_changes=changed,controls=dict(tests=168,unique_case_ids=len(set(ids)),failed=0,errors=0,skipped=0),
    evidence={str(p.relative_to(ROOT)):sha(p)for p in(HERE/"change.json",HERE/"controls-first.json",HERE/"controls-first.log",
        HERE/"sources-before.json",HERE/"identity-before.json",HERE/"identity-after.json",Path(__file__).resolve(),
        BASE/"application_capabilities.py",BASE/"test_application_source_versions.py")},
    source_capture_calls=0,api_calls=0,compiler_calls=0,encrypted_calls=0,
    new_backend_validation="not_run; prior v28 acceptance remains under its original identity")
record["binding"]=digest(record)
with(HERE/"consistency-audit.json").open("x")as stream:json.dump(record,stream,sort_keys=True,indent=2)
print(json.dumps(dict(status=record["status"],before_sdk=record["before_sdk"],after_sdk=record["after_sdk"],
    stage_changed_paths={k:v["changed_paths"]for k,v in changed.items()},old_request_count=len(rows),
    tests=168,unique_case_ids=len(set(ids)),api_calls=0,compiler_calls=0,encrypted_calls=0,
    binding=record["binding"]),ensure_ascii=False),flush=True)
