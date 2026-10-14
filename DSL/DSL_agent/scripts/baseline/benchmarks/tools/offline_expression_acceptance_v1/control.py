"""Bounded approval and identity checks for the single r1286 expression."""
import hashlib,json,math,re
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
BASE=ROOT/"scripts/baseline"
PROPOSAL=ROOT/"docs/baseline/agent-ordered-common-expression-offline-proposal-r1286.json"
APPROVAL=ROOT/"docs/baseline/agent-ordered-common-expression-offline-approval-r1286.json"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
RUN=RESULTS/"agent-ordered-common-expression-runtime-r1286"
BINDING="40d2cd267a5de4a169b9d7fb0ccc0ea5062e2375ae87a6d7fda2a1432312b3de"
BACKEND_REPORT_SHA256="45f5f03583439068e5c2d6e58d1b74e7f6e784fe9fdd96896f3b7713b0c47b41"
SCOPE=dict(artifact_bytes=512*1024**2,backend_changes=0,cache_execution=False,compile_jobs=2,
 compiler_attempts_max=1,downloads=0,encrypted_phases_max=1,fixed_input_groups=4,
 global_bytes=32*1024**3,http_attempts=0,installations=0,link_jobs=1,native_workers=1,
 new_input_execution=False,parameter_changes=0,program_export=True,provider_calls=0,
 qualification_seconds=180,readonly_package_reload=True,reserve_bytes=4*1024**3,wall_seconds=900)
def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):h.update(block)
    return h.hexdigest()
def read(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=4*1024**2,"Regular bounded control JSON")
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"Duplicate control JSON field");result[key]=value
        return result
    def invalid(_):raise ValueError("Nonfinite control JSON")
    return json.loads(p.read_text(),object_pairs_hook=pairs,parse_constant=invalid)
def write(path,value):
    with Path(path).open("xb") as stream:stream.write(canonical(value))
def check_scope(scope):
    require(type(scope) is dict and set(scope)==set(SCOPE) and
            all(type(scope[k]) is type(v) and scope[k]==v for k,v in SCOPE.items()),"Exact approved offline scope")
def check_approval(proposal,receipt,proposal_sha256,driver_hashes):
    """Pure validator; tests use synthetic data, never an authoritative receipt."""
    require(type(proposal) is dict and proposal.get("format")=="poseidon-manual-expression-offline-proposal-v1",
            "Original offline proposal version")
    require(proposal.get("binding")==BINDING and
            digest({k:v for k,v in proposal.items() if k!="binding"})==BINDING,"Original proposal binding")
    check_scope(proposal["scope"])
    fields={"format","approved","proposal_binding","proposal_sha256","scope","driver_hashes","human_message"}
    require(type(receipt) is dict and set(receipt)==fields and
            receipt["format"]=="poseidon-manual-expression-offline-approval-v1" and
            receipt["approved"] is True,"Explicit separate human approval receipt")
    require(receipt["proposal_binding"]==BINDING and receipt["proposal_sha256"]==proposal_sha256,
            "Approval names the exact proposal")
    check_scope(receipt["scope"])
    require(type(driver_hashes) is dict and {"control.py","run.py"}<=set(driver_hashes) and
            len(driver_hashes)<=8 and receipt["driver_hashes"]==driver_hashes and
            all(type(k) is str and Path(k).name==k and k.endswith(".py") and
                type(v) is str and re.fullmatch("[0-9a-f]{64}",v)
                for k,v in driver_hashes.items()),"Approved control tool identity")
    require(type(receipt["human_message"]) is str and 1<=len(receipt["human_message"])<=1024,
            "Human approval message required")
    require(proposal["exclusions"]==["paid_r1232","fused_matrix_V14_backend_qualification",
                                    "backend_or_parameter_modifications"],"Other requests are excluded")
    return dict(proposal_binding=BINDING,scope=dict(SCOPE),paid_calls=0)
def expected_backend(proposal):
    path=Path(proposal["historical_failure"]["compiler_log"]).parents[1]/"report.json"
    require(path==RESULTS/"candidate-replay-qm6k630_/report.json" and
            sha(path)==BACKEND_REPORT_SHA256,"Historical native backend identity changed")
    value=read(path)["backend_identity"]
    require(value.get("format")=="poseidon-backend-identity-v2","Native backend identity version")
    return value
def check_backend(actual,expected):
    require(type(actual) is dict and canonical(actual)==canonical(expected),
            "Exact SDK binary/source/parameter/build identity required")
def local_driver_hashes():
    return {p.name:sha(p) for p in sorted(HERE.glob("*.py"))}
def check_sources(proposal):
    require(proposal["root"]==str(ROOT) and proposal["branch"]=="lhy-agent-dsl","Approved development checkout")
    require(len(proposal["source_hashes"])==704 and len(proposal["protected_sources"])==654 and
            len(proposal["helper_hashes"])==3,"Original source/protected/helper inventory")
    for label in ("source_hashes","protected_sources"):
        for relative,want in proposal[label].items():
            p=Path(relative)
            require(not p.is_absolute() and ".." not in p.parts and
                    sha(ROOT/p)==want,"Frozen production/backend/reference changed")
    for path,want in proposal["helper_hashes"].items():
        p=Path(path);require(p.is_relative_to(ROOT) and sha(p)==want,"Original expression helper changed")
    for key in ("historical_failure","plaintext_probe"):
        v=proposal[key];require(sha(v["report"])==v["sha256"],"Original failure/plaintext evidence changed")
    entry=proposal["entry"]
    require(set(entry)=={"api","candidate","request","validation_level"} and
            entry["api"]=="component_backend.qualify" and entry["validation_level"]=="numerical",
            "Existing numerical qualification entry")
    expected_backend(proposal)
    request=read(entry["request"])
    require(sha(entry["request"])==proposal["request_file_sha256"] and
            digest(request)==proposal["request_canonical_sha256"] and
            request["request_id"]==proposal["request_id"] and
            digest(request["model"])==proposal["model_sha256"] and
            digest(request["public_constants"])==proposal["public_constants_sha256"] and
            request["compiler_configuration"]==proposal["compiler_configuration"],
            "Original model/weights/layout/configuration request")
    require(sha(entry["candidate"])==proposal["candidate_sha256"] and
            sha(proposal["transport_request"])==proposal["transport_request_sha256"],
            "Original manual candidate and transport")
    require(proposal["tolerance"]=={"atol":1e-5,"rtol":1e-4},"Frozen tolerance")
    return request
def approved_plan():
    # This order is intentional: no Nix/SDK import, lock or output creation on denial.
    require(APPROVAL.is_file() and not APPROVAL.is_symlink(),"Pending r1286 human approval")
    proposal,receipt=read(PROPOSAL),read(APPROVAL)
    check_approval(proposal,receipt,sha(PROPOSAL),local_driver_hashes())
    request=check_sources(proposal)
    return proposal,receipt,request
def fresh_result(path=RUN):
    require(Path(path)==RUN and RUN.parent==RESULTS and not RUN.exists() and not RUN.is_symlink(),
            "Single-use original offline execution; preserve any previous attempt")
def check_plan(plan):
    require(type(plan) is dict and plan.get("format")=="poseidon-manual-expression-runtime-plan-v1" and
            digest({k:v for k,v in plan.items() if k!="binding"})==plan.get("binding"),
            "Controller plan binding")
    check_scope(plan["scope"])
    proposal,receipt,request=approved_plan()
    import sys
    sys.path.insert(0,str(BASE))
    from poseidon_backend_identity import identity
    check_backend(identity(),expected_backend(proposal))
    require(plan["proposal_sha256"]==sha(PROPOSAL) and plan["approval_sha256"]==sha(APPROVAL)
            and plan["driver_hashes"]==local_driver_hashes(),"Controller/approval/source identity")
    require(type(plan["deadline_wall"]) in (int,float) and math.isfinite(plan["deadline_wall"]) and
            type(plan["started_wall"]) in (int,float) and math.isfinite(plan["started_wall"]) and
            plan["deadline_wall"]-plan["started_wall"]==SCOPE["wall_seconds"],"Original finite deadline")
    return proposal,receipt,request
def added_directories(plan):
    originals=set(plan["initial_entries"])
    return [p for p in RESULTS.iterdir() if p.name not in originals]
def measure(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())
def resource_boundary(plan):
    import shutil,time
    require(time.time()<plan["deadline_wall"],"Original offline deadline exhausted")
    new=sum(measure(p) for p in added_directories(plan) if p.is_dir())
    require(new<=SCOPE["artifact_bytes"],"Original new artifact bound")
    require(plan["initial_bytes"]+new<=SCOPE["global_bytes"],"Global artifact bound")
    require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"Disk reserve")
    return new
